//! Live node-blind commitment-chain persistence and verification.
//!
//! This extension owns canonical block decoding, snapshot audits, atomic append
//! transactions, verified range serving, and legacy tip compatibility. It is
//! deliberately separate from checkpoint evidence, coordinator authority, and
//! sync telemetry so those domains cannot alter chain persistence semantics.
//!
//! [MEMCHAIN-COMMITMENT-CHAIN-SPLIT 2026-09-25 by Codex] Extracted from
//! storage_ops while preserving existing methods, SQL, sidecar behavior, and
//! public re-export paths.

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::info;

use aeronyx_core::ledger::{
    RecordCommitmentBlockV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH,
};

use super::storage::{MemoryStorage, RecordCommitmentIntegrityRuntime};
use super::storage_coordinator_authority::{
    read_record_coordinator_handover_history_transaction, record_commitment_authority_at_height,
    verify_record_commitment_proposer_history_transaction,
};
use super::storage_ops::{
    persist_record_commitment_tip_anchor, read_record_commitment_tip_transaction,
};

/// Result of atomically appending a V1 commitment block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordCommitmentAppendOutcome {
    /// The block extended the local verified tip.
    Inserted,
    /// The exact same block was already stored at that height.
    AlreadyPresent,
}

/// Aggregate result of one atomic bounded commitment-block append.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecordCommitmentBatchAppendOutcome {
    /// Blocks committed by this transaction.
    pub inserted: usize,
    /// Exact blocks that were already durable when the transaction began.
    pub already_present: usize,
}

/// One audit-gated block page and the canonical tip observed in the same
/// `SQLite` read snapshot.
///
/// The page contains only signed commitment blocks. It never contains memory
/// records, owners, payload plaintext, routes, endpoints, or client metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRecordCommitmentBlockPage {
    /// Canonically decoded and signature-verified blocks in height order.
    pub blocks: Vec<RecordCommitmentBlockV1>,
    /// Canonical chain tip height in the same snapshot.
    pub tip_height: u64,
    /// Canonical chain tip hash in the same snapshot.
    pub tip_hash: [u8; 32],
}

#[derive(Debug)]
struct CommittedRecordCommitmentBatch {
    outcome: RecordCommitmentBatchAppendOutcome,
    inserted_indices: Vec<usize>,
    appended_at: u64,
}

/// Result of a complete persisted commitment-chain integrity audit.
///
/// The report is aggregate and contains no commitment, owner, peer, endpoint,
/// payload, or routing metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordCommitmentChainAudit {
    /// Number of fully verified persisted blocks.
    pub block_count: u64,
    /// Number of commitments verified against the membership index.
    pub commitment_count: u64,
    /// Last verified height, or zero for an empty chain.
    pub tip_height: u64,
}

struct StoredRecordCommitmentBlockRow {
    height: i64,
    block_hash: Vec<u8>,
    chain_id: Vec<u8>,
    protocol_version: i64,
    timestamp: i64,
    prev_block_hash: Vec<u8>,
    merkle_root: Vec<u8>,
    record_count: i64,
    proposer: Vec<u8>,
    proposer_signature: Vec<u8>,
    payload: Vec<u8>,
}
pub(super) const MAX_STORED_COMMITMENT_BLOCK_BYTES: usize = 16 * 1024;
pub(super) const MAX_ATOMIC_COMMITMENT_BLOCK_BATCH: usize = 32;
fn read_stored_record_commitment_block_row(
    row: &rusqlite::Row<'_>,
    context: &str,
) -> Result<StoredRecordCommitmentBlockRow, String> {
    Ok(StoredRecordCommitmentBlockRow {
        height: row
            .get(0)
            .map_err(|error| format!("read {context} height: {error}"))?,
        block_hash: row
            .get(1)
            .map_err(|error| format!("read {context} hash: {error}"))?,
        chain_id: row
            .get(2)
            .map_err(|error| format!("read {context} chain: {error}"))?,
        protocol_version: row
            .get(3)
            .map_err(|error| format!("read {context} version: {error}"))?,
        timestamp: row
            .get(4)
            .map_err(|error| format!("read {context} timestamp: {error}"))?,
        prev_block_hash: row
            .get(5)
            .map_err(|error| format!("read {context} previous hash: {error}"))?,
        merkle_root: row
            .get(6)
            .map_err(|error| format!("read {context} merkle root: {error}"))?,
        record_count: row
            .get(7)
            .map_err(|error| format!("read {context} count: {error}"))?,
        proposer: row
            .get(8)
            .map_err(|error| format!("read {context} proposer: {error}"))?,
        proposer_signature: row
            .get(9)
            .map_err(|error| format!("read {context} signature: {error}"))?,
        payload: row
            .get(10)
            .map_err(|error| format!("read {context} payload: {error}"))?,
    })
}
fn read_record_commitment_predecessor_transaction(
    transaction: &rusqlite::Transaction<'_>,
    from_height: u64,
    tip_height: u64,
) -> Result<[u8; 32], String> {
    if from_height == 1 {
        return Ok(GENESIS_PREV_HASH);
    }
    if from_height > tip_height.saturating_add(1) {
        // A request beyond tip+1 returns an empty signed page while still
        // binding the current verified tip. No predecessor is consumed.
        return Ok(GENESIS_PREV_HASH);
    }
    let previous_height = i64::try_from(from_height.saturating_sub(1))
        .map_err(|_| "verified commitment range predecessor overflow".to_string())?;
    let hash: Vec<u8> = transaction
        .query_row(
            "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
            params![previous_height],
            |row| row.get(0),
        )
        .map_err(|error| format!("read verified commitment range predecessor: {error}"))?;
    hash.try_into().map_err(|hash: Vec<u8>| {
        format!(
            "verified commitment range predecessor hash length {}",
            hash.len()
        )
    })
}

fn decode_and_verify_stored_record_commitment_block(
    stored: &StoredRecordCommitmentBlockRow,
    expected_height: u64,
    expected_prev_hash: &[u8; 32],
    context: &str,
) -> Result<RecordCommitmentBlockV1, String> {
    let stored_height = u64::try_from(stored.height)
        .map_err(|_| format!("{context} found an invalid stored height"))?;
    if stored.payload.len() > MAX_STORED_COMMITMENT_BLOCK_BYTES {
        return Err(format!(
            "{context} payload exceeds bound at height {stored_height}"
        ));
    }
    let block = bincode::deserialize::<RecordCommitmentBlockV1>(&stored.payload)
        .map_err(|_| format!("{context} payload decode failed at height {stored_height}"))?;
    let canonical_payload = bincode::serialize(&block)
        .map_err(|_| format!("{context} payload encode failed at height {stored_height}"))?;
    if canonical_payload != stored.payload {
        return Err(format!(
            "{context} payload is non-canonical at height {stored_height}"
        ));
    }

    block
        .verify(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            expected_height,
            expected_prev_hash,
        )
        .map_err(|error| {
            format!("{context} block validation failed at height {stored_height}: {error}")
        })?;

    let block_height = i64::try_from(block.header.height)
        .map_err(|_| format!("{context} height overflow at height {stored_height}"))?;
    let block_timestamp = i64::try_from(block.header.timestamp)
        .map_err(|_| format!("{context} timestamp overflow at height {stored_height}"))?;
    let block_hash = block.hash();
    if stored.height != block_height
        || stored.block_hash.as_slice() != block_hash.as_slice()
        || stored.chain_id.as_slice() != block.header.chain_id.as_slice()
        || stored.protocol_version != i64::from(block.header.protocol_version)
        || stored.timestamp != block_timestamp
        || stored.prev_block_hash.as_slice() != block.header.prev_block_hash.as_slice()
        || stored.merkle_root.as_slice() != block.header.merkle_root.as_slice()
        || stored.record_count != i64::from(block.header.record_count)
        || stored.proposer.as_slice() != block.header.proposer.as_slice()
        || stored.proposer_signature.as_slice() != block.proposer_signature.as_slice()
    {
        return Err(format!(
            "{context} stored row mismatch at height {stored_height}"
        ));
    }
    Ok(block)
}

fn persist_record_commitment_block_transaction(
    transaction: &rusqlite::Transaction<'_>,
    block: &RecordCommitmentBlockV1,
    received_from: Option<&[u8; 32]>,
    expected_height: u64,
    expected_prev_hash: &[u8; 32],
    created_at: i64,
) -> Result<RecordCommitmentAppendOutcome, String> {
    let block_height = i64::try_from(block.header.height)
        .map_err(|_| "commitment block height exceeds SQLite range".to_string())?;
    let block_timestamp = i64::try_from(block.header.timestamp)
        .map_err(|_| "commitment block timestamp exceeds SQLite range".to_string())?;
    let block_hash = block.hash();
    let existing_hash: Option<Vec<u8>> = transaction
        .query_row(
            "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
            params![block_height],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| format!("read existing commitment block: {error}"))?;
    if let Some(existing_hash) = existing_hash {
        if existing_hash.as_slice() == block_hash.as_slice() {
            return Ok(RecordCommitmentAppendOutcome::AlreadyPresent);
        }
        return Err(format!(
            "commitment chain fork at height {}",
            block.header.height
        ));
    }

    block
        .verify(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            expected_height,
            expected_prev_hash,
        )
        .map_err(|error| format!("commitment block validation failed: {error}"))?;
    let payload = bincode::serialize(block)
        .map_err(|error| format!("serialize commitment block: {error}"))?;
    if payload.len() > MAX_STORED_COMMITMENT_BLOCK_BYTES {
        return Err("serialized commitment block exceeds storage limit".to_string());
    }

    transaction
        .execute(
            "INSERT INTO record_commitment_blocks
             (height,block_hash,chain_id,protocol_version,timestamp,
              prev_block_hash,merkle_root,record_count,proposer,
              proposer_signature,payload,received_from,created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![
                block_height,
                block_hash.as_slice(),
                block.header.chain_id.as_slice(),
                i64::from(block.header.protocol_version),
                block_timestamp,
                block.header.prev_block_hash.as_slice(),
                block.header.merkle_root.as_slice(),
                i64::from(block.header.record_count),
                block.header.proposer.as_slice(),
                block.proposer_signature.as_slice(),
                payload,
                received_from.map(<[u8; 32]>::as_slice),
                created_at,
            ],
        )
        .map_err(|error| format!("persist commitment block: {error}"))?;

    for record_id in &block.record_ids {
        transaction
            .execute(
                "INSERT INTO record_block_commitments (record_id,block_height)
                 VALUES (?1,?2)",
                params![record_id.as_slice(), block_height],
            )
            .map_err(|error| {
                format!(
                    "persist commitment membership at height {}: {error}",
                    block.header.height
                )
            })?;
    }
    Ok(RecordCommitmentAppendOutcome::Inserted)
}

fn persist_record_commitment_tip_transaction(
    transaction: &rusqlite::Transaction<'_>,
    height: u64,
    hash: &[u8; 32],
) -> Result<(), String> {
    for (key, value) in [
        ("record_block_tip_hash", hash.to_vec()),
        ("record_block_tip_height", height.to_le_bytes().to_vec()),
        (
            "record_block_chain_id",
            AERONYX_MEMCHAIN_MAINNET_CHAIN_ID.to_vec(),
        ),
    ] {
        transaction
            .execute(
                "INSERT OR REPLACE INTO chain_state (key,value) VALUES (?1,?2)",
                params![key, value],
            )
            .map_err(|error| format!("update commitment chain state: {error}"))?;
    }
    Ok(())
}

impl MemoryStorage {
    pub async fn audit_record_commitment_chain(
        &self,
    ) -> Result<RecordCommitmentChainAudit, String> {
        // Clear first so a failed re-audit can never leave stale `verified`
        // evidence visible to operators or the central health plane.
        *self.commitment_integrity.write() = None;
        let started = Instant::now();
        let authority_root = *self.commitment_authority_root.read();
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|error| format!("begin commitment audit snapshot: {error}"))?;
        let mut block_statement = transaction
            .prepare(
                "SELECT height,block_hash,chain_id,protocol_version,timestamp,
                        prev_block_hash,merkle_root,record_count,proposer,
                        proposer_signature,payload
                 FROM record_commitment_blocks ORDER BY height ASC",
            )
            .map_err(|error| format!("prepare commitment audit blocks: {error}"))?;
        let mut membership_statement = transaction
            .prepare(
                "SELECT record_id FROM record_block_commitments
                 WHERE block_height=?1 ORDER BY record_id ASC",
            )
            .map_err(|error| format!("prepare commitment audit memberships: {error}"))?;
        let mut rows = block_statement
            .query([])
            .map_err(|error| format!("query commitment audit blocks: {error}"))?;

        let mut expected_height = 1u64;
        let mut expected_prev_hash = GENESIS_PREV_HASH;
        let mut block_count = 0u64;
        let mut commitment_count = 0u64;

        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read commitment audit block: {error}"))?
        {
            let stored = read_stored_record_commitment_block_row(row, "commitment audit")?;

            let stored_height = u64::try_from(stored.height)
                .map_err(|_| "commitment audit found an invalid stored height".to_string())?;
            let block = decode_and_verify_stored_record_commitment_block(
                &stored,
                expected_height,
                &expected_prev_hash,
                "commitment audit",
            )?;
            let block_hash = block.hash();

            let mut membership_rows =
                membership_statement
                    .query(params![stored.height])
                    .map_err(|error| {
                        format!(
                            "query commitment audit memberships at height {stored_height}: {error}"
                        )
                    })?;
            let mut indexed_record_ids = Vec::with_capacity(block.record_ids.len());
            while let Some(membership_row) = membership_rows.next().map_err(|error| {
                format!("read commitment audit membership at height {stored_height}: {error}")
            })? {
                let bytes: Vec<u8> = membership_row.get(0).map_err(|error| {
                    format!("decode commitment audit membership at height {stored_height}: {error}")
                })?;
                let record_id: [u8; 32] = bytes.try_into().map_err(|bytes: Vec<u8>| {
                    format!(
                        "commitment audit membership length {} at height {stored_height}",
                        bytes.len()
                    )
                })?;
                indexed_record_ids.push(record_id);
            }
            if indexed_record_ids != block.record_ids {
                return Err(format!(
                    "commitment audit membership index mismatch at height {stored_height}"
                ));
            }

            block_count = block_count.saturating_add(1);
            commitment_count = commitment_count.saturating_add(block.record_ids.len() as u64);
            expected_height = expected_height.saturating_add(1);
            expected_prev_hash = block_hash;
        }

        drop(rows);
        drop(block_statement);
        drop(membership_statement);
        let indexed_total: i64 = transaction
            .query_row("SELECT COUNT(*) FROM record_block_commitments", [], |row| {
                row.get(0)
            })
            .map_err(|error| format!("count commitment audit memberships: {error}"))?;
        let indexed_total = u64::try_from(indexed_total)
            .map_err(|_| "commitment audit found an invalid membership count".to_string())?;
        if indexed_total != commitment_count {
            return Err("commitment audit contains orphaned membership rows".to_string());
        }
        if let Some(root_coordinator) = authority_root {
            // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Re-audit the
            // complete dual-signed key schedule and every stored proposer in
            // this same SQLite snapshot before publishing a verified baseline.
            let authority_history = read_record_coordinator_handover_history_transaction(
                &transaction,
                &root_coordinator,
            )?;
            verify_record_commitment_proposer_history_transaction(
                &transaction,
                &root_coordinator,
                &authority_history,
            )?;
        }
        transaction
            .commit()
            .map_err(|error| format!("commit commitment audit snapshot: {error}"))?;

        let report = RecordCommitmentChainAudit {
            block_count,
            commitment_count,
            tip_height: expected_height.saturating_sub(1),
        };
        let verified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let verification_duration_ms =
            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        *self.commitment_integrity.write() = Some(RecordCommitmentIntegrityRuntime {
            baseline_verified_at: verified_at,
            last_verified_at: verified_at,
            verification_duration_ms,
            verified_block_count: report.block_count,
            verified_commitment_count: report.commitment_count,
            verified_tip_height: report.tip_height,
            verified_tip_hash: expected_prev_hash,
        });
        drop(conn);
        Ok(report)
    }

    /// Verifies and appends one node-blind commitment block.
    ///
    /// This compatibility API delegates to the bounded batch primitive so
    /// local mining and peer catch-up cannot drift into different validation,
    /// durability, integrity-baseline, or rollback-anchor behavior.
    ///
    /// # Errors
    ///
    /// Returns an error when validation, persistence, integrity advancement,
    /// or configured signed-tip anchor persistence fails.
    pub async fn append_record_commitment_block(
        &self,
        block: &RecordCommitmentBlockV1,
        received_from: Option<&[u8; 32]>,
    ) -> Result<RecordCommitmentAppendOutcome, String> {
        let outcome = self
            .append_record_commitment_blocks_atomic(std::slice::from_ref(block), received_from)
            .await?;
        match (outcome.inserted, outcome.already_present) {
            (1, 0) => Ok(RecordCommitmentAppendOutcome::Inserted),
            (0, 1) => Ok(RecordCommitmentAppendOutcome::AlreadyPresent),
            _ => Err("single commitment append produced an invalid aggregate outcome".to_string()),
        }
    }

    /// Atomically verifies and appends one bounded node-blind block page.
    ///
    /// Height order, previous-hash continuity, proposer signatures, Merkle
    /// integrity, commitment uniqueness, block rows, membership indexes, and
    /// the final tip share one `SQLite` `IMMEDIATE` transaction. Exact durable
    /// prefixes are idempotent, which permits safe retry after a lost response;
    /// any fork, gap, invalid block, or storage failure rolls back every newly
    /// inserted block in the page.
    ///
    /// The signed sidecar remains intentionally outside `SQLite`. After a page
    /// commit, it advances once to the final verified tip. A sidecar failure
    /// fails closed exactly like the legacy single-block path: `SQLite` remains
    /// durable, verified runtime state is cleared, and restart audit must
    /// safely repair the high-water mark before another append.
    ///
    /// # Errors
    ///
    /// Returns an error for oversized or non-contiguous input, a fork, an
    /// invalid block, a storage failure, a stale integrity baseline, or a
    /// configured signed-tip anchor failure.
    pub async fn append_record_commitment_blocks_atomic(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        received_from: Option<&[u8; 32]>,
    ) -> Result<RecordCommitmentBatchAppendOutcome, String> {
        if blocks.is_empty() {
            return Ok(RecordCommitmentBatchAppendOutcome::default());
        }
        if received_from.is_none() {
            if let Some(error) = self.local_record_commitment_production_error() {
                return Err(error.to_string());
            }
        }
        if blocks.len() > MAX_ATOMIC_COMMITMENT_BLOCK_BATCH {
            return Err(format!(
                "commitment block batch exceeds maximum of {MAX_ATOMIC_COMMITMENT_BLOCK_BATCH}"
            ));
        }

        let (anchor_enabled, anchor_ready) = {
            let runtime = self.commitment_tip_anchor.read();
            (
                runtime.config.is_some(),
                matches!(runtime.state, "initialized" | "verified" | "repaired"),
            )
        };
        if anchor_enabled && !anchor_ready {
            return Err(
                "commitment tip anchor is not ready; restart and complete startup audit"
                    .to_string(),
            );
        }

        let committed = self
            .append_record_commitment_block_page_transaction(blocks, received_from)
            .await?;
        if committed.inserted_indices.is_empty() {
            return Ok(committed.outcome);
        }

        let can_advance = self.advance_record_commitment_integrity_for_batch(
            blocks,
            &committed.inserted_indices,
            committed.appended_at,
        );
        self.persist_record_commitment_batch_anchor(
            blocks,
            &committed.inserted_indices,
            anchor_enabled,
            can_advance,
        )
        .await?;

        for index in committed.inserted_indices {
            let block = &blocks[index];
            info!(
                height = block.header.height,
                commitments = block.record_ids.len(),
                hash = %block.header.hash_hex(),
                source = if received_from.is_some() { "peer" } else { "local" },
                "[MEMCHAIN_BLOCK] Verified commitment block appended"
            );
        }
        Ok(committed.outcome)
    }

    async fn append_record_commitment_block_page_transaction(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        received_from: Option<&[u8; 32]>,
    ) -> Result<CommittedRecordCommitmentBatch, String> {
        let authority_root = *self.commitment_authority_root.read();
        let mut conn = self.conn.lock().await;
        if received_from.is_none() {
            if let Some(error) = self.local_record_commitment_production_error() {
                return Err(error.to_string());
            }
        }
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin commitment block batch transaction: {error}"))?;
        let tip: Option<(i64, Vec<u8>)> = transaction
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| format!("read commitment chain tip: {error}"))?;
        let (mut current_height, mut current_hash) = match tip {
            Some((height, hash)) => {
                let height = u64::try_from(height)
                    .map_err(|_| "stored commitment tip has invalid height".to_string())?;
                let hash: [u8; 32] = hash.try_into().map_err(|value: Vec<u8>| {
                    format!("stored tip hash has invalid length {}", value.len())
                })?;
                (height, hash)
            }
            None => (0, GENESIS_PREV_HASH),
        };
        let authority_history = authority_root
            .as_ref()
            .map(|root| read_record_coordinator_handover_history_transaction(&transaction, root))
            .transpose()?;
        let appended_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let created_at = i64::try_from(appended_at)
            .map_err(|_| "commitment append time exceeds SQLite range".to_string())?;
        let mut outcome = RecordCommitmentBatchAppendOutcome::default();
        let mut inserted_indices = Vec::with_capacity(blocks.len());
        let mut previous_input_height: Option<u64> = None;

        for (index, block) in blocks.iter().enumerate() {
            if previous_input_height
                .is_some_and(|height| height.checked_add(1) != Some(block.header.height))
            {
                return Err("commitment block batch is not height-contiguous".to_string());
            }
            previous_input_height = Some(block.header.height);
            if let (Some(root), Some(history)) =
                (authority_root.as_ref(), authority_history.as_ref())
            {
                // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Enforce
                // the exact height-scoped key schedule before any row in the
                // batch is inserted. A wrong old/new key fails the entire
                // IMMEDIATE transaction without advancing runtime integrity.
                let (_, expected) =
                    record_commitment_authority_at_height(root, history, block.header.height);
                if block.header.proposer != expected {
                    return Err(format!(
                        "commitment block has an unauthorized proposer at height {}",
                        block.header.height
                    ));
                }
            }
            let expected_height = current_height
                .checked_add(1)
                .ok_or_else(|| "commitment chain height exhausted".to_string())?;
            match persist_record_commitment_block_transaction(
                &transaction,
                block,
                received_from,
                expected_height,
                &current_hash,
                created_at,
            )? {
                RecordCommitmentAppendOutcome::Inserted => {
                    current_height = block.header.height;
                    current_hash = block.hash();
                    inserted_indices.push(index);
                    outcome.inserted = outcome.inserted.saturating_add(1);
                }
                RecordCommitmentAppendOutcome::AlreadyPresent => {
                    outcome.already_present = outcome.already_present.saturating_add(1);
                }
            }
        }

        if !inserted_indices.is_empty() {
            persist_record_commitment_tip_transaction(&transaction, current_height, &current_hash)?;
        }
        transaction
            .commit()
            .map_err(|error| format!("commit commitment block batch transaction: {error}"))?;
        drop(conn);
        Ok(CommittedRecordCommitmentBatch {
            outcome,
            inserted_indices,
            appended_at,
        })
    }

    fn advance_record_commitment_integrity_for_batch(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        inserted_indices: &[usize],
        appended_at: u64,
    ) -> bool {
        let mut integrity = self.commitment_integrity.write();
        let mut can_advance = integrity.is_some();
        for index in inserted_indices {
            let Some(block) = blocks.get(*index) else {
                can_advance = false;
                break;
            };
            let step_valid = integrity.as_ref().is_some_and(|runtime| {
                runtime.verified_tip_height.checked_add(1) == Some(block.header.height)
                    && runtime.verified_block_count.checked_add(1) == Some(block.header.height)
                    && runtime.verified_tip_hash == block.header.prev_block_hash
            });
            if !step_valid {
                can_advance = false;
                break;
            }
            if let Some(runtime) = integrity.as_mut() {
                runtime.last_verified_at = appended_at;
                runtime.verified_block_count = runtime.verified_block_count.saturating_add(1);
                runtime.verified_commitment_count = runtime
                    .verified_commitment_count
                    .saturating_add(block.record_ids.len() as u64);
                runtime.verified_tip_height = block.header.height;
                runtime.verified_tip_hash = block.hash();
            }
        }
        if !can_advance && integrity.is_some() {
            *integrity = None;
        }
        can_advance
    }

    async fn persist_record_commitment_batch_anchor(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        inserted_indices: &[usize],
        anchor_enabled: bool,
        can_advance: bool,
    ) -> Result<(), String> {
        let final_block = inserted_indices
            .last()
            .and_then(|index| blocks.get(*index))
            .ok_or_else(|| "commitment batch lost its inserted outcome".to_string())?;
        if anchor_enabled && !can_advance {
            self.fail_record_commitment_tip_anchor("invalid", final_block.header.height, false);
            return Err(
                "commitment block was committed in an atomic batch but the verified runtime baseline could not advance; restart and re-audit before producing another block"
                    .to_string(),
            );
        }

        let anchor_config = self
            .commitment_tip_anchor
            .read()
            .config
            .as_ref()
            .map(|config| (config.path.clone(), config.identity.clone()));
        let Some((path, identity)) = anchor_config else {
            return Ok(());
        };
        let persisted_at = match persist_record_commitment_tip_anchor(
            path,
            final_block.header.height,
            final_block.hash(),
            &identity,
        )
        .await
        {
            Ok(persisted_at) => persisted_at,
            Err(error) => {
                let previous_height = inserted_indices
                    .first()
                    .and_then(|index| blocks.get(*index))
                    .map_or(final_block.header.height, |block| block.header.height)
                    .saturating_sub(1);
                self.fail_record_commitment_tip_anchor("write_failed", previous_height, true);
                return Err(format!(
                    "commitment block was committed in an atomic batch but signed tip anchor persistence failed; restart and re-audit: {error}"
                ));
            }
        };
        let mut runtime = self.commitment_tip_anchor.write();
        runtime.state = "verified";
        runtime.anchored_height = final_block.header.height;
        runtime.last_verified_at = Some(persisted_at);
        runtime.last_persisted_at = Some(persisted_at);
        drop(runtime);
        Ok(())
    }

    /// Reads one canonically reverified block page and its tip from a single
    /// audit-gated `SQLite` snapshot.
    ///
    /// This is the only range primitive suitable for a signed peer response.
    /// It refuses to serve when the process has no complete audit baseline,
    /// when the baseline no longer matches the database tip, or when any
    /// selected payload, denormalized row, signature, height, or parent link
    /// fails canonical verification.
    ///
    /// # Errors
    ///
    /// Returns an error when the chain is unaudited/stale, the snapshot cannot
    /// be read, or a selected stored block fails canonical verification.
    pub async fn get_verified_record_commitment_block_page(
        &self,
        from_height: u64,
        limit: usize,
    ) -> Result<VerifiedRecordCommitmentBlockPage, String> {
        let from_height = from_height.max(1);
        let query_from_height = i64::try_from(from_height).unwrap_or(i64::MAX);
        let query_limit = i64::try_from(limit.clamp(1, 32)).unwrap_or(32);
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|error| format!("begin verified commitment range snapshot: {error}"))?;
        let integrity = (*self.commitment_integrity.read())
            .ok_or_else(|| "commitment chain is not fully audited".to_string())?;
        let (tip_height, tip_hash) =
            read_record_commitment_tip_transaction(&transaction, "verified commitment range")?;
        if integrity.verified_tip_height != tip_height
            || integrity.verified_tip_hash != tip_hash
            || integrity.verified_block_count != tip_height
        {
            return Err("commitment chain audit baseline is stale".to_string());
        }

        let mut expected_height = from_height;
        let mut expected_prev_hash =
            read_record_commitment_predecessor_transaction(&transaction, from_height, tip_height)?;

        let blocks = {
            let mut statement = transaction
                .prepare(
                    "SELECT height,block_hash,chain_id,protocol_version,timestamp,
                            prev_block_hash,merkle_root,record_count,proposer,
                            proposer_signature,payload
                     FROM record_commitment_blocks
                     WHERE height>=?1 ORDER BY height ASC LIMIT ?2",
                )
                .map_err(|error| format!("prepare verified commitment range: {error}"))?;
            let mut rows = statement
                .query(params![query_from_height, query_limit])
                .map_err(|error| format!("query verified commitment range: {error}"))?;
            let mut blocks = Vec::new();
            while let Some(row) = rows
                .next()
                .map_err(|error| format!("read verified commitment range block: {error}"))?
            {
                let stored = read_stored_record_commitment_block_row(row, "commitment range")?;
                let block = decode_and_verify_stored_record_commitment_block(
                    &stored,
                    expected_height,
                    &expected_prev_hash,
                    "commitment range",
                )?;
                expected_height = expected_height
                    .checked_add(1)
                    .ok_or_else(|| "verified commitment range height exhausted".to_string())?;
                expected_prev_hash = block.hash();
                blocks.push(block);
            }
            blocks
        };
        transaction
            .commit()
            .map_err(|error| format!("commit verified commitment range snapshot: {error}"))?;
        drop(conn);
        Ok(VerifiedRecordCommitmentBlockPage {
            blocks,
            tip_height,
            tip_hash,
        })
    }

    /// Reads a bounded height-ordered range for compatibility callers.
    ///
    /// This method does not establish an audit-gated snapshot and must never
    /// feed a signed peer response. New sync serving code must call
    /// `get_verified_record_commitment_block_page`.
    ///
    /// # Errors
    ///
    /// Returns an error when the range query or stored payload decode fails.
    pub async fn get_record_commitment_block_range(
        &self,
        from_height: u64,
        limit: usize,
    ) -> Result<Vec<RecordCommitmentBlockV1>, String> {
        let from_height = from_height.max(1);
        let from_height = i64::try_from(from_height).unwrap_or(i64::MAX);
        let limit = limit.clamp(1, 32);
        let conn = self.conn.lock().await;
        let mut statement = conn
            .prepare(
                "SELECT payload FROM record_commitment_blocks
                 WHERE height>=?1 ORDER BY height ASC LIMIT ?2",
            )
            .map_err(|error| format!("prepare commitment range query: {error}"))?;
        let rows = statement
            .query_map(params![from_height, limit as i64], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .map_err(|error| format!("query commitment block range: {error}"))?;

        let mut blocks = Vec::new();
        for row in rows {
            let payload = row.map_err(|error| format!("read commitment block payload: {error}"))?;
            let block = bincode::deserialize::<RecordCommitmentBlockV1>(&payload)
                .map_err(|error| format!("decode stored commitment block: {error}"))?;
            blocks.push(block);
        }
        Ok(blocks)
    }

    /// Returns the raw commitment tip from the authoritative block table.
    ///
    /// This compatibility/telemetry helper intentionally has no `Result` and
    /// therefore cannot prove an audit baseline. Security-sensitive sync code
    /// must use `record_commitment_chain_checkpoint` or the verified page API.
    pub async fn record_commitment_chain_tip(&self) -> (u64, [u8; 32]) {
        let conn = self.conn.lock().await;
        let tip: Option<(u64, Vec<u8>)> = conn
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, i64>(0)? as u64, row.get(1)?)),
            )
            .optional()
            .unwrap_or(None);
        match tip {
            Some((height, hash)) if hash.len() == 32 => {
                let mut value = [0u8; 32];
                value.copy_from_slice(&hash);
                (height, value)
            }
            _ => (0, GENESIS_PREV_HASH),
        }
    }

    /// Returns one audit-backed comparison checkpoint and the current tip from
    /// a single SQLite view. Height zero uses the protocol genesis hash.
    pub async fn record_commitment_chain_checkpoint(
        &self,
        requested_height: u64,
    ) -> Result<(u64, [u8; 32], u64, [u8; 32]), String> {
        let conn = self.conn.lock().await;
        let integrity = (*self.commitment_integrity.read())
            .ok_or_else(|| "commitment chain is not fully audited".to_string())?;
        let tip: Option<(i64, Vec<u8>)> = conn
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| format!("read commitment checkpoint tip: {error}"))?;
        let (tip_height, tip_hash) = match tip {
            Some((height, hash)) => {
                let height = u64::try_from(height)
                    .map_err(|_| "commitment checkpoint tip height is invalid".to_string())?;
                let hash: [u8; 32] = hash.try_into().map_err(|hash: Vec<u8>| {
                    format!("commitment checkpoint tip hash length {}", hash.len())
                })?;
                (height, hash)
            }
            None => (0, GENESIS_PREV_HASH),
        };
        if integrity.verified_tip_height != tip_height || integrity.verified_tip_hash != tip_hash {
            return Err("commitment chain audit baseline is stale".to_string());
        }

        let checkpoint_height = requested_height.min(tip_height);
        let checkpoint_hash = if checkpoint_height == 0 {
            GENESIS_PREV_HASH
        } else {
            let height = i64::try_from(checkpoint_height)
                .map_err(|_| "commitment checkpoint height exceeds SQLite range".to_string())?;
            let hash: Vec<u8> = conn
                .query_row(
                    "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                    params![height],
                    |row| row.get(0),
                )
                .map_err(|error| format!("read commitment checkpoint hash: {error}"))?;
            hash.try_into().map_err(|hash: Vec<u8>| {
                format!("commitment checkpoint hash length {}", hash.len())
            })?
        };
        Ok((checkpoint_height, checkpoint_hash, tip_height, tip_hash))
    }

    /// Returns aggregate chain health without exposing record commitments,
    /// proposer identities, or peer metadata.
    pub async fn set_chain_state(&self, block_hash: &[u8; 32], height: u64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT OR REPLACE INTO chain_state (key,value) VALUES ('last_block_hash',?1)",
            params![block_hash.as_slice()],
        );
        let _ = conn.execute(
            "INSERT OR REPLACE INTO chain_state (key,value) VALUES ('last_block_height',?1)",
            params![height.to_le_bytes().as_slice()],
        );
    }

    pub async fn last_block_hash(&self) -> [u8; 32] {
        let (commitment_height, commitment_hash) = self.record_commitment_chain_tip().await;
        if commitment_height > 0 {
            return commitment_hash;
        }
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT value FROM chain_state WHERE key='last_block_hash'",
            [],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                let mut h = [0u8; 32];
                if blob.len() == 32 {
                    h.copy_from_slice(&blob);
                }
                Ok(h)
            },
        )
        .unwrap_or([0u8; 32])
    }

    pub async fn last_block_height(&self) -> u64 {
        let (commitment_height, _) = self.record_commitment_chain_tip().await;
        if commitment_height > 0 {
            return commitment_height;
        }
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT value FROM chain_state WHERE key='last_block_height'",
            [],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                if blob.len() == 8 {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&blob);
                    Ok(u64::from_le_bytes(b))
                } else {
                    Ok(0u64)
                }
            },
        )
        .unwrap_or(0)
    }
}
