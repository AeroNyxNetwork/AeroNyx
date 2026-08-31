// ============================================
// File: crates/aeronyx-server/src/services/blind_vault_replica_recovery_store.rs
// ============================================
//! Atomic generation adapter for source-owned Blind Vault replica recovery.
//!
//! ## Creation Reason
//! The core workflow defines prepared/committed/resolve durability semantics,
//! while the node needs a concrete crash-consistent implementation that does
//! not expose private continuations or depend on the terminal vault database.
//!
//! ## Main Functionality
//! - Implements `BlindVaultReplicaRecoveryStore` over one atomic generation.
//! - Freezes a bounded versioned V1 disk body independent from core wire types.
//! - Commits snapshot and journal relation through explicit phase transitions.
//! - Detects sequence rollback, workflow drift, and journal substitution.
//! - Verifies checksums and sealed-record commitments before returning state.
//! - Re-confirms durability before accepting an exact idempotent retry.
//!
//! ## Dependencies
//! - `blind_vault_replica_recovery_io.rs`: restrictive atomic private I/O.
//! - `aeronyx_core::protocol`: storage-neutral recovery records and trait.
//! - `bincode`/`serde`: fixed local V1 body encoding.
//! - `sha2`: accidental-corruption and sealed-record commitments.
//!
//! ## Main Logical Flow
//! 1. Load and validate the complete current generation under one process lock.
//! 2. Apply exactly one allowed monotonic domain transition in memory.
//! 3. Encode a versioned body plus checksum within a fixed maximum size.
//! 4. Atomically publish the complete generation through the private I/O host.
//! 5. On restart, validate shape and return only sealed containers to core.
//!
//! ## Important Note For The Next Developer
//! - This file provides crash consistency, not hardware anti-rollback.
//! - A TPM/OS monotonic adapter must independently protect accepted sequences.
//! - Never store request plaintext, reply keys, routes, endpoints, or contacts.
//! - Never permit normal snapshots to cross unresolved attempt phases.
//! - Never weaken exact work/attempt/sequence/commitment comparisons.
//!
//! Last Modified: v1.6.0-MacOSTestFixtureCanonicalization - Canonicalized the
//! temporary test root without weakening production no-follow enforcement.
//! v1.5.0-FrozenV1GoldenVector - Froze the complete V1 encoded
//! generation and exact decode/re-encode compatibility guard.
//! v1.4.0-CrashRecoveryFormatHardening - Added fail-closed V1
//! corruption, restart-phase, rollback, conflict, and exact-retry coverage.
//! v1.3.0-IdempotentDurabilityConfirmation - Re-synchronized
//! the current file and directory before exact retry success.
//! v1.2.0-PreparedAbort - Added idempotent exact cleanup for a
//! journal proven not to have reached committed dispatch state.
//! v1.1.0-ExactCrashRetry - Made every durable transition
//! idempotent only for the exact same sequence and sealed commitment.
//! v1.0.0-AtomicRecoveryGeneration - Initial single-writer,
//! versioned, fail-closed recovery-store adapter.
//! ============================================

#![cfg(unix)]

use std::path::Path;

use aeronyx_core::protocol::{
    BlindVaultReplicaAttemptDurabilityPhase, BlindVaultReplicaCommittedAttemptRecord,
    BlindVaultReplicaPreparedAttemptRecord, BlindVaultReplicaRecoveryState,
    BlindVaultReplicaRecoveryStore, BlindVaultReplicaSnapshotRecord,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES, MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES,
};
use bincode::Options;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::{Zeroize, Zeroizing};

use super::blind_vault_replica_recovery_io::{PrivateAtomicRecoveryFile, PrivateRecoveryIoError};

const RECOVERY_FILE_MAGIC: [u8; 4] = *b"AXVR";
const RECOVERY_FILE_VERSION_V1: u16 = 1;
const RECOVERY_FILE_CHECKSUM_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Recovery-File-v1";
const SEALED_RECORD_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Recovery-Sealed-Record-v1";
const RECOVERY_FILE_HEADER_BYTES: usize = 4 + 2 + 4;
const RECOVERY_FILE_CHECKSUM_BYTES: usize = 32;
const MAX_RECOVERY_FILE_BYTES: usize = MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES
    + MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES
    + 4 * 1024;

/// Fail-closed host and generation transition errors.
#[derive(Debug, Error)]
pub(crate) enum BlindVaultReplicaRecoveryStoreError {
    #[error(transparent)]
    Host(#[from] PrivateRecoveryIoError),
    #[error("blind vault replica recovery state is corrupt")]
    CorruptState,
    #[error("blind vault replica recovery transition is invalid")]
    InvalidTransition,
    #[error("blind vault replica recovery sequence rollback was detected")]
    RollbackDetected,
    #[error("blind vault replica recovery binding conflicts with durable state")]
    StateConflict,
    #[error("blind vault replica recovery encoding exceeds its bound")]
    TooLarge,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum StoredAttemptPhaseV1 {
    Resolved,
    Prepared,
    Committed,
}

#[derive(Serialize, Deserialize)]
struct StoredAttemptV1 {
    work_sequence: u16,
    attempt: u8,
    journal_sequence: u64,
    retain_until_ms: u64,
    journal_commitment: [u8; 32],
    sealed_journal: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct StoredResolvedAttemptV1 {
    journal_sequence: u64,
    journal_commitment: [u8; 32],
}

impl Drop for StoredAttemptV1 {
    fn drop(&mut self) {
        self.sealed_journal.zeroize();
        self.journal_commitment.zeroize();
    }
}

#[derive(Serialize, Deserialize)]
struct StoredRecoveryStateV1 {
    workflow_id: [u8; 16],
    phase: StoredAttemptPhaseV1,
    snapshot_sequence: u64,
    accepted_journal_sequence: u64,
    last_resolved_attempt: Option<StoredResolvedAttemptV1>,
    snapshot_commitment: [u8; 32],
    sealed_snapshot: Vec<u8>,
    attempt: Option<StoredAttemptV1>,
}

impl Drop for StoredRecoveryStateV1 {
    fn drop(&mut self) {
        self.sealed_snapshot.zeroize();
        self.snapshot_commitment.zeroize();
    }
}

/// Production single-writer recovery generation store.
pub(crate) struct FileBlindVaultReplicaRecoveryStore {
    file: PrivateAtomicRecoveryFile,
}

impl FileBlindVaultReplicaRecoveryStore {
    pub(crate) fn open(directory: &Path) -> Result<Self, BlindVaultReplicaRecoveryStoreError> {
        let store = Self {
            file: PrivateAtomicRecoveryFile::open(directory)?,
        };
        if let Some(state) = store.load_stored()? {
            state.validate()?;
        }
        Ok(store)
    }

    fn load_stored(
        &self,
    ) -> Result<Option<StoredRecoveryStateV1>, BlindVaultReplicaRecoveryStoreError> {
        let Some(bytes) = self.file.read(MAX_RECOVERY_FILE_BYTES)? else {
            return Ok(None);
        };
        let bytes = Zeroizing::new(bytes);
        decode_state(bytes.as_slice()).map(Some)
    }

    fn publish(
        &self,
        state: &StoredRecoveryStateV1,
    ) -> Result<(), BlindVaultReplicaRecoveryStoreError> {
        state.validate()?;
        let encoded = encode_state(state)?;
        self.file
            .replace(encoded.as_slice(), MAX_RECOVERY_FILE_BYTES)?;
        Ok(())
    }
}

impl BlindVaultReplicaRecoveryStore for FileBlindVaultReplicaRecoveryStore {
    type Error = BlindVaultReplicaRecoveryStoreError;

    fn persist_snapshot(
        &mut self,
        snapshot: &BlindVaultReplicaSnapshotRecord<'_>,
    ) -> Result<(), Self::Error> {
        let next = match self.load_stored()? {
            None => StoredRecoveryStateV1::from_snapshot(snapshot, 0, None),
            Some(current) => {
                current.require_workflow(snapshot.workflow_id())?;
                if current.phase != StoredAttemptPhaseV1::Resolved || current.attempt.is_some() {
                    return Err(BlindVaultReplicaRecoveryStoreError::InvalidTransition);
                }
                if current.snapshot_matches(snapshot) {
                    self.file.confirm_current_durable()?;
                    return Ok(());
                }
                current.require_new_snapshot_sequence(snapshot.snapshot_sequence())?;
                StoredRecoveryStateV1::from_snapshot(
                    snapshot,
                    current.accepted_journal_sequence,
                    current.last_resolved_attempt,
                )
            }
        };
        self.publish(&next)
    }

    fn persist_prepared_attempt(
        &mut self,
        prepared: &BlindVaultReplicaPreparedAttemptRecord<'_>,
    ) -> Result<(), Self::Error> {
        let mut current = self
            .load_stored()?
            .ok_or(BlindVaultReplicaRecoveryStoreError::InvalidTransition)?;
        current.require_workflow(prepared.work_id().workflow_id())?;
        if current.phase == StoredAttemptPhaseV1::Prepared
            && current.attempt_matches_prepared(prepared)
        {
            self.file.confirm_current_durable()?;
            return Ok(());
        }
        if current.phase != StoredAttemptPhaseV1::Resolved || current.attempt.is_some() {
            return Err(BlindVaultReplicaRecoveryStoreError::InvalidTransition);
        }
        if prepared.journal_sequence() <= current.accepted_journal_sequence {
            return Err(BlindVaultReplicaRecoveryStoreError::RollbackDetected);
        }
        current.phase = StoredAttemptPhaseV1::Prepared;
        current.accepted_journal_sequence = prepared.journal_sequence();
        current.attempt = Some(StoredAttemptV1 {
            work_sequence: prepared.work_id().sequence(),
            attempt: prepared.attempt(),
            journal_sequence: prepared.journal_sequence(),
            retain_until_ms: prepared.retain_until_ms(),
            journal_commitment: prepared.sealed_commitment(),
            sealed_journal: prepared.sealed_journal().to_vec(),
        });
        self.publish(&current)
    }

    fn persist_committed_attempt(
        &mut self,
        committed: &BlindVaultReplicaCommittedAttemptRecord<'_>,
    ) -> Result<(), Self::Error> {
        let mut current = self
            .load_stored()?
            .ok_or(BlindVaultReplicaRecoveryStoreError::InvalidTransition)?;
        current.require_workflow(committed.work_id().workflow_id())?;
        current.require_workflow(committed.snapshot().workflow_id())?;
        if current.phase == StoredAttemptPhaseV1::Committed
            && current.attempt_matches_committed(committed)
            && current.snapshot_matches(committed.snapshot())
        {
            self.file.confirm_current_durable()?;
            return Ok(());
        }
        current.require_new_snapshot_sequence(committed.snapshot().snapshot_sequence())?;
        let attempt = current
            .attempt
            .as_ref()
            .ok_or(BlindVaultReplicaRecoveryStoreError::InvalidTransition)?;
        if current.phase != StoredAttemptPhaseV1::Prepared
            || attempt.work_sequence != committed.work_id().sequence()
            || attempt.attempt != committed.attempt()
            || attempt.journal_sequence != committed.journal_sequence()
            || attempt.journal_commitment != committed.journal_commitment()
        {
            return Err(BlindVaultReplicaRecoveryStoreError::StateConflict);
        }
        current.phase = StoredAttemptPhaseV1::Committed;
        current.snapshot_sequence = committed.snapshot().snapshot_sequence();
        current.snapshot_commitment = committed.snapshot().sealed_commitment();
        current.sealed_snapshot = committed.snapshot().sealed_snapshot().to_vec();
        self.publish(&current)
    }

    fn abort_prepared_attempt(
        &mut self,
        journal_sequence: u64,
        journal_commitment: [u8; 32],
    ) -> Result<(), Self::Error> {
        let mut current = self
            .load_stored()?
            .ok_or(BlindVaultReplicaRecoveryStoreError::InvalidTransition)?;
        if current.phase == StoredAttemptPhaseV1::Resolved
            && current.attempt.is_none()
            && current.last_resolved_attempt
                == Some(StoredResolvedAttemptV1 {
                    journal_sequence,
                    journal_commitment,
                })
        {
            self.file.confirm_current_durable()?;
            return Ok(());
        }
        let attempt = current
            .attempt
            .as_ref()
            .ok_or(BlindVaultReplicaRecoveryStoreError::InvalidTransition)?;
        if current.phase != StoredAttemptPhaseV1::Prepared
            || attempt.journal_sequence != journal_sequence
            || attempt.journal_commitment != journal_commitment
        {
            return Err(BlindVaultReplicaRecoveryStoreError::StateConflict);
        }
        // [BLIND-VAULT-RECOVERY-PREPARED-ABORT 2026-08-29 by Codex] The
        // journal high-water survives cleanup. Only the private container and
        // its attempt metadata are removed; the prior snapshot stays current.
        current.phase = StoredAttemptPhaseV1::Resolved;
        current.last_resolved_attempt = Some(StoredResolvedAttemptV1 {
            journal_sequence,
            journal_commitment,
        });
        current.attempt = None;
        self.publish(&current)
    }

    fn resolve_attempt(
        &mut self,
        snapshot: &BlindVaultReplicaSnapshotRecord<'_>,
        journal_sequence: u64,
        journal_commitment: [u8; 32],
    ) -> Result<(), Self::Error> {
        let mut current = self
            .load_stored()?
            .ok_or(BlindVaultReplicaRecoveryStoreError::InvalidTransition)?;
        current.require_workflow(snapshot.workflow_id())?;
        if current.phase == StoredAttemptPhaseV1::Resolved
            && current.attempt.is_none()
            && current.last_resolved_attempt
                == Some(StoredResolvedAttemptV1 {
                    journal_sequence,
                    journal_commitment,
                })
            && current.snapshot_matches(snapshot)
        {
            self.file.confirm_current_durable()?;
            return Ok(());
        }
        current.require_new_snapshot_sequence(snapshot.snapshot_sequence())?;
        let attempt = current
            .attempt
            .as_ref()
            .ok_or(BlindVaultReplicaRecoveryStoreError::InvalidTransition)?;
        if current.phase != StoredAttemptPhaseV1::Committed
            || attempt.journal_sequence != journal_sequence
            || attempt.journal_commitment != journal_commitment
        {
            return Err(BlindVaultReplicaRecoveryStoreError::StateConflict);
        }
        current.phase = StoredAttemptPhaseV1::Resolved;
        current.snapshot_sequence = snapshot.snapshot_sequence();
        current.snapshot_commitment = snapshot.sealed_commitment();
        current.sealed_snapshot = snapshot.sealed_snapshot().to_vec();
        current.last_resolved_attempt = Some(StoredResolvedAttemptV1 {
            journal_sequence,
            journal_commitment,
        });
        current.attempt = None;
        self.publish(&current)
    }

    fn load_recovery_state(
        &mut self,
    ) -> Result<Option<BlindVaultReplicaRecoveryState>, Self::Error> {
        let Some(mut stored) = self.load_stored()? else {
            return Ok(None);
        };
        stored.validate()?;
        let phase = match stored.phase {
            StoredAttemptPhaseV1::Resolved => BlindVaultReplicaAttemptDurabilityPhase::Resolved,
            StoredAttemptPhaseV1::Prepared => BlindVaultReplicaAttemptDurabilityPhase::Prepared,
            StoredAttemptPhaseV1::Committed => BlindVaultReplicaAttemptDurabilityPhase::Committed,
        };
        let journal = stored
            .attempt
            .as_mut()
            .map(|attempt| std::mem::take(&mut attempt.sealed_journal));
        let snapshot = std::mem::take(&mut stored.sealed_snapshot);
        BlindVaultReplicaRecoveryState::new(
            phase,
            stored.snapshot_sequence,
            stored.accepted_journal_sequence,
            snapshot,
            journal,
        )
        .ok_or(BlindVaultReplicaRecoveryStoreError::CorruptState)
        .map(Some)
    }
}

impl StoredRecoveryStateV1 {
    fn from_snapshot(
        snapshot: &BlindVaultReplicaSnapshotRecord<'_>,
        accepted_journal_sequence: u64,
        last_resolved_attempt: Option<StoredResolvedAttemptV1>,
    ) -> Self {
        Self {
            workflow_id: snapshot.workflow_id(),
            phase: StoredAttemptPhaseV1::Resolved,
            snapshot_sequence: snapshot.snapshot_sequence(),
            accepted_journal_sequence,
            last_resolved_attempt,
            snapshot_commitment: snapshot.sealed_commitment(),
            sealed_snapshot: snapshot.sealed_snapshot().to_vec(),
            attempt: None,
        }
    }

    fn require_workflow(
        &self,
        workflow_id: [u8; 16],
    ) -> Result<(), BlindVaultReplicaRecoveryStoreError> {
        if self.workflow_id != workflow_id {
            return Err(BlindVaultReplicaRecoveryStoreError::StateConflict);
        }
        Ok(())
    }

    fn require_new_snapshot_sequence(
        &self,
        snapshot_sequence: u64,
    ) -> Result<(), BlindVaultReplicaRecoveryStoreError> {
        if snapshot_sequence <= self.snapshot_sequence {
            return Err(BlindVaultReplicaRecoveryStoreError::RollbackDetected);
        }
        Ok(())
    }

    fn snapshot_matches(&self, snapshot: &BlindVaultReplicaSnapshotRecord<'_>) -> bool {
        self.snapshot_sequence == snapshot.snapshot_sequence()
            && self.snapshot_commitment == snapshot.sealed_commitment()
    }

    fn attempt_matches_prepared(
        &self,
        prepared: &BlindVaultReplicaPreparedAttemptRecord<'_>,
    ) -> bool {
        self.attempt.as_ref().is_some_and(|attempt| {
            attempt.work_sequence == prepared.work_id().sequence()
                && attempt.attempt == prepared.attempt()
                && attempt.journal_sequence == prepared.journal_sequence()
                && attempt.retain_until_ms == prepared.retain_until_ms()
                && attempt.journal_commitment == prepared.sealed_commitment()
        })
    }

    fn attempt_matches_committed(
        &self,
        committed: &BlindVaultReplicaCommittedAttemptRecord<'_>,
    ) -> bool {
        self.attempt.as_ref().is_some_and(|attempt| {
            attempt.work_sequence == committed.work_id().sequence()
                && attempt.attempt == committed.attempt()
                && attempt.journal_sequence == committed.journal_sequence()
                && attempt.journal_commitment == committed.journal_commitment()
        })
    }

    fn validate(&self) -> Result<(), BlindVaultReplicaRecoveryStoreError> {
        // [BLIND-VAULT-RECOVERY-GENERATION 2026-08-29 by Codex] The adapter
        // validates all public metadata and ciphertext commitments before core
        // performs identity-bound AEAD authentication on the sealed records.
        if self.workflow_id == [0; 16]
            || self.snapshot_sequence == 0
            || self.sealed_snapshot.is_empty()
            || self.sealed_snapshot.len() > MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES
            || self.snapshot_commitment != sealed_record_commitment(&self.sealed_snapshot)
        {
            return Err(BlindVaultReplicaRecoveryStoreError::CorruptState);
        }
        match (self.phase, self.attempt.as_ref()) {
            (StoredAttemptPhaseV1::Resolved, None)
                if valid_resolved_attempt_high_water(
                    self.accepted_journal_sequence,
                    self.last_resolved_attempt,
                ) =>
            {
                Ok(())
            }
            (StoredAttemptPhaseV1::Prepared | StoredAttemptPhaseV1::Committed, Some(attempt))
                if attempt.attempt > 0
                    && attempt.journal_sequence > 0
                    && attempt.journal_sequence == self.accepted_journal_sequence
                    && valid_prior_resolved_attempt(
                        self.accepted_journal_sequence,
                        self.last_resolved_attempt,
                    )
                    && attempt.retain_until_ms > 0
                    && !attempt.sealed_journal.is_empty()
                    && attempt.sealed_journal.len()
                        <= MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES
                    && attempt.journal_commitment
                        == sealed_record_commitment(&attempt.sealed_journal) =>
            {
                Ok(())
            }
            _ => Err(BlindVaultReplicaRecoveryStoreError::CorruptState),
        }
    }
}

fn valid_resolved_attempt_high_water(
    accepted_journal_sequence: u64,
    last_resolved_attempt: Option<StoredResolvedAttemptV1>,
) -> bool {
    match last_resolved_attempt {
        None => accepted_journal_sequence == 0,
        Some(resolved) => {
            resolved.journal_sequence > 0 && resolved.journal_sequence == accepted_journal_sequence
        }
    }
}

fn valid_prior_resolved_attempt(
    accepted_journal_sequence: u64,
    last_resolved_attempt: Option<StoredResolvedAttemptV1>,
) -> bool {
    match last_resolved_attempt {
        None => true,
        Some(resolved) => {
            resolved.journal_sequence > 0 && resolved.journal_sequence < accepted_journal_sequence
        }
    }
}

fn encode_state(
    state: &StoredRecoveryStateV1,
) -> Result<Zeroizing<Vec<u8>>, BlindVaultReplicaRecoveryStoreError> {
    let body = Zeroizing::new(
        recovery_options()
            .serialize(state)
            .map_err(|_| BlindVaultReplicaRecoveryStoreError::TooLarge)?,
    );
    let body_length =
        u32::try_from(body.len()).map_err(|_| BlindVaultReplicaRecoveryStoreError::TooLarge)?;
    let total_length = RECOVERY_FILE_HEADER_BYTES
        .checked_add(body.len())
        .and_then(|length| length.checked_add(RECOVERY_FILE_CHECKSUM_BYTES))
        .ok_or(BlindVaultReplicaRecoveryStoreError::TooLarge)?;
    if total_length > MAX_RECOVERY_FILE_BYTES {
        return Err(BlindVaultReplicaRecoveryStoreError::TooLarge);
    }

    let mut encoded = Zeroizing::new(Vec::with_capacity(total_length));
    encoded.extend_from_slice(&RECOVERY_FILE_MAGIC);
    encoded.extend_from_slice(&RECOVERY_FILE_VERSION_V1.to_be_bytes());
    encoded.extend_from_slice(&body_length.to_be_bytes());
    encoded.extend_from_slice(body.as_slice());
    let checksum = recovery_file_checksum(&encoded);
    encoded.extend_from_slice(&checksum);
    Ok(encoded)
}

fn decode_state(
    encoded: &[u8],
) -> Result<StoredRecoveryStateV1, BlindVaultReplicaRecoveryStoreError> {
    if encoded.len() < RECOVERY_FILE_HEADER_BYTES + RECOVERY_FILE_CHECKSUM_BYTES
        || encoded[..4] != RECOVERY_FILE_MAGIC
    {
        return Err(BlindVaultReplicaRecoveryStoreError::CorruptState);
    }
    let version = u16::from_be_bytes(
        encoded[4..6]
            .try_into()
            .map_err(|_| BlindVaultReplicaRecoveryStoreError::CorruptState)?,
    );
    if version != RECOVERY_FILE_VERSION_V1 {
        return Err(BlindVaultReplicaRecoveryStoreError::CorruptState);
    }
    let body_length = usize::try_from(u32::from_be_bytes(
        encoded[6..10]
            .try_into()
            .map_err(|_| BlindVaultReplicaRecoveryStoreError::CorruptState)?,
    ))
    .map_err(|_| BlindVaultReplicaRecoveryStoreError::TooLarge)?;
    let checksum_offset = RECOVERY_FILE_HEADER_BYTES
        .checked_add(body_length)
        .ok_or(BlindVaultReplicaRecoveryStoreError::TooLarge)?;
    if checksum_offset + RECOVERY_FILE_CHECKSUM_BYTES != encoded.len() {
        return Err(BlindVaultReplicaRecoveryStoreError::CorruptState);
    }
    let expected_checksum = recovery_file_checksum(&encoded[..checksum_offset]);
    if encoded[checksum_offset..] != expected_checksum {
        return Err(BlindVaultReplicaRecoveryStoreError::CorruptState);
    }
    let state = recovery_options()
        .deserialize::<StoredRecoveryStateV1>(&encoded[RECOVERY_FILE_HEADER_BYTES..checksum_offset])
        .map_err(|_| BlindVaultReplicaRecoveryStoreError::CorruptState)?;
    state.validate()?;
    Ok(state)
}

fn recovery_file_checksum(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(RECOVERY_FILE_CHECKSUM_DOMAIN);
    hasher.update(bytes);
    hasher.finalize().into()
}

fn sealed_record_commitment(sealed: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SEALED_RECORD_COMMITMENT_DOMAIN);
    hasher.update((sealed.len() as u64).to_be_bytes());
    hasher.update(sealed);
    hasher.finalize().into()
}

fn recovery_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_RECOVERY_FILE_BYTES as u64)
        .reject_trailing_bytes()
}

// [BLIND-VAULT-RECOVERY-HARDENING 2026-08-31 by Codex] These tests freeze V1
// fail-closed corruption checks and crash-restart transition invariants using
// only opaque sealed byte containers.
#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use tempfile::TempDir;

    use super::*;

    const WORKFLOW_ID: [u8; 16] = [0x61; 16];
    const SEALED_SNAPSHOT_A: &[u8] = &[0xa1, 0x00, 0x7f, 0x93, 0x28];
    const SEALED_SNAPSHOT_B: &[u8] = &[0xb2, 0x10, 0x6e, 0x84, 0x39];
    const SEALED_SNAPSHOT_C: &[u8] = &[0xc3, 0x20, 0x5d, 0x75, 0x4a];
    const SEALED_JOURNAL: &[u8] = &[0xd4, 0x30, 0x4c, 0x66, 0x5b, 0xe1];
    const STATE_FILE_NAME: &str = "recovery-state-v1.bin";
    const GOLDEN_V1_ENCODED: &[u8] = &[
        0x41, 0x58, 0x56, 0x52, 0x00, 0x01, 0x00, 0x00, 0x00, 0xbc, 0x61, 0x61, 0x61, 0x61, 0x61,
        0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x02, 0x00, 0x00, 0x00,
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12,
        0x11, 0x01, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
        0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
        0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0xe4, 0x5c, 0x60,
        0x25, 0xc0, 0x9c, 0xc9, 0x38, 0xbc, 0x8b, 0x57, 0x08, 0xc6, 0x59, 0xb5, 0x34, 0x08, 0x31,
        0xe4, 0xc3, 0x84, 0xfa, 0x4e, 0x18, 0xfb, 0x7d, 0x7c, 0xae, 0x96, 0x66, 0xe5, 0xc0, 0x05,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa1, 0x00, 0x7f, 0x93, 0x28, 0x01, 0x45, 0x23,
        0x07, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23,
        0x22, 0x21, 0x0b, 0x32, 0x2e, 0xfc, 0x6f, 0x88, 0x8a, 0xc6, 0x68, 0x6d, 0xd7, 0xf3, 0x3d,
        0x3f, 0xae, 0x69, 0x62, 0x35, 0xe9, 0x83, 0xfb, 0x7a, 0xae, 0xb8, 0x96, 0xda, 0x8e, 0xd3,
        0xdc, 0xff, 0xe6, 0xca, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd4, 0x30, 0x4c,
        0x66, 0x5b, 0xe1, 0x12, 0x89, 0x27, 0x6b, 0x45, 0x83, 0x1a, 0xcf, 0x38, 0x3c, 0x84, 0xbf,
        0x5c, 0xa5, 0x90, 0xe5, 0xcb, 0xde, 0x0c, 0xaf, 0xc9, 0xea, 0x12, 0x57, 0x3e, 0x45, 0xb8,
        0xeb, 0xb1, 0x59, 0x95, 0x76,
    ];

    fn resolved_state(
        snapshot_sequence: u64,
        accepted_journal_sequence: u64,
        last_resolved_attempt: Option<StoredResolvedAttemptV1>,
    ) -> StoredRecoveryStateV1 {
        StoredRecoveryStateV1 {
            workflow_id: WORKFLOW_ID,
            phase: StoredAttemptPhaseV1::Resolved,
            snapshot_sequence,
            accepted_journal_sequence,
            last_resolved_attempt,
            snapshot_commitment: sealed_record_commitment(SEALED_SNAPSHOT_A),
            sealed_snapshot: SEALED_SNAPSHOT_A.to_vec(),
            attempt: None,
        }
    }

    fn unresolved_state(
        phase: StoredAttemptPhaseV1,
        snapshot_sequence: u64,
        journal_sequence: u64,
    ) -> StoredRecoveryStateV1 {
        StoredRecoveryStateV1 {
            workflow_id: WORKFLOW_ID,
            phase,
            snapshot_sequence,
            accepted_journal_sequence: journal_sequence,
            last_resolved_attempt: None,
            snapshot_commitment: sealed_record_commitment(SEALED_SNAPSHOT_A),
            sealed_snapshot: SEALED_SNAPSHOT_A.to_vec(),
            attempt: Some(StoredAttemptV1 {
                work_sequence: 3,
                attempt: 2,
                journal_sequence,
                retain_until_ms: 1_900_000_000_000,
                journal_commitment: sealed_record_commitment(SEALED_JOURNAL),
                sealed_journal: SEALED_JOURNAL.to_vec(),
            }),
        }
    }

    fn golden_v1_state() -> StoredRecoveryStateV1 {
        let accepted_journal_sequence = 0x1112_1314_1516_1718;
        StoredRecoveryStateV1 {
            workflow_id: WORKFLOW_ID,
            phase: StoredAttemptPhaseV1::Committed,
            snapshot_sequence: 0x0102_0304_0506_0708,
            accepted_journal_sequence,
            last_resolved_attempt: Some(StoredResolvedAttemptV1 {
                journal_sequence: 0x0102_0304_0506_0708,
                journal_commitment: [0x5a; 32],
            }),
            snapshot_commitment: sealed_record_commitment(SEALED_SNAPSHOT_A),
            sealed_snapshot: SEALED_SNAPSHOT_A.to_vec(),
            attempt: Some(StoredAttemptV1 {
                work_sequence: 0x2345,
                attempt: 7,
                journal_sequence: accepted_journal_sequence,
                retain_until_ms: 0x2122_2324_2526_2728,
                journal_commitment: sealed_record_commitment(SEALED_JOURNAL),
                sealed_journal: SEALED_JOURNAL.to_vec(),
            }),
        }
    }

    fn recovery_directory(root: &TempDir) -> PathBuf {
        // [BLIND-VAULT-RECOVERY-MACOS-FIXTURE 2026-08-31 by Codex] Resolve
        // tempfile's `/var` alias before the production O_NOFOLLOW checks.
        std::fs::canonicalize(root.path())
            .expect("canonical temporary recovery root")
            .join("recovery")
    }

    fn publish_then_reopen(
        directory: &Path,
        state: &StoredRecoveryStateV1,
    ) -> FileBlindVaultReplicaRecoveryStore {
        {
            let store = FileBlindVaultReplicaRecoveryStore::open(directory)
                .expect("open recovery generation for publication");
            store.publish(state).expect("publish recovery generation");
        }
        FileBlindVaultReplicaRecoveryStore::open(directory)
            .expect("reopen published recovery generation")
    }

    fn assert_corrupt(encoded: &[u8]) {
        assert!(matches!(
            decode_state(encoded),
            Err(BlindVaultReplicaRecoveryStoreError::CorruptState)
        ));
    }

    fn rewrite_checksum(encoded: &mut [u8]) {
        let checksum_offset = encoded.len() - RECOVERY_FILE_CHECKSUM_BYTES;
        let checksum = recovery_file_checksum(&encoded[..checksum_offset]);
        encoded[checksum_offset..].copy_from_slice(&checksum);
    }

    #[test]
    fn v1_encoding_matches_frozen_golden_vector_and_roundtrips_exactly() {
        // [BLIND-VAULT-RECOVERY-GOLDEN-V1 2026-08-31 by Codex] V1 serde field
        // order is a disk ABI. Adding even an optional field or reordering a
        // field must fail this test; evolve under a new version instead of
        // refreshing this vector while RECOVERY_FILE_VERSION_V1 remains 1.
        let encoded = encode_state(&golden_v1_state()).expect("encode frozen V1 fixture");
        assert_eq!(encoded.as_slice(), GOLDEN_V1_ENCODED);
        assert_eq!(&GOLDEN_V1_ENCODED[..4], RECOVERY_FILE_MAGIC);
        assert_eq!(
            u16::from_be_bytes(GOLDEN_V1_ENCODED[4..6].try_into().unwrap()),
            RECOVERY_FILE_VERSION_V1
        );

        let body_length = usize::try_from(u32::from_be_bytes(
            GOLDEN_V1_ENCODED[6..RECOVERY_FILE_HEADER_BYTES]
                .try_into()
                .unwrap(),
        ))
        .unwrap();
        let checksum_offset = RECOVERY_FILE_HEADER_BYTES + body_length;
        assert_eq!(body_length, 188);
        assert_eq!(
            GOLDEN_V1_ENCODED.len(),
            checksum_offset + RECOVERY_FILE_CHECKSUM_BYTES
        );
        assert_eq!(
            &GOLDEN_V1_ENCODED[checksum_offset..],
            recovery_file_checksum(&GOLDEN_V1_ENCODED[..checksum_offset])
        );

        let decoded = decode_state(GOLDEN_V1_ENCODED).expect("decode frozen V1 fixture");
        let reencoded = encode_state(&decoded).expect("re-encode frozen V1 fixture");
        assert_eq!(reencoded.as_slice(), GOLDEN_V1_ENCODED);
    }

    #[test]
    fn v1_decoder_rejects_header_length_checksum_trailing_body_and_commitment_damage() {
        // Every V1 structural layer fails closed independently; commitment
        // cases retain a valid outer checksum so that inner check is exercised.
        let state = resolved_state(7, 0, None);
        let encoded = encode_state(&state).expect("encode valid V1 generation");
        decode_state(encoded.as_slice()).expect("decode valid V1 generation");
        assert_corrupt(&encoded[..RECOVERY_FILE_HEADER_BYTES]);

        let mut damaged_magic = encoded.to_vec();
        damaged_magic[0] ^= 0xff;
        assert_corrupt(&damaged_magic);

        let mut unsupported_version = encoded.to_vec();
        unsupported_version[4..6]
            .copy_from_slice(&RECOVERY_FILE_VERSION_V1.saturating_add(1).to_be_bytes());
        assert_corrupt(&unsupported_version);

        let mut invalid_length = encoded.to_vec();
        let body_length = u32::from_be_bytes(invalid_length[6..10].try_into().unwrap());
        invalid_length[6..10].copy_from_slice(&body_length.saturating_add(1).to_be_bytes());
        assert_corrupt(&invalid_length);

        let mut damaged_checksum = encoded.to_vec();
        let checksum_byte = damaged_checksum.len() - 1;
        damaged_checksum[checksum_byte] ^= 0x01;
        assert_corrupt(&damaged_checksum);

        let mut trailing_bytes = encoded.to_vec();
        trailing_bytes.push(0);
        assert_corrupt(&trailing_bytes);

        let mut invalid_body = resolved_state(7, 0, None);
        invalid_body.workflow_id = [0; 16];
        assert_corrupt(
            encode_state(&invalid_body)
                .expect("encode checksummed invalid body")
                .as_slice(),
        );

        let mut stale_snapshot_commitment = resolved_state(7, 0, None);
        stale_snapshot_commitment.sealed_snapshot[0] ^= 0x80;
        assert_corrupt(
            encode_state(&stale_snapshot_commitment)
                .expect("encode stale snapshot commitment")
                .as_slice(),
        );

        let mut stale_journal_commitment = unresolved_state(StoredAttemptPhaseV1::Prepared, 7, 11);
        stale_journal_commitment
            .attempt
            .as_mut()
            .unwrap()
            .sealed_journal[0] ^= 0x40;
        assert_corrupt(
            encode_state(&stale_journal_commitment)
                .expect("encode stale journal commitment")
                .as_slice(),
        );

        let mut malformed_body = encoded.to_vec();
        malformed_body[RECOVERY_FILE_HEADER_BYTES + 16] = 0xff;
        rewrite_checksum(&mut malformed_body);
        assert_corrupt(&malformed_body);
    }

    #[test]
    fn restart_load_preserves_prepared_committed_and_resolved_phases() {
        let cases = [
            (
                unresolved_state(StoredAttemptPhaseV1::Prepared, 7, 11),
                BlindVaultReplicaAttemptDurabilityPhase::Prepared,
                7,
                Some(SEALED_JOURNAL),
            ),
            (
                unresolved_state(StoredAttemptPhaseV1::Committed, 8, 11),
                BlindVaultReplicaAttemptDurabilityPhase::Committed,
                8,
                Some(SEALED_JOURNAL),
            ),
            (
                resolved_state(
                    9,
                    11,
                    Some(StoredResolvedAttemptV1 {
                        journal_sequence: 11,
                        journal_commitment: sealed_record_commitment(SEALED_JOURNAL),
                    }),
                ),
                BlindVaultReplicaAttemptDurabilityPhase::Resolved,
                9,
                None,
            ),
        ];

        for (state, expected_phase, expected_snapshot_sequence, expected_journal) in cases {
            let root = tempfile::tempdir().expect("temporary recovery root");
            let directory = recovery_directory(&root);
            let mut reopened = publish_then_reopen(&directory, &state);
            let loaded = reopened
                .load_recovery_state()
                .expect("load recovery state after restart")
                .expect("published recovery state");

            assert_eq!(loaded.phase(), expected_phase);
            assert_eq!(
                loaded.accepted_snapshot_sequence(),
                expected_snapshot_sequence
            );
            assert_eq!(loaded.accepted_journal_sequence(), 11);
            assert_eq!(loaded.sealed_snapshot(), SEALED_SNAPSHOT_A);
            assert_eq!(loaded.sealed_attempt_journal(), expected_journal);
        }
    }

    #[test]
    fn exact_retries_are_idempotent_while_conflicts_and_rollbacks_fail_closed() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let directory = recovery_directory(&root);
        let mut store =
            FileBlindVaultReplicaRecoveryStore::open(&directory).expect("open recovery generation");
        let snapshot =
            BlindVaultReplicaSnapshotRecord::new(WORKFLOW_ID, 7, SEALED_SNAPSHOT_A).unwrap();
        store.persist_snapshot(&snapshot).expect("persist snapshot");
        store
            .persist_snapshot(&snapshot)
            .expect("exact snapshot retry is idempotent");

        let conflicting_snapshot =
            BlindVaultReplicaSnapshotRecord::new(WORKFLOW_ID, 7, SEALED_SNAPSHOT_B).unwrap();
        assert!(matches!(
            store.persist_snapshot(&conflicting_snapshot),
            Err(BlindVaultReplicaRecoveryStoreError::RollbackDetected)
        ));
        let rollback_snapshot =
            BlindVaultReplicaSnapshotRecord::new(WORKFLOW_ID, 6, SEALED_SNAPSHOT_A).unwrap();
        assert!(matches!(
            store.persist_snapshot(&rollback_snapshot),
            Err(BlindVaultReplicaRecoveryStoreError::RollbackDetected)
        ));
        drop(store);

        let committed = unresolved_state(StoredAttemptPhaseV1::Committed, 8, 11);
        let journal_commitment = committed.attempt.as_ref().unwrap().journal_commitment;
        let mut store = publish_then_reopen(&directory, &committed);
        let resolved_snapshot =
            BlindVaultReplicaSnapshotRecord::new(WORKFLOW_ID, 9, SEALED_SNAPSHOT_C).unwrap();
        store
            .resolve_attempt(&resolved_snapshot, 11, journal_commitment)
            .expect("resolve committed attempt");
        drop(store);

        let mut reopened = FileBlindVaultReplicaRecoveryStore::open(&directory)
            .expect("reopen resolved generation");
        reopened
            .resolve_attempt(&resolved_snapshot, 11, journal_commitment)
            .expect("exact resolution retry is idempotent");
        assert!(matches!(
            reopened.resolve_attempt(&resolved_snapshot, 11, [0x55; 32]),
            Err(BlindVaultReplicaRecoveryStoreError::RollbackDetected)
        ));
    }

    #[test]
    fn journal_commitment_phase_and_attempt_inconsistencies_fail_closed() {
        let mut prepared = unresolved_state(StoredAttemptPhaseV1::Prepared, 7, 11);
        let correct_commitment = prepared.attempt.as_ref().unwrap().journal_commitment;
        let root = tempfile::tempdir().expect("temporary recovery root");
        let directory = recovery_directory(&root);
        let mut store = publish_then_reopen(&directory, &prepared);
        assert!(matches!(
            store.abort_prepared_attempt(11, [0x66; 32]),
            Err(BlindVaultReplicaRecoveryStoreError::StateConflict)
        ));
        store
            .abort_prepared_attempt(11, correct_commitment)
            .expect("abort exact prepared attempt");
        store
            .abort_prepared_attempt(11, correct_commitment)
            .expect("exact abort retry is idempotent");

        let mut resolved_with_attempt = resolved_state(7, 0, None);
        resolved_with_attempt.attempt = prepared.attempt.take();
        assert_corrupt(
            encode_state(&resolved_with_attempt)
                .expect("encode resolved state with attempt")
                .as_slice(),
        );

        let mut prepared_without_attempt = resolved_state(7, 11, None);
        prepared_without_attempt.phase = StoredAttemptPhaseV1::Prepared;
        assert_corrupt(
            encode_state(&prepared_without_attempt)
                .expect("encode prepared state without attempt")
                .as_slice(),
        );

        let mut attempt_sequence_mismatch =
            unresolved_state(StoredAttemptPhaseV1::Committed, 8, 11);
        attempt_sequence_mismatch
            .attempt
            .as_mut()
            .unwrap()
            .journal_sequence = 10;
        assert_corrupt(
            encode_state(&attempt_sequence_mismatch)
                .expect("encode mismatched attempt sequence")
                .as_slice(),
        );

        let committed = unresolved_state(StoredAttemptPhaseV1::Committed, 8, 12);
        let committed_commitment = committed.attempt.as_ref().unwrap().journal_commitment;
        let root = tempfile::tempdir().expect("temporary committed root");
        let directory = recovery_directory(&root);
        let mut committed_store = publish_then_reopen(&directory, &committed);
        assert!(matches!(
            committed_store.abort_prepared_attempt(12, committed_commitment),
            Err(BlindVaultReplicaRecoveryStoreError::StateConflict)
        ));
    }

    #[test]
    fn open_rejects_a_checksummed_but_commitment_corrupt_generation() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let directory = recovery_directory(&root);
        let state = resolved_state(7, 0, None);
        {
            let store = FileBlindVaultReplicaRecoveryStore::open(&directory)
                .expect("open recovery generation");
            store.publish(&state).expect("publish recovery generation");
        }

        let state_path = directory.join(STATE_FILE_NAME);
        let mut stale_commitment = state;
        stale_commitment.sealed_snapshot[0] ^= 0x20;
        let corrupted = encode_state(&stale_commitment)
            .expect("encode checksummed commitment-corrupt generation");
        fs::write(&state_path, corrupted).expect("write corrupt recovery generation");

        assert!(matches!(
            FileBlindVaultReplicaRecoveryStore::open(&directory),
            Err(BlindVaultReplicaRecoveryStoreError::CorruptState)
        ));
    }
}
