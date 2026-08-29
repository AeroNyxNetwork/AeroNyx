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
//! Last Modified: v1.1.0-ExactCrashRetry - Made every durable transition
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
            None => StoredRecoveryStateV1::from_snapshot(snapshot, 0),
            Some(current) => {
                current.require_workflow(snapshot.workflow_id())?;
                if current.phase != StoredAttemptPhaseV1::Resolved || current.attempt.is_some() {
                    return Err(BlindVaultReplicaRecoveryStoreError::InvalidTransition);
                }
                if current.snapshot_matches(snapshot) {
                    return Ok(());
                }
                current.require_new_snapshot_sequence(snapshot.snapshot_sequence())?;
                StoredRecoveryStateV1::from_snapshot(snapshot, current.accepted_journal_sequence)
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
            && current.accepted_journal_sequence == journal_sequence
            && current.snapshot_matches(snapshot)
        {
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
    ) -> Self {
        Self {
            workflow_id: snapshot.workflow_id(),
            phase: StoredAttemptPhaseV1::Resolved,
            snapshot_sequence: snapshot.snapshot_sequence(),
            accepted_journal_sequence,
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
            (StoredAttemptPhaseV1::Resolved, None) => Ok(()),
            (StoredAttemptPhaseV1::Prepared | StoredAttemptPhaseV1::Committed, Some(attempt))
                if attempt.attempt > 0
                    && attempt.journal_sequence > 0
                    && attempt.journal_sequence == self.accepted_journal_sequence
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
