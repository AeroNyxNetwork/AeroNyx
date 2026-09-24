// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_coordinator_authority.rs
// ============================================
//! Durable coordinator authority, handover, and witness-lease boundary.
//!
//! [MEMCHAIN-COORDINATOR-AUTHORITY-SPLIT 2026-09-25 by Codex] This module
//! owns the transaction-scoped authority history audit, exact-height proposer
//! resolution, duplicate-coordinator lease fencing, and production gates.
//! Stable storage_ops re-exports preserve all existing callers and contracts.

use std::fs::{File, OpenOptions, Permissions};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use nix::errno::Errno;
use nix::fcntl::{openat, Flock, FlockArg, OFlag};
use nix::sys::stat::Mode;
use rusqlite::{params, OptionalExtension};

use aeronyx_core::ledger::{
    RecordCoordinatorHandoverV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH,
};
use aeronyx_core::protocol::memchain::{
    MAX_COORDINATOR_LEASE_TTL_SECS_V1, MIN_COORDINATOR_LEASE_TTL_SECS_V1,
};

use super::storage::{
    probe_storage_database_file_identity, MemoryStorage, RecordCommitmentWitnessLeaseClockRuntime,
    RecordCommitmentWitnessLeaseHold, StorageDatabaseFileIdentity,
};
use super::storage_ops::{
    read_record_commitment_tip_transaction, unix_now_secs, COORDINATOR_LEASE_HANDOVER_GRACE_SECS,
    MAX_COORDINATOR_HANDOVER_HISTORY, MAX_STORED_COORDINATOR_HANDOVER_BYTES,
    WITNESS_LEASE_RESTART_HOLD_SECS,
};

/// Result of one serialized durable witness lease decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordCoordinatorLeaseGrantOutcome {
    /// This process instance owns the witness lease until the signed expiry.
    Granted {
        /// Monotonic lease generation stored by this witness.
        lease_epoch: u64,
        /// Witness wall-clock expiry used for restart-safe refusal.
        lease_expires_at: u64,
    },
    /// The witness chain advanced or changed before the lease transaction.
    TipMismatch,
    /// A different coordinator identity or process still owns the chain lease.
    Contended,
}

/// Result of one authenticated durable lease release decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordCoordinatorLeaseReleaseOutcome {
    /// The matching process lease is durably marked released.
    Released {
        /// Lease generation released by this witness.
        lease_epoch: u64,
        /// Witness wall-clock release time.
        released_at: u64,
    },
    /// No matching instance currently owns the coordinator row.
    NotHolder,
}

/// Result of one append-only coordinator authority transition decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordCoordinatorHandoverPersistOutcome {
    /// A new contiguous dual-signed transition was committed.
    Inserted,
    /// The exact same transition was already durable at this epoch.
    AlreadyPresent,
}

/// Audited proposer authority effective for the next local commitment height.
///
/// [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] This process-internal
/// value may guide authenticated node transport, but must not be exported in
/// public telemetry because coordinator identities are operational metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordCommitmentAuthorityState {
    /// Highest contiguous handover epoch already durable locally.
    pub authority_epoch: u64,
    /// Coordinator authorised to propose `next_block_height`.
    pub coordinator: [u8; 32],
    /// Exact next height after the fully audited local tip.
    pub next_block_height: u64,
}

/// One fixed-size authenticated authority-history page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordCoordinatorHandoverPage {
    /// Exact next proof after the requested epoch, or `None` at history head.
    pub handover: Option<RecordCoordinatorHandoverV1>,
    /// Highest contiguous authority epoch in the audited local store.
    pub latest_authority_epoch: u64,
}

#[derive(Clone, Copy)]
struct StoredCoordinatorLease {
    coordinator: [u8; 32],
    chain_id: [u8; 32],
    instance_id: [u8; 32],
    lease_epoch: u64,
    lease_expires_at: u64,
    updated_at: u64,
}

impl StoredCoordinatorLease {
    fn is_released(self) -> bool {
        self.lease_expires_at <= self.updated_at
    }
}

#[derive(Clone, Copy)]
pub(super) enum WitnessLeaseCommitObservation {
    Observed,
    #[cfg(test)]
    UnknownAfterCommit,
}

#[derive(Clone, Copy)]
enum WitnessLeaseClockAcquireError {
    Contended,
    UnsafeFile,
    Io,
}

impl WitnessLeaseClockAcquireError {
    const fn message(self) -> &'static str {
        match self {
            Self::Contended => "witness lease clock is held by another database handle",
            Self::UnsafeFile => "witness lease clock file is unsafe",
            Self::Io => "witness lease clock could not be acquired",
        }
    }
}

fn decode_stored_coordinator_lease(
    row: (Vec<u8>, Vec<u8>, Vec<u8>, i64, i64, i64),
) -> Result<StoredCoordinatorLease, String> {
    let (coordinator, chain_id, instance_id, lease_epoch, lease_expires_at, updated_at) = row;
    Ok(StoredCoordinatorLease {
        coordinator: coordinator.try_into().map_err(|value: Vec<u8>| {
            format!("coordinator lease identity length {}", value.len())
        })?,
        chain_id: chain_id.try_into().map_err(|value: Vec<u8>| {
            format!("coordinator lease chain id length {}", value.len())
        })?,
        instance_id: instance_id.try_into().map_err(|value: Vec<u8>| {
            format!("coordinator lease instance id length {}", value.len())
        })?,
        lease_epoch: u64::try_from(lease_epoch)
            .map_err(|_| "coordinator lease epoch is invalid".to_string())?,
        lease_expires_at: u64::try_from(lease_expires_at)
            .map_err(|_| "coordinator lease expiry is invalid".to_string())?,
        updated_at: u64::try_from(updated_at)
            .map_err(|_| "coordinator lease update time is invalid".to_string())?,
    })
}

fn witness_lease_deadline(now: Instant, hold_secs: u64) -> Result<Instant, String> {
    now.checked_add(Duration::from_secs(hold_secs))
        .ok_or_else(|| "witness lease monotonic deadline overflow".to_string())
}

fn initialize_witness_lease_chain_clock(
    runtime: &mut RecordCommitmentWitnessLeaseClockRuntime,
    chain_id: &[u8; 32],
    leases: &[StoredCoordinatorLease],
    now: Instant,
) -> Result<(), String> {
    if runtime.initialized_chains.contains(chain_id) {
        return Ok(());
    }
    let mut active = leases
        .iter()
        .copied()
        .filter(|lease| lease.chain_id == *chain_id && !lease.is_released());
    if let Some(first) = active.next() {
        let (coordinator, instance_id, lease_epoch) = if active.next().is_some() {
            // Multiple legacy unreleased rows are ambiguous. A holder that
            // matches either row must not renew through the restart fence.
            ([0; 32], [0; 32], first.lease_epoch)
        } else {
            (first.coordinator, first.instance_id, first.lease_epoch)
        };
        runtime.holds.insert(
            *chain_id,
            RecordCommitmentWitnessLeaseHold {
                coordinator,
                instance_id,
                lease_epoch,
                valid_until: witness_lease_deadline(now, WITNESS_LEASE_RESTART_HOLD_SECS)?,
            },
        );
    }
    runtime.initialized_chains.insert(*chain_id);
    Ok(())
}

/// Reads and cryptographically re-audits the complete authority history from
/// one SQLite snapshot.
///
/// [COORDINATOR-HANDOVER 2026-08-12 by Codex] Denormalized columns are checked
/// against the canonical bincode payload, and every transition is checked
/// against the exact retained predecessor block. This makes disk tampering,
/// skipped epochs, height gaps, key substitution, and alternate-prefix replay
/// fail closed before the latest coordinator can become runtime authority.
///
/// [HANDOVER-ACCEPTANCE-AUDIT 2026-08-14 by Codex] The local acceptance time is
/// also revalidated on every history read. Although it is not consensus data,
/// allowing it to precede the dual-signed issue time would make durable audit
/// metadata rollbackable after restart and weaken incident reconstruction.
pub(super) fn read_record_coordinator_handover_history_transaction(
    transaction: &rusqlite::Transaction<'_>,
    root_coordinator: &[u8; 32],
) -> Result<Vec<RecordCoordinatorHandoverV1>, String> {
    let rows = {
        let mut statement = transaction
            .prepare(
                "SELECT authority_epoch,activation_height,previous_height,
                        chain_id,protocol_version,previous_tip_hash,
                        previous_coordinator,next_coordinator,authorization_id,
                        issued_at,previous_signature,next_signature,payload,
                        accepted_at
                 FROM record_coordinator_handovers
                 ORDER BY authority_epoch ASC LIMIT ?1",
            )
            .map_err(|error| format!("prepare coordinator handover history: {error}"))?;
        let limit = i64::try_from(MAX_COORDINATOR_HANDOVER_HISTORY.saturating_add(1))
            .map_err(|_| "coordinator handover history bound overflow".to_string())?;
        let rows = statement
            .query_map(params![limit], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Vec<u8>>(5)?,
                    row.get::<_, Vec<u8>>(6)?,
                    row.get::<_, Vec<u8>>(7)?,
                    row.get::<_, Vec<u8>>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, Vec<u8>>(10)?,
                    row.get::<_, Vec<u8>>(11)?,
                    row.get::<_, Vec<u8>>(12)?,
                    row.get::<_, i64>(13)?,
                ))
            })
            .map_err(|error| format!("read coordinator handover history: {error}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("decode coordinator handover history row: {error}"))?;
        rows
    };
    if rows.len() > MAX_COORDINATOR_HANDOVER_HISTORY {
        return Err("coordinator handover history exceeds supported bound".to_string());
    }

    let mut current_epoch = 0u64;
    let mut current_coordinator = *root_coordinator;
    let mut history = Vec::with_capacity(rows.len());
    for (
        stored_epoch,
        stored_activation_height,
        stored_previous_height,
        stored_chain_id,
        stored_protocol_version,
        stored_previous_tip_hash,
        stored_previous_coordinator,
        stored_next_coordinator,
        stored_authorization_id,
        stored_issued_at,
        stored_previous_signature,
        stored_next_signature,
        payload,
        stored_accepted_at,
    ) in rows
    {
        if payload.len() > MAX_STORED_COORDINATOR_HANDOVER_BYTES {
            return Err("stored coordinator handover payload exceeds bound".to_string());
        }
        let proof = bincode::deserialize::<RecordCoordinatorHandoverV1>(&payload)
            .map_err(|_| "stored coordinator handover payload decode failed".to_string())?;
        let canonical_payload = bincode::serialize(&proof)
            .map_err(|_| "stored coordinator handover payload encode failed".to_string())?;
        if canonical_payload != payload {
            return Err("stored coordinator handover payload is non-canonical".to_string());
        }

        let authority_epoch = u64::try_from(stored_epoch)
            .map_err(|_| "stored coordinator handover epoch is invalid".to_string())?;
        let activation_height = u64::try_from(stored_activation_height)
            .map_err(|_| "stored coordinator handover activation is invalid".to_string())?;
        let previous_height = u64::try_from(stored_previous_height)
            .map_err(|_| "stored coordinator handover predecessor is invalid".to_string())?;
        let issued_at = u64::try_from(stored_issued_at)
            .map_err(|_| "stored coordinator handover issue time is invalid".to_string())?;
        let accepted_at = u64::try_from(stored_accepted_at)
            .map_err(|_| "stored coordinator handover acceptance time is invalid".to_string())?;
        if authority_epoch != proof.header.authority_epoch
            || activation_height != proof.header.activation_height
            || activation_height.checked_sub(1) != Some(previous_height)
            || stored_chain_id.as_slice() != proof.header.chain_id.as_slice()
            || stored_protocol_version != i64::from(proof.header.protocol_version)
            || stored_previous_tip_hash.as_slice() != proof.header.previous_tip_hash.as_slice()
            || stored_previous_coordinator.as_slice()
                != proof.header.previous_coordinator.as_slice()
            || stored_next_coordinator.as_slice() != proof.header.next_coordinator.as_slice()
            || stored_authorization_id.as_slice() != proof.header.authorization_id.as_slice()
            || issued_at != proof.header.issued_at
            || stored_previous_signature.as_slice() != proof.previous_signature.as_slice()
            || stored_next_signature.as_slice() != proof.next_signature.as_slice()
        {
            return Err(format!(
                "stored coordinator handover row mismatch at epoch {authority_epoch}"
            ));
        }
        if accepted_at == 0 || accepted_at < issued_at {
            return Err(format!(
                "stored coordinator handover acceptance time is invalid at epoch {authority_epoch}"
            ));
        }

        let previous_tip_hash: Vec<u8> = transaction
            .query_row(
                "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                params![stored_previous_height],
                |row| row.get(0),
            )
            .map_err(|error| {
                format!(
                    "read coordinator handover predecessor at height {previous_height}: {error}"
                )
            })?;
        let previous_tip_hash: [u8; 32] =
            previous_tip_hash.try_into().map_err(|value: Vec<u8>| {
                format!(
                    "coordinator handover predecessor hash length {}",
                    value.len()
                )
            })?;
        proof
            .verify_successor(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                current_epoch,
                &current_coordinator,
                previous_height,
                &previous_tip_hash,
            )
            .map_err(|error| {
                format!("coordinator handover audit failed at epoch {authority_epoch}: {error}")
            })?;
        current_epoch = authority_epoch;
        current_coordinator = proof.header.next_coordinator;
        history.push(proof);
    }
    // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] This helper audits
    // the bounded handover schedule only. Callers that establish or expose a
    // full-chain authority baseline separately invoke the O(blocks) proposer
    // scan; live append stays O(handovers) instead of degrading with height.
    Ok(history)
}

/// Verifies that every retained block was signed by the coordinator authorised
/// for that exact height.
///
/// Block cryptography and denormalized row integrity are covered by the normal
/// commitment audit. This authority audit supplies the missing historical key
/// schedule: root key before epoch one, then each dual-signed transition from
/// its exact activation height onward.
pub(super) fn verify_record_commitment_proposer_history_transaction(
    transaction: &rusqlite::Transaction<'_>,
    root_coordinator: &[u8; 32],
    history: &[RecordCoordinatorHandoverV1],
) -> Result<(), String> {
    let mut statement = transaction
        .prepare("SELECT height,proposer FROM record_commitment_blocks ORDER BY height ASC")
        .map_err(|error| format!("prepare commitment proposer authority audit: {error}"))?;
    let mut rows = statement
        .query([])
        .map_err(|error| format!("query commitment proposer authority audit: {error}"))?;
    let mut current_coordinator = *root_coordinator;
    let mut transitions = history.iter().peekable();
    while let Some(row) = rows
        .next()
        .map_err(|error| format!("read commitment proposer authority row: {error}"))?
    {
        let height = row
            .get::<_, i64>(0)
            .map_err(|error| format!("decode commitment proposer height: {error}"))?;
        let height = u64::try_from(height)
            .map_err(|_| "commitment proposer height is invalid".to_string())?;
        let proposer = row
            .get::<_, Vec<u8>>(1)
            .map_err(|error| format!("decode commitment proposer: {error}"))?;
        let proposer: [u8; 32] = proposer.try_into().map_err(|value: Vec<u8>| {
            format!(
                "commitment proposer length {} at height {height}",
                value.len()
            )
        })?;

        if let Some(transition) = transitions.peek() {
            if transition.header.activation_height < height {
                return Err(format!(
                    "coordinator handover activation was skipped before height {height}"
                ));
            }
            if transition.header.activation_height == height {
                current_coordinator = transition.header.next_coordinator;
                transitions.next();
            }
        }
        if proposer != current_coordinator {
            return Err(format!(
                "commitment block has an unauthorized proposer at height {height}"
            ));
        }
    }
    Ok(())
}

/// Resolves the coordinator authorised for one exact commitment height.
///
/// [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] The caller supplies a
/// history already verified from the immutable root. This resolver is used by
/// live append only; the startup verifier above deliberately retains its
/// stronger transition-consumption checks as an independent fail-closed audit.
pub(super) fn record_commitment_authority_at_height(
    root_coordinator: &[u8; 32],
    history: &[RecordCoordinatorHandoverV1],
    height: u64,
) -> (u64, [u8; 32]) {
    history
        .iter()
        .take_while(|transition| transition.header.activation_height <= height)
        .fold((0, *root_coordinator), |_, transition| {
            (
                transition.header.authority_epoch,
                transition.header.next_coordinator,
            )
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CoordinatorFenceAcquireError {
    Contended,
    UnsafeFile,
    Io,
}

impl CoordinatorFenceAcquireError {
    pub(super) const fn state(self) -> &'static str {
        match self {
            Self::Contended => "contended",
            Self::UnsafeFile | Self::Io => "failed",
        }
    }

    pub(super) const fn message(self) -> &'static str {
        match self {
            Self::Contended => {
                "commitment coordinator production fence is held by another local process"
            }
            Self::UnsafeFile => "commitment coordinator production fence file is unsafe",
            Self::Io => "commitment coordinator production fence could not be acquired",
        }
    }
}

pub(super) fn commitment_coordinator_fence_path(database_path: &Path) -> Result<PathBuf, String> {
    let file_name = database_path
        .file_name()
        .ok_or_else(|| "commitment coordinator database path has no file name".to_string())?;
    let mut lock_name = file_name.to_os_string();
    lock_name.push(".commitment-coordinator-v1.lock");
    Ok(database_path.with_file_name(lock_name))
}

#[cfg(target_os = "macos")]
const WITNESS_LEASE_CLOCK_DIRECTORY: &str = "/private/tmp";
#[cfg(not(target_os = "macos"))]
const WITNESS_LEASE_CLOCK_DIRECTORY: &str = "/tmp";

pub(super) fn commitment_witness_lease_clock_path(
    identity: StorageDatabaseFileIdentity,
) -> PathBuf {
    // [MEMCHAIN-WITNESS-DB-IDENTITY 2026-09-05 by Codex] A fixed host-local
    // namespace plus device/inode makes every safe spelling of one repository
    // contend on the same advisory lock. No database path enters the artifact.
    Path::new(WITNESS_LEASE_CLOCK_DIRECTORY).join(format!(
        ".aeronyx-commitment-witness-clock-v2-{:016x}-{:016x}.lock",
        identity.device, identity.inode
    ))
}

fn acquire_commitment_witness_lease_clock(
    identity: StorageDatabaseFileIdentity,
) -> Result<Flock<File>, WitnessLeaseClockAcquireError> {
    let path = commitment_witness_lease_clock_path(identity);
    let parent_path = Path::new(WITNESS_LEASE_CLOCK_DIRECTORY);
    let name = path.file_name().ok_or(WitnessLeaseClockAcquireError::Io)?;
    let parent = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_DIRECTORY)
        .open(parent_path)
        .map_err(|error| match error.raw_os_error() {
            Some(nix::libc::ELOOP | nix::libc::ENOTDIR) => {
                WitnessLeaseClockAcquireError::UnsafeFile
            }
            _ => WitnessLeaseClockAcquireError::Io,
        })?;
    let raw_fd = openat(
        Some(parent.as_raw_fd()),
        name,
        OFlag::O_RDWR | OFlag::O_CREAT | OFlag::O_CLOEXEC | OFlag::O_NOFOLLOW,
        Mode::from_bits_truncate(0o600),
    )
    .map_err(|error| match error {
        Errno::ELOOP | Errno::ENOTDIR => WitnessLeaseClockAcquireError::UnsafeFile,
        _ => WitnessLeaseClockAcquireError::Io,
    })?;
    // SAFETY: `openat` returned a new descriptor owned only by this value.
    let file = unsafe { File::from_raw_fd(raw_fd) };
    let metadata = file
        .metadata()
        .map_err(|_| WitnessLeaseClockAcquireError::Io)?;
    // Validate identity before chmod or lock effects. A hardlink to unrelated
    // same-owner state must not be normalized through this path.
    if !metadata.file_type().is_file()
        || metadata.nlink() != 1
        || metadata.uid() != unsafe { nix::libc::geteuid() }
    {
        return Err(WitnessLeaseClockAcquireError::UnsafeFile);
    }
    file.set_permissions(Permissions::from_mode(0o600))
        .map_err(|_| WitnessLeaseClockAcquireError::Io)?;
    let revalidated = file
        .metadata()
        .map_err(|_| WitnessLeaseClockAcquireError::Io)?;
    if !revalidated.file_type().is_file()
        || revalidated.nlink() != 1
        || revalidated.uid() != metadata.uid()
    {
        return Err(WitnessLeaseClockAcquireError::UnsafeFile);
    }
    let lock = Flock::lock(file, FlockArg::LockExclusiveNonblock).map_err(|(_, error)| {
        if error == Errno::EWOULDBLOCK {
            WitnessLeaseClockAcquireError::Contended
        } else {
            WitnessLeaseClockAcquireError::Io
        }
    })?;
    Ok(lock)
}

fn ensure_commitment_witness_lease_clock(
    runtime: &mut RecordCommitmentWitnessLeaseClockRuntime,
    database_path: Option<&Path>,
    database_identity: Option<StorageDatabaseFileIdentity>,
) -> Result<(), String> {
    let Some(database_path) = database_path else {
        return Ok(());
    };
    let identity =
        database_identity.ok_or_else(|| "witness lease database identity is unsafe".to_string())?;
    // [MEMCHAIN-WITNESS-DB-REVALIDATION 2026-09-13 by Codex] A retained
    // advisory-lock handle proves only that the original inode remains locked.
    // Re-probe the configured name before every lease mutation: an atomic
    // replacement otherwise lets the old SQLite handle renew an unlinked
    // database while a new handle acquires a separate inode-derived lock.
    if probe_storage_database_file_identity(database_path) != Ok(Some(identity)) {
        return Err("witness lease database identity is unsafe".to_string());
    }
    if runtime.handle.is_some() {
        return Ok(());
    }
    runtime.handle = Some(
        acquire_commitment_witness_lease_clock(identity)
            .map_err(|error| error.message().to_string())?,
    );
    Ok(())
}

pub(super) fn acquire_commitment_coordinator_fence(
    database_path: &Path,
) -> Result<Flock<File>, CoordinatorFenceAcquireError> {
    let path = commitment_coordinator_fence_path(database_path)
        .map_err(|_| CoordinatorFenceAcquireError::Io)?;
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.file_type().is_file() => {
            return Err(CoordinatorFenceAcquireError::UnsafeFile);
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(CoordinatorFenceAcquireError::Io),
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW)
        .open(&path)
        .map_err(|error| {
            if error.raw_os_error() == Some(nix::libc::ELOOP) {
                CoordinatorFenceAcquireError::UnsafeFile
            } else {
                CoordinatorFenceAcquireError::Io
            }
        })?;
    let metadata = file
        .metadata()
        .map_err(|_| CoordinatorFenceAcquireError::Io)?;
    if !metadata.file_type().is_file() {
        return Err(CoordinatorFenceAcquireError::UnsafeFile);
    }
    let lock = Flock::lock(file, FlockArg::LockExclusiveNonblock).map_err(|(_, error)| {
        if error == Errno::EWOULDBLOCK {
            CoordinatorFenceAcquireError::Contended
        } else {
            CoordinatorFenceAcquireError::Io
        }
    })?;
    lock.set_permissions(Permissions::from_mode(0o600))
        .map_err(|_| CoordinatorFenceAcquireError::Io)?;
    Ok(lock)
}

impl MemoryStorage {
    /// Returns the complete cryptographically audited coordinator history.
    ///
    /// The root key is an operator-configured trust anchor. Returned proofs
    /// contain only node control-plane keys and chain positions; no memory,
    /// owner, message, route, endpoint, or client metadata is involved.
    pub async fn record_coordinator_handover_history(
        &self,
        root_coordinator: &[u8; 32],
    ) -> Result<Vec<RecordCoordinatorHandoverV1>, String> {
        if self
            .commitment_authority_root
            .read()
            .is_some_and(|configured| configured != *root_coordinator)
        {
            return Err(
                "coordinator handover root does not match configured authority".to_string(),
            );
        }
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|error| format!("begin coordinator handover audit: {error}"))?;
        let integrity = (*self.commitment_integrity.read())
            .ok_or_else(|| "commitment chain is not fully audited".to_string())?;
        let (tip_height, tip_hash) =
            read_record_commitment_tip_transaction(&transaction, "coordinator handover audit")?;
        if integrity.verified_tip_height != tip_height
            || integrity.verified_tip_hash != tip_hash
            || integrity.verified_block_count != tip_height
        {
            return Err("commitment chain audit baseline is stale".to_string());
        }
        let history =
            read_record_coordinator_handover_history_transaction(&transaction, root_coordinator)?;
        verify_record_commitment_proposer_history_transaction(
            &transaction,
            root_coordinator,
            &history,
        )?;
        transaction
            .commit()
            .map_err(|error| format!("commit coordinator handover audit snapshot: {error}"))?;
        Ok(history)
    }

    /// Returns whether exact-height proposer authority is enforced.
    #[must_use]
    pub fn record_commitment_authority_enforced(&self) -> bool {
        self.commitment_authority_root.read().is_some()
    }

    /// Reads one audit-baseline-bound authority snapshot.
    ///
    /// [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] Runtime sync reads
    /// only the bounded handover schedule after startup established the full
    /// block audit baseline. This keeps each follower page `O(handovers)` while
    /// refusing stale or unaudited storage. No identity is logged or retained
    /// outside the process-local return value.
    async fn audited_record_commitment_authority_snapshot(
        &self,
    ) -> Result<Option<([u8; 32], u64, Vec<RecordCoordinatorHandoverV1>)>, String> {
        let Some(root_coordinator) = *self.commitment_authority_root.read() else {
            return Ok(None);
        };
        let integrity = (*self.commitment_integrity.read())
            .ok_or_else(|| "commitment chain is not fully audited".to_string())?;
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|error| format!("begin commitment authority snapshot: {error}"))?;
        let (tip_height, tip_hash) =
            read_record_commitment_tip_transaction(&transaction, "commitment authority")?;
        if integrity.verified_tip_height != tip_height
            || integrity.verified_tip_hash != tip_hash
            || integrity.verified_block_count != tip_height
        {
            return Err("commitment chain audit baseline is stale".to_string());
        }
        let history =
            read_record_coordinator_handover_history_transaction(&transaction, &root_coordinator)?;
        transaction
            .commit()
            .map_err(|error| format!("commit commitment authority snapshot: {error}"))?;
        Ok(Some((root_coordinator, tip_height, history)))
    }

    /// Returns the audited coordinator authorised for the next local block.
    pub async fn record_commitment_authority_state(
        &self,
    ) -> Result<Option<RecordCommitmentAuthorityState>, String> {
        let Some((root, tip_height, history)) =
            self.audited_record_commitment_authority_snapshot().await?
        else {
            return Ok(None);
        };
        let next_block_height = tip_height
            .checked_add(1)
            .ok_or_else(|| "commitment chain height exhausted".to_string())?;
        let (authority_epoch, coordinator) =
            record_commitment_authority_at_height(&root, &history, next_block_height);
        Ok(Some(RecordCommitmentAuthorityState {
            authority_epoch,
            coordinator,
            next_block_height,
        }))
    }

    /// Resolves the audited proposer authorised at one exact positive height.
    pub async fn record_commitment_authority_for_height(
        &self,
        height: u64,
    ) -> Result<Option<[u8; 32]>, String> {
        if height == 0 {
            return Err("commitment authority height must be positive".to_string());
        }
        let Some((root, _, history)) = self.audited_record_commitment_authority_snapshot().await?
        else {
            return Ok(None);
        };
        Ok(Some(
            record_commitment_authority_at_height(&root, &history, height).1,
        ))
    }

    /// Returns at most the exact next proof after `after_authority_epoch`.
    ///
    /// The fixed one-proof page lets cold followers verify the corresponding
    /// block prefix before accepting each transition. It also keeps the peer
    /// response size independent of total authority-history length.
    pub async fn next_record_coordinator_handover_page(
        &self,
        after_authority_epoch: u64,
    ) -> Result<RecordCoordinatorHandoverPage, String> {
        let Some((_, _, history)) = self.audited_record_commitment_authority_snapshot().await?
        else {
            return Err("commitment authority root is not configured".to_string());
        };
        let latest_authority_epoch = history
            .last()
            .map_or(0, |handover| handover.header.authority_epoch);
        if after_authority_epoch > latest_authority_epoch {
            return Err("requested authority epoch is ahead of local history".to_string());
        }
        let handover = if after_authority_epoch == latest_authority_epoch {
            None
        } else {
            let next_epoch = after_authority_epoch
                .checked_add(1)
                .ok_or_else(|| "authority epoch exhausted".to_string())?;
            Some(
                history
                    .iter()
                    .find(|handover| handover.header.authority_epoch == next_epoch)
                    .cloned()
                    .ok_or_else(|| "coordinator handover history is not contiguous".to_string())?,
            )
        };
        Ok(RecordCoordinatorHandoverPage {
            handover,
            latest_authority_epoch,
        })
    }

    /// Persists a handover against the immutable configured authority root.
    ///
    /// This wrapper prevents node-peer transport from supplying or replacing
    /// a trust anchor. The root remains process-local and is copied before the
    /// async persistence boundary so no lock guard crosses an await.
    pub async fn persist_configured_record_coordinator_handover(
        &self,
        proof: &RecordCoordinatorHandoverV1,
        accepted_at: u64,
    ) -> Result<RecordCoordinatorHandoverPersistOutcome, String> {
        let root = (*self.commitment_authority_root.read())
            .ok_or_else(|| "commitment authority root is not configured".to_string())?;
        self.persist_record_coordinator_handover(&root, proof, accepted_at)
            .await
    }

    /// Atomically verifies and appends one coordinator authority transition.
    ///
    /// [COORDINATOR-HANDOVER 2026-08-12 by Codex] The entire prior authority
    /// history is re-audited in the same `IMMEDIATE` transaction. The new
    /// proof must be the exact next epoch, activate at audited tip + 1, bind
    /// that tip hash, be signed by both keys, and observe an expired or
    /// explicitly released witness lease. Exact retries are idempotent.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or non-contiguous proofs, conflicting
    /// durable history, an active coordinator lease, a future issue time,
    /// storage corruption, integer overflow, or SQLite failure.
    pub async fn persist_record_coordinator_handover(
        &self,
        root_coordinator: &[u8; 32],
        proof: &RecordCoordinatorHandoverV1,
        accepted_at: u64,
    ) -> Result<RecordCoordinatorHandoverPersistOutcome, String> {
        // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] A caller cannot
        // substitute an alternate root after startup configuration. This is a
        // trust-anchor check, independent of the proof's two valid signatures.
        if self
            .commitment_authority_root
            .read()
            .is_some_and(|configured| configured != *root_coordinator)
        {
            return Err(
                "coordinator handover root does not match configured authority".to_string(),
            );
        }
        proof
            .verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID)
            .map_err(|error| format!("coordinator handover proof is invalid: {error}"))?;
        if accepted_at == 0 {
            return Err("coordinator handover acceptance time is invalid".to_string());
        }
        if proof.header.issued_at > accepted_at {
            return Err("coordinator handover issue time is in the future".to_string());
        }
        let payload = bincode::serialize(proof)
            .map_err(|error| format!("serialize coordinator handover: {error}"))?;
        if payload.len() > MAX_STORED_COORDINATOR_HANDOVER_BYTES {
            return Err("coordinator handover payload exceeds storage limit".to_string());
        }

        let accepted_at_i64 = i64::try_from(accepted_at)
            .map_err(|_| "coordinator handover acceptance exceeds SQLite range".to_string())?;
        let authority_epoch_i64 = i64::try_from(proof.header.authority_epoch)
            .map_err(|_| "coordinator handover epoch exceeds SQLite range".to_string())?;
        let activation_height_i64 = i64::try_from(proof.header.activation_height)
            .map_err(|_| "coordinator handover activation exceeds SQLite range".to_string())?;
        let previous_height = proof
            .header
            .activation_height
            .checked_sub(1)
            .ok_or_else(|| "coordinator handover predecessor underflow".to_string())?;
        let previous_height_i64 = i64::try_from(previous_height)
            .map_err(|_| "coordinator handover predecessor exceeds SQLite range".to_string())?;
        let issued_at_i64 = i64::try_from(proof.header.issued_at)
            .map_err(|_| "coordinator handover issue time exceeds SQLite range".to_string())?;

        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin coordinator handover transaction: {error}"))?;
        let integrity = (*self.commitment_integrity.read())
            .ok_or_else(|| "commitment chain is not fully audited".to_string())?;
        let (tip_height, tip_hash) =
            read_record_commitment_tip_transaction(&transaction, "coordinator handover")?;
        if integrity.verified_tip_height != tip_height
            || integrity.verified_tip_hash != tip_hash
            || integrity.verified_block_count != tip_height
        {
            return Err("commitment chain audit baseline is stale".to_string());
        }
        let history =
            read_record_coordinator_handover_history_transaction(&transaction, root_coordinator)?;
        verify_record_commitment_proposer_history_transaction(
            &transaction,
            root_coordinator,
            &history,
        )?;
        if let Some(existing) = history
            .iter()
            .find(|entry| entry.header.authority_epoch == proof.header.authority_epoch)
        {
            if existing == proof {
                transaction
                    .commit()
                    .map_err(|error| format!("commit idempotent coordinator handover: {error}"))?;
                return Ok(RecordCoordinatorHandoverPersistOutcome::AlreadyPresent);
            }
            return Err(format!(
                "conflicting coordinator handover at epoch {}",
                proof.header.authority_epoch
            ));
        }

        let (current_epoch, current_coordinator) =
            history.last().map_or((0, *root_coordinator), |entry| {
                (entry.header.authority_epoch, entry.header.next_coordinator)
            });
        proof
            .verify_successor(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                current_epoch,
                &current_coordinator,
                tip_height,
                &tip_hash,
            )
            .map_err(|error| format!("coordinator handover is not the next authority: {error}"))?;

        let leases = {
            let mut statement = transaction
                .prepare(
                    "SELECT lease_expires_at,updated_at
                     FROM record_coordinator_leases WHERE chain_id=?1",
                )
                .map_err(|error| format!("prepare coordinator handover lease check: {error}"))?;
            let rows = statement
                .query_map(params![proof.header.chain_id.as_slice()], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                })
                .map_err(|error| format!("read coordinator handover leases: {error}"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("decode coordinator handover lease: {error}"))?;
            rows
        };
        for (lease_expires_at, updated_at) in leases {
            let lease_expires_at = u64::try_from(lease_expires_at)
                .map_err(|_| "coordinator handover lease expiry is invalid".to_string())?;
            let updated_at = u64::try_from(updated_at)
                .map_err(|_| "coordinator handover lease update time is invalid".to_string())?;
            let explicitly_released = lease_expires_at <= updated_at;
            if !explicitly_released
                && accepted_at
                    < lease_expires_at.saturating_add(COORDINATOR_LEASE_HANDOVER_GRACE_SECS)
            {
                return Err(
                    "coordinator handover rejected while a witness lease remains active"
                        .to_string(),
                );
            }
        }

        transaction
            .execute(
                "INSERT INTO record_coordinator_handovers
                    (authority_epoch,activation_height,previous_height,chain_id,
                     protocol_version,previous_tip_hash,previous_coordinator,
                     next_coordinator,authorization_id,issued_at,
                     previous_signature,next_signature,payload,accepted_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                params![
                    authority_epoch_i64,
                    activation_height_i64,
                    previous_height_i64,
                    proof.header.chain_id.as_slice(),
                    i64::from(proof.header.protocol_version),
                    proof.header.previous_tip_hash.as_slice(),
                    proof.header.previous_coordinator.as_slice(),
                    proof.header.next_coordinator.as_slice(),
                    proof.header.authorization_id.as_slice(),
                    issued_at_i64,
                    proof.previous_signature.as_slice(),
                    proof.next_signature.as_slice(),
                    payload,
                    accepted_at_i64,
                ],
            )
            .map_err(|error| format!("persist coordinator handover: {error}"))?;
        transaction
            .commit()
            .map_err(|error| format!("commit coordinator handover: {error}"))?;
        Ok(RecordCoordinatorHandoverPersistOutcome::Inserted)
    }

    /// Atomically grants or renews one durable witness-side chain lease.
    ///
    /// [MEMCHAIN-CHAIN-LEASE 2026-08-12 by Codex] Lease ownership is scoped to
    /// the chain, not merely to a coordinator identity. This prevents old and
    /// replacement coordinator keys from holding overlapping leases during an
    /// identity handover. A successful grant also normalizes legacy per-key
    /// rows to one durable row while retaining a chain-wide monotonic epoch.
    /// The short handover grace prevents takeover at the exact expiry boundary.
    /// No endpoint, host identity, process id, user data, or block payload is
    /// stored.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid TTL, integer conversion failure,
    /// persisted chain mismatch, malformed existing row, or SQLite failure.
    pub async fn grant_record_commitment_coordinator_lease(
        &self,
        chain_id: &[u8; 32],
        coordinator: &[u8; 32],
        instance_id: &[u8; 32],
        expected_tip_height: u64,
        expected_tip_hash: &[u8; 32],
        now: u64,
        ttl_secs: u32,
    ) -> Result<RecordCoordinatorLeaseGrantOutcome, String> {
        self.grant_record_commitment_coordinator_lease_at(
            chain_id,
            coordinator,
            instance_id,
            expected_tip_height,
            expected_tip_hash,
            now,
            ttl_secs,
            Instant::now(),
            WitnessLeaseCommitObservation::Observed,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn grant_record_commitment_coordinator_lease_at(
        &self,
        chain_id: &[u8; 32],
        coordinator: &[u8; 32],
        instance_id: &[u8; 32],
        expected_tip_height: u64,
        expected_tip_hash: &[u8; 32],
        now: u64,
        ttl_secs: u32,
        monotonic_now: Instant,
        commit_observation: WitnessLeaseCommitObservation,
    ) -> Result<RecordCoordinatorLeaseGrantOutcome, String> {
        if !(MIN_COORDINATOR_LEASE_TTL_SECS_V1..=MAX_COORDINATOR_LEASE_TTL_SECS_V1)
            .contains(&ttl_secs)
        {
            return Err("coordinator lease TTL is outside the protocol bounds".to_string());
        }
        let now_i64 = i64::try_from(now)
            .map_err(|_| "coordinator lease time is outside SQLite range".to_string())?;
        let requested_expires_at = now
            .checked_add(u64::from(ttl_secs))
            .ok_or_else(|| "coordinator lease expiry overflow".to_string())?;

        // [MEMCHAIN-WITNESS-CLOCK 2026-09-05 by Codex] This mutex is the sole
        // witness-side clock authority. It remains held across the SQLite
        // transaction so durable and volatile lease decisions cannot interleave.
        let mut clock = self.commitment_witness_lease_clock.lock().await;
        ensure_commitment_witness_lease_clock(
            &mut clock,
            self.database_path.as_deref(),
            self.database_identity,
        )?;
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin coordinator lease transaction: {error}"))?;
        let persisted_tip = tx
            .query_row(
                "SELECT height, block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()
            .map_err(|error| format!("read coordinator lease chain tip: {error}"))?;
        let persisted_tip = match persisted_tip {
            Some((height, hash)) => {
                let height = u64::try_from(height)
                    .map_err(|_| "coordinator lease chain height is invalid".to_string())?;
                let hash: [u8; 32] = hash.try_into().map_err(|hash: Vec<u8>| {
                    format!("coordinator lease chain hash length {}", hash.len())
                })?;
                (height, hash)
            }
            None => (0, GENESIS_PREV_HASH),
        };
        if persisted_tip != (expected_tip_height, *expected_tip_hash) {
            return Ok(RecordCoordinatorLeaseGrantOutcome::TipMismatch);
        }
        let existing = {
            let mut statement = tx
                .prepare(
                    "SELECT coordinator, chain_id, instance_id, lease_epoch,
                            lease_expires_at, updated_at
                     FROM record_coordinator_leases
                     WHERE chain_id=?1 OR coordinator=?2",
                )
                .map_err(|error| format!("prepare coordinator chain leases: {error}"))?;
            let rows = statement
                .query_map(
                    params![chain_id.as_slice(), coordinator.as_slice()],
                    |row| {
                        Ok((
                            row.get::<_, Vec<u8>>(0)?,
                            row.get::<_, Vec<u8>>(1)?,
                            row.get::<_, Vec<u8>>(2)?,
                            row.get::<_, i64>(3)?,
                            row.get::<_, i64>(4)?,
                            row.get::<_, i64>(5)?,
                        ))
                    },
                )
                .map_err(|error| format!("read coordinator chain leases: {error}"))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("decode coordinator chain leases: {error}"))?
                .into_iter()
                .map(decode_stored_coordinator_lease)
                .collect::<Result<Vec<_>, _>>()?
        };

        initialize_witness_lease_chain_clock(&mut clock, chain_id, &existing, monotonic_now)?;
        if clock.holds.get(chain_id).is_some_and(|hold| {
            monotonic_now < hold.valid_until
                && (hold.coordinator != *coordinator || hold.instance_id != *instance_id)
        }) {
            return Ok(RecordCoordinatorLeaseGrantOutcome::Contended);
        }

        let mut maximum_epoch = None::<u64>;
        let mut matching_epoch = None::<u64>;
        let mut matching_expiry = None::<u64>;
        for stored in existing {
            if stored.coordinator == *coordinator && stored.chain_id != *chain_id {
                return Err("coordinator lease chain id mismatch".to_string());
            }
            if stored.chain_id != *chain_id {
                continue;
            }
            maximum_epoch = Some(maximum_epoch.map_or(stored.lease_epoch, |current| {
                current.max(stored.lease_epoch)
            }));

            let same_holder =
                stored.coordinator == *coordinator && stored.instance_id == *instance_id;
            if same_holder && stored.is_released() {
                return Ok(RecordCoordinatorLeaseGrantOutcome::Contended);
            }
            if !same_holder
                && !stored.is_released()
                && now
                    < stored
                        .lease_expires_at
                        .saturating_add(COORDINATOR_LEASE_HANDOVER_GRACE_SECS)
            {
                return Ok(RecordCoordinatorLeaseGrantOutcome::Contended);
            }
            if same_holder {
                matching_epoch = Some(stored.lease_epoch);
                matching_expiry = Some(stored.lease_expires_at);
            }
        }
        if let Some(hold) = clock.holds.get(chain_id) {
            if monotonic_now < hold.valid_until
                && hold.coordinator == *coordinator
                && hold.instance_id == *instance_id
                && matching_epoch != Some(hold.lease_epoch)
            {
                return Err("witness lease clock epoch does not match durable row".to_string());
            }
        }

        let lease_epoch = match (matching_epoch, maximum_epoch) {
            (Some(holder_epoch), Some(maximum_epoch)) if holder_epoch == maximum_epoch => {
                holder_epoch
            }
            (_, Some(maximum_epoch)) => maximum_epoch
                .checked_add(1)
                .ok_or_else(|| "coordinator lease epoch exhausted".to_string())?,
            (_, None) => 1,
        };
        let lease_epoch_i64 = i64::try_from(lease_epoch)
            .map_err(|_| "coordinator lease epoch is outside SQLite range".to_string())?;
        // A backwards wall step must not shorten an existing holder's durable
        // expiry even though the monotonic hold is independently renewed.
        let lease_expires_at = matching_expiry.map_or(requested_expires_at, |stored| {
            stored.max(requested_expires_at)
        });
        let lease_expires_at_i64 = i64::try_from(lease_expires_at)
            .map_err(|_| "coordinator lease expiry is outside SQLite range".to_string())?;
        tx.execute(
            "DELETE FROM record_coordinator_leases
             WHERE chain_id=?1 AND coordinator<>?2",
            params![chain_id.as_slice(), coordinator.as_slice()],
        )
        .map_err(|error| format!("normalize coordinator chain leases: {error}"))?;
        tx.execute(
            "INSERT INTO record_coordinator_leases
                (coordinator, chain_id, instance_id, lease_epoch, lease_expires_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(coordinator) DO UPDATE SET
                chain_id=excluded.chain_id,
                instance_id=excluded.instance_id,
                lease_epoch=excluded.lease_epoch,
                lease_expires_at=excluded.lease_expires_at,
                updated_at=excluded.updated_at",
            params![
                coordinator.as_slice(),
                chain_id.as_slice(),
                instance_id.as_slice(),
                lease_epoch_i64,
                lease_expires_at_i64,
                now_i64,
            ],
        )
        .map_err(|error| format!("persist coordinator lease: {error}"))?;
        let valid_until = witness_lease_deadline(
            monotonic_now,
            u64::from(ttl_secs).saturating_add(COORDINATOR_LEASE_HANDOVER_GRACE_SECS),
        )?;
        if let Err(error) = tx.commit() {
            // A failed commit can be ambiguous to the caller. Retain the
            // attempted holder locally for the full refusal window.
            clock.holds.insert(
                *chain_id,
                RecordCommitmentWitnessLeaseHold {
                    coordinator: *coordinator,
                    instance_id: *instance_id,
                    lease_epoch,
                    valid_until,
                },
            );
            return Err(format!("commit coordinator lease: {error}"));
        }
        clock.holds.insert(
            *chain_id,
            RecordCommitmentWitnessLeaseHold {
                coordinator: *coordinator,
                instance_id: *instance_id,
                lease_epoch,
                valid_until,
            },
        );
        #[cfg(test)]
        if matches!(
            commit_observation,
            WitnessLeaseCommitObservation::UnknownAfterCommit
        ) {
            return Err("coordinator lease commit outcome is unknown".to_string());
        }
        #[cfg(not(test))]
        let _ = commit_observation;
        Ok(RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch,
            lease_expires_at,
        })
    }

    /// Durably releases only the exact coordinator process instance.
    ///
    /// The row is retained so the witness keeps a monotonic generation and can
    /// reject a delayed renewal from the released instance. A released row is
    /// encoded as `lease_expires_at <= updated_at`; the next different process
    /// may therefore acquire immediately with the next lease epoch.
    pub async fn release_record_commitment_coordinator_lease(
        &self,
        chain_id: &[u8; 32],
        coordinator: &[u8; 32],
        instance_id: &[u8; 32],
        now: u64,
    ) -> Result<RecordCoordinatorLeaseReleaseOutcome, String> {
        self.release_record_commitment_coordinator_lease_at(
            chain_id,
            coordinator,
            instance_id,
            now,
            Instant::now(),
            WitnessLeaseCommitObservation::Observed,
        )
        .await
    }

    async fn release_record_commitment_coordinator_lease_at(
        &self,
        chain_id: &[u8; 32],
        coordinator: &[u8; 32],
        instance_id: &[u8; 32],
        now: u64,
        monotonic_now: Instant,
        commit_observation: WitnessLeaseCommitObservation,
    ) -> Result<RecordCoordinatorLeaseReleaseOutcome, String> {
        let now_i64 = i64::try_from(now)
            .map_err(|_| "coordinator lease release time is outside SQLite range".to_string())?;
        let mut clock = self.commitment_witness_lease_clock.lock().await;
        ensure_commitment_witness_lease_clock(
            &mut clock,
            self.database_path.as_deref(),
            self.database_identity,
        )?;
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin coordinator lease release transaction: {error}"))?;
        let existing = {
            let mut statement = tx
                .prepare(
                    "SELECT coordinator, chain_id, instance_id, lease_epoch,
                            lease_expires_at, updated_at
                     FROM record_coordinator_leases
                     WHERE chain_id=?1 OR coordinator=?2",
                )
                .map_err(|error| format!("prepare coordinator lease release: {error}"))?;
            let rows = statement
                .query_map(
                    params![chain_id.as_slice(), coordinator.as_slice()],
                    |row| {
                        Ok((
                            row.get::<_, Vec<u8>>(0)?,
                            row.get::<_, Vec<u8>>(1)?,
                            row.get::<_, Vec<u8>>(2)?,
                            row.get::<_, i64>(3)?,
                            row.get::<_, i64>(4)?,
                            row.get::<_, i64>(5)?,
                        ))
                    },
                )
                .map_err(|error| format!("read coordinator leases for release: {error}"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("decode coordinator leases for release: {error}"))?
                .into_iter()
                .map(decode_stored_coordinator_lease)
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        initialize_witness_lease_chain_clock(&mut clock, chain_id, &existing, monotonic_now)?;
        let Some(stored) = existing
            .iter()
            .copied()
            .find(|stored| stored.coordinator == *coordinator)
        else {
            return Ok(RecordCoordinatorLeaseReleaseOutcome::NotHolder);
        };
        if stored.chain_id != *chain_id || stored.instance_id != *instance_id {
            return Ok(RecordCoordinatorLeaseReleaseOutcome::NotHolder);
        }
        let lease_epoch = stored.lease_epoch;
        if let Some(hold) = clock.holds.get(chain_id) {
            if monotonic_now < hold.valid_until
                && hold.coordinator == *coordinator
                && hold.instance_id == *instance_id
                && hold.lease_epoch != lease_epoch
            {
                return Err("witness lease clock epoch does not match durable row".to_string());
            }
        }
        let updated = tx
            .execute(
                "UPDATE record_coordinator_leases
                 SET lease_expires_at=?1, updated_at=?1
                 WHERE coordinator=?2 AND chain_id=?3 AND instance_id=?4",
                params![
                    now_i64,
                    coordinator.as_slice(),
                    chain_id.as_slice(),
                    instance_id.as_slice(),
                ],
            )
            .map_err(|error| format!("persist coordinator lease release: {error}"))?;
        if updated != 1 {
            return Err("coordinator lease release did not update exactly one row".to_string());
        }
        let ambiguous_valid_until =
            witness_lease_deadline(monotonic_now, WITNESS_LEASE_RESTART_HOLD_SECS)?;
        if let Err(error) = tx.commit() {
            clock
                .holds
                .entry(*chain_id)
                .and_modify(|hold| {
                    if hold.valid_until < ambiguous_valid_until {
                        hold.valid_until = ambiguous_valid_until;
                    }
                })
                .or_insert(RecordCommitmentWitnessLeaseHold {
                    coordinator: *coordinator,
                    instance_id: *instance_id,
                    lease_epoch,
                    valid_until: ambiguous_valid_until,
                });
            return Err(format!("commit coordinator lease release: {error}"));
        }
        #[cfg(test)]
        if matches!(
            commit_observation,
            WitnessLeaseCommitObservation::UnknownAfterCommit
        ) {
            clock
                .holds
                .entry(*chain_id)
                .and_modify(|hold| {
                    if hold.valid_until < ambiguous_valid_until {
                        hold.valid_until = ambiguous_valid_until;
                    }
                })
                .or_insert(RecordCommitmentWitnessLeaseHold {
                    coordinator: *coordinator,
                    instance_id: *instance_id,
                    lease_epoch,
                    valid_until: ambiguous_valid_until,
                });
            return Err("coordinator lease release commit outcome is unknown".to_string());
        }
        #[cfg(not(test))]
        let _ = commit_observation;
        if existing.iter().any(|other| {
            other.chain_id == *chain_id && other.coordinator != *coordinator && !other.is_released()
        }) {
            clock.holds.insert(
                *chain_id,
                RecordCommitmentWitnessLeaseHold {
                    coordinator: [0; 32],
                    instance_id: [0; 32],
                    lease_epoch,
                    valid_until: ambiguous_valid_until,
                },
            );
        } else {
            clock.holds.remove(chain_id);
        }
        Ok(RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch,
            released_at: now,
        })
    }

    /// Configures whether cross-host witness authority is mandatory.
    pub fn configure_record_commitment_coordinator_lease(
        &self,
        required: bool,
        required_witnesses: usize,
    ) {
        let mut runtime = self.commitment_coordinator_lease.write();
        *runtime = Default::default();
        runtime.required = required;
        runtime.required_witnesses = if required { required_witnesses } else { 0 };
        runtime.state = if required { "acquiring" } else { "disabled" };
    }

    /// Installs one fully verified all-witness lease round.
    pub fn apply_record_commitment_coordinator_lease(
        &self,
        granted_witnesses: usize,
        valid_for_secs: u64,
        now: u64,
    ) -> Result<(), String> {
        let mut runtime = self.commitment_coordinator_lease.write();
        if !runtime.required {
            return Err("coordinator lease enforcement is disabled".to_string());
        }
        if granted_witnesses < runtime.required_witnesses || valid_for_secs == 0 {
            return Err("coordinator lease grant threshold is incomplete".to_string());
        }
        if runtime.consecutive_failures > 0 {
            runtime.recoveries_total = runtime.recoveries_total.saturating_add(1);
        }
        runtime.state = "held";
        runtime.granted_witnesses = granted_witnesses;
        runtime.valid_until = Some(Instant::now() + std::time::Duration::from_secs(valid_for_secs));
        runtime.expires_at = Some(now.saturating_add(valid_for_secs));
        runtime.last_attempted_at = Some(now);
        runtime.last_renewed_at = Some(now);
        runtime.consecutive_failures = 0;
        Ok(())
    }

    /// Records one failed lease round without extending existing authority.
    pub fn record_commitment_coordinator_lease_failure(&self, granted_witnesses: usize) {
        let mut runtime = self.commitment_coordinator_lease.write();
        if !runtime.required {
            return;
        }
        let now = unix_now_secs();
        runtime.granted_witnesses = granted_witnesses;
        runtime.last_attempted_at = Some(now);
        runtime.last_failure_at = Some(now);
        runtime.renewal_failures_total = runtime.renewal_failures_total.saturating_add(1);
        runtime.consecutive_failures = runtime.consecutive_failures.saturating_add(1);
        let still_valid = runtime
            .valid_until
            .is_some_and(|deadline| Instant::now() < deadline);
        runtime.state = if still_valid {
            "renewal_degraded"
        } else if runtime.valid_until.is_some() {
            "expired"
        } else {
            "unavailable"
        };
    }

    /// Returns whether every configured production safety gate is currently valid.
    #[must_use]
    pub fn record_commitment_production_permitted(&self) -> bool {
        if self.record_commitment_production_halted() {
            return false;
        }
        let runtime = self.commitment_coordinator_lease.read();
        !runtime.required
            || runtime
                .valid_until
                .is_some_and(|deadline| Instant::now() < deadline)
    }

    /// Preserves the historical incident error while distinguishing a
    /// recoverable lease outage for callers and existing operational checks.
    pub(super) fn local_record_commitment_production_error(&self) -> Option<&'static str> {
        if self.record_commitment_production_halted() {
            Some("local commitment production halted by trusted witness security incident")
        } else if !self.record_commitment_production_permitted() {
            Some("local commitment production authority is unavailable")
        } else {
            None
        }
    }

    /// Returns the one-way process-local coordinator safety latch.
    #[must_use]
    pub fn record_commitment_production_halted(&self) -> bool {
        self.commitment_production_halted.load(Ordering::Acquire)
    }

    /// Configures the immutable process-local proposer-authority trust root.
    ///
    /// [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Call this before the
    /// startup chain audit. Once installed, only an identical value may be
    /// configured again; authority rotation must use the dual-signed handover
    /// schedule instead of replacing this trust anchor. `None` preserves
    /// legacy tooling and protocol-disabled deployments before installation.
    ///
    /// # Errors
    ///
    /// Returns an error when a configured root is the all-zero sentinel, or
    /// when a caller attempts to replace or disable an installed root.
    pub fn configure_record_commitment_authority_root(
        &self,
        root: Option<[u8; 32]>,
    ) -> Result<(), String> {
        if root.is_some_and(|value| value.iter().all(|byte| *byte == 0)) {
            return Err("commitment authority root must be non-zero".to_string());
        }
        let mut configured = self.commitment_authority_root.write();
        match (*configured, root) {
            (Some(existing), Some(candidate)) if existing != candidate => {
                return Err("commitment authority root is immutable after installation".to_string());
            }
            (Some(_), None) => {
                return Err(
                    "commitment authority root cannot be disabled after installation".to_string(),
                );
            }
            (None, Some(candidate)) => {
                *configured = Some(candidate);
                drop(configured);
                *self.commitment_integrity.write() = None;
            }
            _ => {}
        }
        Ok(())
    }
}
