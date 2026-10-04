// ============================================
// File: crates/aeronyx-server/src/services/reverse_onion_queue_db.rs
// ============================================
//! Owner-private SQLite boundary for the reverse-onion public queue.
//!
//! [REVERSE-ONION-QUEUE-DB 2026-10-04 by Codex] The queue adapter owns
//! schema, row, replay, and transaction policy. This module owns the
//! production database boundary: descriptor identity, process fencing,
//! SQLite durability, page/sidecar bounds, and post-commit ambiguity.
//!
//! [REVERSE-ONION-QUEUE-DB-IDENTITY 2026-10-04 by Codex] The path identity
//! captured after descriptor-relative preparation is rechecked against the
//! retained inode and every pathname observation. This catches observed or
//! accidental replacement without claiming hostile same-euid pathname-race
//! resistance; rusqlite opens by pathname and a trusted VFS/external isolation
//! is still required for that stronger threat model.
//!
//! The module is intentionally not registered by this change. Runtime wiring
//! must explicitly choose this boundary instead of passing a raw Connection
//! to `SqliteReverseOnionQueue`.
//!
//! [REVERSE-ONION-RESULT-CONTEXT-LOOKUP 2026-10-04 by Codex] The wrapper
//! exposes only the queue's authenticated read-only recovery context and
//! retains the durable post-operation fence before publication.
//!
//! [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Source reads additionally
//! fence observed time in memory under the operation lock. This fence is lost
//! on restart; only SQL mutation-clock observations are durable.
//!
//! [REVERSE-ONION-SOURCE-BINDING 2026-10-04 by Codex] Source snapshots are
//! exposed only through the exact source/route/request tuple and never carry
//! the persisted envelope.

use std::path::{Path, PathBuf};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;
use rusqlite::{Connection, OpenFlags};
use thiserror::Error;

use super::chat_relay_mailbox::{
    prepare_private_sqlite_target, verify_private_file, AnonymousMailboxStoreError,
};
use super::reverse_onion_queue::{
    ReverseOnionQueueAdmission, ReverseOnionQueueCompletion, ReverseOnionQueueError,
    ReverseOnionQueueIssue, ReverseOnionQueueItem, ReverseOnionQueueIssuedLease,
    ReverseOnionQueueLimits, ReverseOnionQueueResult, ReverseOnionQueueResultContext,
    ReverseOnionQueueSourceSnapshot,
    ReverseOnionQueueStoredResultContext,
    SqliteReverseOnionQueue,
};

#[cfg(unix)]
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const MAX_QUEUE_ENTRIES: u64 = 1024;
const MAX_QUEUE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_QUEUE_PHYSICAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_QUEUE_CLAIM_BYTES: u64 = 8 * 1024;
const MAX_QUEUE_LEASE_BYTES: u64 = 512 * 1024;
const MAX_QUEUE_RESULT_BYTES: u64 = 512 * 1024;
const SQLITE_MIN_PAGE_SIZE: i64 = 512;
const SQLITE_MAX_PAGE_SIZE: i64 = 65_536;

#[cfg(test)]
thread_local! {
    static FORCE_POST_OPERATION_FENCE_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ReverseOnionQueueDbError {
    #[error("reverse onion queue database rejected")]
    Rejected,
    #[error("reverse onion queue database busy")]
    Busy,
    #[error("reverse onion queue database at capacity")]
    Capacity,
    #[error("reverse onion queue database is corrupt")]
    Corrupt,
    #[error("reverse onion queue database requires migration")]
    MigrationRequired,
    #[error("reverse onion queue database is ambiguous")]
    Ambiguous,
    #[error("reverse onion queue database unavailable")]
    Unavailable,
    #[error("reverse onion queue database has no work")]
    NoWork,
    #[error("reverse onion queue database has a lease conflict")]
    Conflict,
    #[error("reverse onion queue database lease was lost")]
    LeaseLost,
    #[error("reverse onion queue database result is already complete")]
    AlreadyComplete,
}

impl From<ReverseOnionQueueError> for ReverseOnionQueueDbError {
    fn from(error: ReverseOnionQueueError) -> Self {
        match error {
            ReverseOnionQueueError::Rejected => Self::Rejected,
            ReverseOnionQueueError::Conflict => Self::Conflict,
            ReverseOnionQueueError::Capacity => Self::Capacity,
            ReverseOnionQueueError::NoWork => Self::NoWork,
            ReverseOnionQueueError::Ambiguous => Self::Ambiguous,
            ReverseOnionQueueError::LeaseLost => Self::LeaseLost,
            ReverseOnionQueueError::AlreadyComplete => Self::AlreadyComplete,
            ReverseOnionQueueError::Unavailable => Self::Unavailable,
            ReverseOnionQueueError::Corrupt => Self::Corrupt,
            ReverseOnionQueueError::MigrationRequired => Self::MigrationRequired,
        }
    }
}

impl From<AnonymousMailboxStoreError> for ReverseOnionQueueDbError {
    fn from(error: AnonymousMailboxStoreError) -> Self {
        match error {
            AnonymousMailboxStoreError::Disabled => Self::Rejected,
            AnonymousMailboxStoreError::Busy => Self::Busy,
            AnonymousMailboxStoreError::Rejected => Self::Rejected,
            AnonymousMailboxStoreError::UnsupportedSchema => Self::Corrupt,
            AnonymousMailboxStoreError::Corrupt => Self::Corrupt,
            AnonymousMailboxStoreError::Unavailable => Self::Unavailable,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReverseOnionQueueDbConfig {
    pub(crate) db_path: PathBuf,
    pub(crate) physical_bytes: u64,
    pub(crate) limits: ReverseOnionQueueLimits,
}

impl ReverseOnionQueueDbConfig {
    pub(crate) fn new(
        db_path: PathBuf,
        physical_bytes: u64,
        limits: ReverseOnionQueueLimits,
    ) -> Result<Self, ReverseOnionQueueDbError> {
        validate_limits(physical_bytes, limits)?;
        if db_path.as_os_str().is_empty() || db_path == Path::new(":memory:") {
            return Err(ReverseOnionQueueDbError::Rejected);
        }
        Ok(Self {
            db_path,
            physical_bytes,
            limits,
        })
    }
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InodeIdentity {
    device: u64,
    inode: u64,
}

#[cfg(unix)]
impl InodeIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }
}

/// Production-owned reverse-onion queue database.
///
/// The outer mutex is deliberately separate from the queue's connection
/// mutex. It serializes poison check, delegated queue work, all post-commit
/// fences, and result publication without ever holding the connection mutex
/// while calling a queue method.
pub(crate) struct ReverseOnionQueueDb {
    queue: SqliteReverseOnionQueue,
    connection: Mutex<Connection>,
    operation: Mutex<()>,
    // [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Accessed only while
    // operation is held. Read observations are process-local, not durable;
    // reopening restores only the SQL mutation-clock floor.
    source_read_high_water: AtomicU64,
    poisoned: AtomicBool,
    db_path: PathBuf,
    physical_bytes: u64,
    #[cfg(unix)]
    inode_identity: InodeIdentity,
    #[cfg(unix)]
    parent_identity: InodeIdentity,
    #[cfg(unix)]
    _inode_lock: File,
    #[cfg(unix)]
    _parent: File,
}

// [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Only source lookup may
// omit the full database integrity scan; all pre-existing callers retain it.
#[derive(Clone, Copy)]
enum OperationFence {
    FullIntegrity,
    BoundedSourceRead,
}

impl std::fmt::Debug for ReverseOnionQueueDb {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReverseOnionQueueDb")
            .field("poisoned", &self.poisoned.load(Ordering::Acquire))
            .field("physical_bytes", &self.physical_bytes)
            .finish_non_exhaustive()
    }
}

impl ReverseOnionQueueDb {
    #[cfg(unix)]
    pub(crate) fn open(
        config: ReverseOnionQueueDbConfig,
        now: u64,
    ) -> Result<Self, ReverseOnionQueueDbError> {
        if now == 0 {
            return Err(ReverseOnionQueueDbError::Rejected);
        }
        validate_limits(config.physical_bytes, config.limits)?;

        preflight_existing_boundary(&config.db_path, config.physical_bytes)?;
        let target = prepare_private_sqlite_target(&config.db_path)?;
        verify_private_file(&target.resolved_path, true)?;
        let baseline = std::fs::symlink_metadata(&target.resolved_path)
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        validate_opened_inode(&baseline, config.physical_bytes)?;
        let baseline_identity = InodeIdentity::from_metadata(&baseline);
        audit_sidecars(&target.resolved_path, config.physical_bytes, baseline.len())?;

        let inode_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK)
            .open(&target.resolved_path)
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        let inode_metadata = inode_lock
            .metadata()
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        validate_opened_inode(&inode_metadata, config.physical_bytes)?;
        if InodeIdentity::from_metadata(&inode_metadata) != baseline_identity {
            return Err(ReverseOnionQueueDbError::Rejected);
        }

        // SAFETY: the descriptor is a live owner-private regular-file handle.
        if unsafe { nix::libc::flock(inode_lock.as_raw_fd(), nix::libc::LOCK_EX | nix::libc::LOCK_NB) }
            != 0
        {
            return Err(ReverseOnionQueueDbError::Busy);
        }

        let parent_metadata = target
            .parent
            .metadata()
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        let parent_identity = InodeIdentity::from_metadata(&parent_metadata);
        validate_parent(&parent_metadata)?;

        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        let after = std::fs::symlink_metadata(&target.resolved_path)
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        validate_path_identity(&after, baseline_identity, config.physical_bytes)?;
        verify_private_file(&target.resolved_path, true)?;

        configure_sqlite(&mut connection)?;
        let max_page_count = configure_page_cap(&connection, config.physical_bytes)?;
        audit_sqlite(&connection, config.physical_bytes, max_page_count)?;

        let db = Self {
            queue: SqliteReverseOnionQueue::new(config.limits),
            connection: Mutex::new(connection),
            operation: Mutex::new(()),
            source_read_high_water: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
            db_path: target.resolved_path,
            physical_bytes: config.physical_bytes,
            inode_identity: baseline_identity,
            parent_identity,
            _inode_lock: inode_lock,
            _parent: target.parent,
        };
        db.with_operation(true, |queue, connection| {
            queue
                .initialize_at(connection, now)
                .map_err(ReverseOnionQueueDbError::from)
        })?;
        // Initialization validates schema ownership; cleanup additionally runs
        // the queue's bounded row/no-work audit before production activation.
        db.cleanup(now)?;
        Ok(db)
    }

    #[cfg(not(unix))]
    pub(crate) fn open(
        _config: ReverseOnionQueueDbConfig,
        _now: u64,
    ) -> Result<Self, ReverseOnionQueueDbError> {
        Err(ReverseOnionQueueDbError::Rejected)
    }

    pub(crate) fn enqueue(
        &self,
        item: &ReverseOnionQueueItem,
        now: u64,
    ) -> Result<ReverseOnionQueueAdmission, ReverseOnionQueueDbError> {
        self.with_operation(true, |queue, connection| {
            queue
                .enqueue(connection, item, now)
                .map_err(ReverseOnionQueueDbError::from)
        })
    }

    pub(crate) fn issue_lease<V, F>(
        &self,
        recipient: [u8; 32],
        claim_id: [u8; 16],
        claim_commitment: [u8; 32],
        claim_frame: Vec<u8>,
        now: u64,
        verify_claim: V,
        build_lease: F,
    ) -> Result<ReverseOnionQueueIssue, ReverseOnionQueueDbError>
    where
        V: Fn(&[u8]) -> Result<[u8; 32], ReverseOnionQueueError>,
        F: FnOnce(
            &super::reverse_onion_queue::ReverseOnionQueueStoredItem,
            &[u8],
        ) -> Result<super::reverse_onion_queue::ReverseOnionQueueLeaseMaterial, ReverseOnionQueueError>,
    {
        self.with_operation(true, |queue, connection| {
            queue
                .issue_lease(
                    connection,
                    recipient,
                    claim_id,
                    claim_commitment,
                    claim_frame,
                    now,
                    verify_claim,
                    build_lease,
                )
                .map_err(ReverseOnionQueueDbError::from)
        })
    }

    pub(crate) fn lookup_armed(
        &self,
        queue_key: [u8; 32],
        envelope_commitment: [u8; 32],
        recipient: [u8; 32],
        now: u64,
    ) -> Result<Option<ReverseOnionQueueIssuedLease>, ReverseOnionQueueDbError> {
        self.with_operation(true, |queue, connection| {
            queue
                .lookup_armed(connection, queue_key, envelope_commitment, recipient, now)
                .map_err(ReverseOnionQueueDbError::from)
        })
    }

    pub(crate) fn lookup_result(
        &self,
        queue_key: [u8; 32],
        envelope_commitment: [u8; 32],
        recipient: [u8; 32],
        now: u64,
    ) -> Result<Option<ReverseOnionQueueResult>, ReverseOnionQueueDbError> {
        self.with_operation(true, |queue, connection| {
            queue
                .lookup_result(connection, queue_key, envelope_commitment, recipient, now)
                .map_err(ReverseOnionQueueDbError::from)
        })
    }

    pub(crate) fn lookup_result_context(
        &self,
        recipient: [u8; 32],
        claim_id: [u8; 16],
        lease_id: [u8; 16],
        route_id: [u8; 16],
        now: u64,
    ) -> Result<
        ReverseOnionQueueResultContext,
        ReverseOnionQueueDbError,
    > {
        self.with_operation(true, |queue, connection| {
            queue
                .lookup_result_context(
                    connection,
                    recipient,
                    claim_id,
                    lease_id,
                    route_id,
                    now,
                )
                .map_err(ReverseOnionQueueDbError::from)
        })
    }

    pub(crate) fn lookup_source(
        &self,
        source_node_id: [u8; 32],
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        now: u64,
    ) -> Result<Option<ReverseOnionQueueSourceSnapshot>, ReverseOnionQueueDbError> {
        self.with_operation_fence(true, OperationFence::BoundedSourceRead, |queue, connection| {
            // Reject invalid times before touching the memory fence. Advancing
            // before the post-operation fence is conservative: fence failure
            // poisons this wrapper, so no subsequent read can observe success.
            if now == 0 || i64::try_from(now).is_err()
                || now < self.source_read_high_water.load(Ordering::Relaxed)
            {
                return Err(ReverseOnionQueueDbError::Rejected);
            }
            let snapshot = queue
                .lookup_source(connection, source_node_id, route_id, request_commitment, now)
                .map_err(ReverseOnionQueueDbError::from)?;
            self.source_read_high_water.store(now, Ordering::Relaxed);
            Ok(snapshot)
        })
    }

    pub(crate) fn complete(
        &self,
        lease: &ReverseOnionQueueIssuedLease,
        result_frame: &[u8],
        now: u64,
        verify_result: impl FnOnce(
            &ReverseOnionQueueStoredResultContext,
            &[u8],
        ) -> Result<[u8; 32], ReverseOnionQueueError>,
    ) -> Result<ReverseOnionQueueCompletion, ReverseOnionQueueDbError> {
        self.with_operation(true, |queue, connection| {
            queue
                .complete(connection, lease, result_frame, now, verify_result)
                .map_err(ReverseOnionQueueDbError::from)
        })
    }

    pub(crate) fn cleanup(
        &self,
        now: u64,
    ) -> Result<u64, ReverseOnionQueueDbError> {
        self.with_operation(true, |queue, connection| {
            queue
                .cleanup(connection, now)
                .map_err(ReverseOnionQueueDbError::from)
        })
    }

    fn with_operation<T>(
        &self,
        durable: bool,
        action: impl FnOnce(
            &SqliteReverseOnionQueue,
            &Mutex<Connection>,
        ) -> Result<T, ReverseOnionQueueDbError>,
    ) -> Result<T, ReverseOnionQueueDbError> {
        self.with_operation_fence(durable, OperationFence::FullIntegrity, action)
    }

    fn with_operation_fence<T>(
        &self,
        durable: bool,
        fence: OperationFence,
        action: impl FnOnce(
            &SqliteReverseOnionQueue,
            &Mutex<Connection>,
        ) -> Result<T, ReverseOnionQueueDbError>,
    ) -> Result<T, ReverseOnionQueueDbError> {
        let _operation = self.operation.lock();
        if self.poisoned.load(Ordering::Acquire) {
            return Err(ReverseOnionQueueDbError::Unavailable);
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| action(&self.queue, &self.connection)));
        let result = match outcome {
            Ok(Ok(value)) => {
                if let Err(error) = self.post_operation_fence(fence) {
                    self.poisoned.store(true, Ordering::Release);
                    return Err(if durable {
                        ReverseOnionQueueDbError::Ambiguous
                    } else {
                        error
                    });
                }
                Ok(value)
            }
            Ok(Err(error)) => {
                // [REVERSE-ONION-QUEUE-DURABLE-ERROR-FENCE 2026-10-04 by Codex]
                // Expiry errors are committed clock observations from the
                // queue transaction; fence them before publishing the error.
                if durable
                    && matches!(
                        error,
                        ReverseOnionQueueDbError::NoWork | ReverseOnionQueueDbError::Ambiguous
                    )
                {
                    if self.post_operation_fence(fence).is_err() {
                        self.poisoned.store(true, Ordering::Release);
                        return Err(ReverseOnionQueueDbError::Ambiguous);
                    }
                }
                if matches!(
                    error,
                    ReverseOnionQueueDbError::Corrupt
                        | ReverseOnionQueueDbError::Unavailable
                        | ReverseOnionQueueDbError::Ambiguous
                ) {
                    self.poisoned.store(true, Ordering::Release);
                }
                Err(error)
            }
            Err(_) => {
                self.poisoned.store(true, Ordering::Release);
                Err(ReverseOnionQueueDbError::Unavailable)
            }
        };
        result
    }

    #[cfg(unix)]
    fn post_operation_fence(&self, fence: OperationFence) -> Result<(), ReverseOnionQueueDbError> {
        #[cfg(test)]
        if FORCE_POST_OPERATION_FENCE_FAILURE.with(std::cell::Cell::get) {
            return Err(ReverseOnionQueueDbError::Unavailable);
        }
        let metadata = self
            ._inode_lock
            .metadata()
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        validate_opened_inode(&metadata, self.physical_bytes)?;
        if InodeIdentity::from_metadata(&metadata) != self.inode_identity {
            return Err(ReverseOnionQueueDbError::Rejected);
        }
        let parent_metadata = self
            ._parent
            .metadata()
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        validate_parent(&parent_metadata)?;
        if InodeIdentity::from_metadata(&parent_metadata) != self.parent_identity {
            return Err(ReverseOnionQueueDbError::Rejected);
        }
        let path_metadata = std::fs::symlink_metadata(&self.db_path)
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
        validate_path_identity(&path_metadata, self.inode_identity, self.physical_bytes)?;
        audit_sidecars(&self.db_path, self.physical_bytes, path_metadata.len())?;
        let connection = self.connection.lock();
        let max_page_count = configured_max_page_count(&connection)?;
        match fence {
            OperationFence::FullIntegrity => audit_sqlite(&connection, self.physical_bytes, max_page_count)?,
            OperationFence::BoundedSourceRead => audit_sqlite_pages(&connection, self.physical_bytes, max_page_count)?,
        }
        drop(connection);
        self._parent
            .sync_all()
            .map_err(|_| ReverseOnionQueueDbError::Unavailable)
    }

    #[cfg(not(unix))]
    fn post_operation_fence(&self, _fence: OperationFence) -> Result<(), ReverseOnionQueueDbError> {
        Err(ReverseOnionQueueDbError::Rejected)
    }
}

fn validate_limits(
    physical_bytes: u64,
    limits: ReverseOnionQueueLimits,
) -> Result<(), ReverseOnionQueueDbError> {
    if limits.max_items == 0
        || limits.max_items > MAX_QUEUE_ENTRIES
        || limits.max_items_per_recipient == 0
        || limits.max_items_per_recipient > limits.max_items
        || limits.max_bytes == 0
        || limits.max_bytes > MAX_QUEUE_BYTES
        || limits.lease_max_secs == 0
        || limits.recovery_retention_secs == 0
        || limits.route_max_secs == 0
        || physical_bytes == 0
        || physical_bytes > MAX_QUEUE_PHYSICAL_BYTES
    {
        return Err(ReverseOnionQueueDbError::Rejected);
    }
    let logical_minimum = MAX_QUEUE_CLAIM_BYTES
        .checked_add(MAX_QUEUE_LEASE_BYTES)
        .and_then(|value| value.checked_add(MAX_QUEUE_RESULT_BYTES))
        .ok_or(ReverseOnionQueueDbError::Rejected)?;
    if limits.max_bytes < logical_minimum {
        return Err(ReverseOnionQueueDbError::Rejected);
    }
    let derived = limits
        .max_bytes
        .checked_mul(2)
        .and_then(|value| {
            value.checked_add((limits.max_items + 256).checked_mul(4096)?)
        })
        .ok_or(ReverseOnionQueueDbError::Rejected)?;
    if physical_bytes < derived {
        return Err(ReverseOnionQueueDbError::Rejected);
    }
    Ok(())
}

#[cfg(unix)]
fn validate_parent(metadata: &std::fs::Metadata) -> Result<(), ReverseOnionQueueDbError> {
    if !metadata.is_dir() || metadata.uid() != effective_user_id() || metadata.mode() & 0o077 != 0
    {
        return Err(ReverseOnionQueueDbError::Rejected);
    }
    Ok(())
}

#[cfg(unix)]
fn validate_opened_inode(
    metadata: &std::fs::Metadata,
    physical_bytes: u64,
) -> Result<(), ReverseOnionQueueDbError> {
    if !metadata.is_file()
        || metadata.uid() != effective_user_id()
        || metadata.nlink() != 1
        || metadata.mode() & 0o777 != 0o600
    {
        return Err(ReverseOnionQueueDbError::Rejected);
    }
    let length = metadata.len();
    if length > physical_bytes {
        return Err(ReverseOnionQueueDbError::Capacity);
    }
    Ok(())
}

#[cfg(unix)]
fn preflight_existing_boundary(
    path: &Path,
    physical_bytes: u64,
) -> Result<(), ReverseOnionQueueDbError> {
    let parent_path = path.parent().unwrap_or_else(|| Path::new("."));
    let parent = match std::fs::symlink_metadata(parent_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(ReverseOnionQueueDbError::Unavailable),
    };
    if !parent.is_dir()
        || parent.file_type().is_symlink()
        || parent.uid() != effective_user_id()
        || parent.mode() & 0o077 != 0
    {
        return Err(ReverseOnionQueueDbError::Rejected);
    }

    let primary_bytes = match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            // This is read-only. Do not let the later helper chmod or replace a
            // known unsafe candidate before its aggregate budget is checked.
            validate_opened_inode(&metadata, physical_bytes)?;
            metadata.len()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(_) => return Err(ReverseOnionQueueDbError::Unavailable),
    };
    audit_sidecars(path, physical_bytes, primary_bytes)
}

#[cfg(unix)]
fn validate_path_identity(
    metadata: &std::fs::Metadata,
    expected: InodeIdentity,
    physical_bytes: u64,
) -> Result<(), ReverseOnionQueueDbError> {
    if InodeIdentity::from_metadata(metadata) != expected {
        return Err(ReverseOnionQueueDbError::Rejected);
    }
    validate_opened_inode(metadata, physical_bytes)
}

#[cfg(unix)]
fn configure_sqlite(connection: &mut Connection) -> Result<(), ReverseOnionQueueDbError> {
    connection
        .execute_batch(
            "PRAGMA busy_timeout=0;
             PRAGMA trusted_schema=OFF;
             PRAGMA temp_store=MEMORY;
             PRAGMA locking_mode=EXCLUSIVE;
             PRAGMA journal_mode=DELETE;
             PRAGMA synchronous=EXTRA;
             PRAGMA fullfsync=ON;
             PRAGMA foreign_keys=ON;",
        )
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let busy: i64 = connection
        .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let trusted: i64 = connection
        .query_row("PRAGMA trusted_schema", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let temp_store: i64 = connection
        .query_row("PRAGMA temp_store", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let locking: String = connection
        .query_row("PRAGMA locking_mode", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let journal: String = connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let synchronous: i64 = connection
        .query_row("PRAGMA synchronous", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let fullfsync: i64 = connection
        .query_row("PRAGMA fullfsync", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    let foreign_keys: i64 = connection
        .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    if busy != 0
        || trusted != 0
        || temp_store != 2
        || !locking.eq_ignore_ascii_case("exclusive")
        || !journal.eq_ignore_ascii_case("delete")
        || synchronous != 3
        || fullfsync != 1
        || foreign_keys != 1
    {
        return Err(ReverseOnionQueueDbError::Unavailable);
    }
    Ok(())
}

#[cfg(unix)]
fn configure_page_cap(
    connection: &Connection,
    physical_bytes: u64,
) -> Result<i64, ReverseOnionQueueDbError> {
    let page_size: i64 = connection
        .query_row("PRAGMA page_size", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    validate_page_size(page_size)?;
    let max_pages = physical_bytes
        .checked_div(u64::try_from(page_size).map_err(|_| ReverseOnionQueueDbError::Corrupt)?)
        .ok_or(ReverseOnionQueueDbError::Rejected)?;
    let max_pages = i64::try_from(max_pages).map_err(|_| ReverseOnionQueueDbError::Rejected)?;
    if max_pages == 0 {
        return Err(ReverseOnionQueueDbError::Rejected);
    }
    connection
        .pragma_update(None, "max_page_count", max_pages)
        .map_err(|_| ReverseOnionQueueDbError::Capacity)?;
    let installed: i64 = connection
        .query_row("PRAGMA max_page_count", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    if installed <= 0 || installed > max_pages {
        return Err(ReverseOnionQueueDbError::Capacity);
    }
    Ok(installed)
}

#[cfg(unix)]
fn configured_max_page_count(connection: &Connection) -> Result<i64, ReverseOnionQueueDbError> {
    connection
        .query_row("PRAGMA max_page_count", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)
}

#[cfg(unix)]
fn audit_sqlite(
    connection: &Connection,
    physical_bytes: u64,
    max_page_count: i64,
) -> Result<(), ReverseOnionQueueDbError> {
    audit_sqlite_pages(connection, physical_bytes, max_page_count)?;
    // quick_check(1) bounds reported errors, NOT pages scanned. Keep it at
    // open/maintenance and existing operations, never the source-read fence.
    #[cfg(test)]
    FULL_INTEGRITY_AUDITS.with(|count| count.set(count.get() + 1));
    let integrity: String = connection
        .query_row("PRAGMA quick_check(1)", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Corrupt)?;
    if integrity != "ok" {
        return Err(ReverseOnionQueueDbError::Corrupt);
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    static FULL_INTEGRITY_AUDITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Header/page-budget reads
// do not scan unrelated B-trees. Source reads lose per-read unrelated-page
// auditing; startup, maintenance and mutation integrity checks remain intact.
#[cfg(unix)]
fn audit_sqlite_pages(
    connection: &Connection,
    physical_bytes: u64,
    max_page_count: i64,
) -> Result<(), ReverseOnionQueueDbError> {
    let page_size: i64 = connection
        .query_row("PRAGMA page_size", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    validate_page_size(page_size)?;
    let page_count: i64 = connection
        .query_row("PRAGMA page_count", [], |row| row.get(0))
        .map_err(|_| ReverseOnionQueueDbError::Unavailable)?;
    if page_count < 0 || max_page_count <= 0 || page_count > max_page_count {
        return Err(ReverseOnionQueueDbError::Capacity);
    }
    let bytes = u64::try_from(page_count)
        .ok()
        .and_then(|count| count.checked_mul(u64::try_from(page_size).ok()?))
        .ok_or(ReverseOnionQueueDbError::Capacity)?;
    if bytes > physical_bytes {
        return Err(ReverseOnionQueueDbError::Capacity);
    }
    Ok(())
}

#[cfg(unix)]
fn validate_page_size(page_size: i64) -> Result<(), ReverseOnionQueueDbError> {
    if !(SQLITE_MIN_PAGE_SIZE..=SQLITE_MAX_PAGE_SIZE).contains(&page_size)
        || !(page_size as u64).is_power_of_two()
    {
        return Err(ReverseOnionQueueDbError::Corrupt);
    }
    Ok(())
}

#[cfg(unix)]
fn audit_sidecars(
    path: &Path,
    physical_bytes: u64,
    primary_bytes: u64,
) -> Result<(), ReverseOnionQueueDbError> {
    let mut aggregate = primary_bytes;
    for suffix in ["-journal", "-wal", "-shm"] {
        let sidecar = sidecar_path(path, suffix);
        match std::fs::symlink_metadata(&sidecar) {
            Ok(metadata) => {
                if suffix != "-journal" {
                    return Err(ReverseOnionQueueDbError::Corrupt);
                }
                verify_private_file(&sidecar, true)?;
                aggregate = aggregate
                    .checked_add(metadata.len())
                    .ok_or(ReverseOnionQueueDbError::Capacity)?;
                if aggregate > physical_bytes {
                    return Err(ReverseOnionQueueDbError::Capacity);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(ReverseOnionQueueDbError::Unavailable),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

#[cfg(unix)]
fn effective_user_id() -> u32 {
    // SAFETY: geteuid has no preconditions and does not dereference memory.
    unsafe { nix::libc::geteuid() }
}

#[cfg(unix)]
use std::os::fd::AsRawFd;

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use rusqlite::Connection;
    use tempfile::TempDir;

    const NOW: u64 = 1_800_000_000;

    fn limits() -> ReverseOnionQueueLimits {
        ReverseOnionQueueLimits::new(4, 2 * 1024 * 1024, 2, 60, 120)
            .expect("valid queue limits")
    }

    fn fixture() -> (TempDir, ReverseOnionQueueDbConfig) {
        let directory = tempfile::Builder::new()
            .prefix("r6-reverse-onion-queue-db-")
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp")
            .expect("fixture directory");
        let config = ReverseOnionQueueDbConfig::new(
            directory.path().join("queue.sqlite"),
            16 * 1024 * 1024,
            limits(),
        )
        .expect("valid queue config");
        (directory, config)
    }

    fn queue_item() -> ReverseOnionQueueItem {
        let source_node_id = IdentityKeyPair::from_bytes(&[0x71; 32])
            .expect("valid source identity")
            .public_key_bytes();
        ReverseOnionQueueItem::new(
            [1; 32],
            [2; 16],
            [3; 32],
            source_node_id,
            [4; 32],
            [5; 32],
            [6; 32],
            vec![7; 32],
            NOW + 60,
        )
        .expect("valid opaque queue item")
    }

    // [REVERSE-ONION-SOURCE-INDEX 2026-10-04 by Codex] Authored, unexecuted.
    #[cfg(unix)]
    #[test]
    fn source_read_clock_is_process_local_invalid_times_do_not_advance() {
        let (_directory, config) = fixture();
        let source = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap().public_key_bytes();
        let db = ReverseOnionQueueDb::open(config.clone(), NOW).unwrap();
        db.enqueue(&queue_item(), NOW).unwrap();
        for invalid in [0, u64::MAX] {
            assert!(matches!(db.lookup_source(source, [2; 16], [3; 32], invalid),
                Err(ReverseOnionQueueDbError::Rejected)));
        }
        assert_eq!(db.source_read_high_water.load(Ordering::Relaxed), 0);
        assert!(db.lookup_source(source, [2; 16], [3; 32], NOW + 70).unwrap().is_none());
        assert!(matches!(db.lookup_source(source, [2; 16], [3; 32], NOW + 50),
            Err(ReverseOnionQueueDbError::Rejected)));
        drop(db);
        // No read timestamp was written to SQL: a restart at an earlier time
        // above the durable mutation floor can expose the still-retained row.
        let reopened = ReverseOnionQueueDb::open(config, NOW + 50).unwrap();
        assert!(reopened.lookup_source(source, [2; 16], [3; 32], NOW + 50).unwrap().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn source_read_fence_failure_poisons_before_any_later_publication() {
        let (_directory, config) = fixture();
        let db = ReverseOnionQueueDb::open(config, NOW).unwrap();
        let source = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap().public_key_bytes();
        force_fence(true);
        let outcome = db.lookup_source(source, [2; 16], [3; 32], NOW + 1);
        force_fence(false);
        assert!(matches!(outcome, Err(ReverseOnionQueueDbError::Ambiguous)));
        assert!(matches!(db.lookup_source(source, [2; 16], [3; 32], NOW + 2),
            Err(ReverseOnionQueueDbError::Unavailable)));
    }

    #[cfg(unix)]
    #[test]
    fn source_read_fence_skips_full_integrity_but_open_mutation_cleanup_retain_it() {
        let (_directory, config) = fixture();
        FULL_INTEGRITY_AUDITS.with(|count| count.set(0));
        let db = ReverseOnionQueueDb::open(config, NOW).unwrap();
        assert!(FULL_INTEGRITY_AUDITS.with(std::cell::Cell::get) > 0);
        FULL_INTEGRITY_AUDITS.with(|count| count.set(0));
        db.enqueue(&queue_item(), NOW).unwrap();
        assert!(FULL_INTEGRITY_AUDITS.with(std::cell::Cell::get) > 0);
        FULL_INTEGRITY_AUDITS.with(|count| count.set(0));
        let source = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap().public_key_bytes();
        assert!(db.lookup_source(source, [2; 16], [3; 32], NOW + 1).unwrap().is_some());
        assert!(db.lookup_source(source, [9; 16], [10; 32], NOW + 1).unwrap().is_none());
        assert_eq!(FULL_INTEGRITY_AUDITS.with(std::cell::Cell::get), 0);
        db.cleanup(NOW + 2).unwrap();
        assert!(FULL_INTEGRITY_AUDITS.with(std::cell::Cell::get) > 0);
    }

    #[cfg(unix)]
    fn private_mode(path: &Path) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .expect("private mode");
    }

    #[cfg(unix)]
    fn force_fence(failed: bool) {
        FORCE_POST_OPERATION_FENCE_FAILURE.with(|value| value.set(failed));
    }

    #[test]
    fn physical_bound_requires_logical_queue_bound() {
        let limits = ReverseOnionQueueLimits::new(
            1,
            (MAX_QUEUE_CLAIM_BYTES + MAX_QUEUE_LEASE_BYTES + MAX_QUEUE_RESULT_BYTES) as u64,
            1,
            1,
            1,
        )
        .unwrap();
        assert!(ReverseOnionQueueDbConfig::new(PathBuf::from(":memory:"), 1, limits).is_err());
        assert!(ReverseOnionQueueDbConfig::new(
            PathBuf::from("queue.sqlite"),
            MAX_QUEUE_PHYSICAL_BYTES,
            limits,
        )
        .is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn competing_open_is_busy_and_restart_releases_inode_fence() {
        let (_directory, config) = fixture();
        let first = ReverseOnionQueueDb::open(config.clone(), NOW).expect("first owner");
        assert_eq!(
            ReverseOnionQueueDb::open(config.clone(), NOW + 1).unwrap_err(),
            ReverseOnionQueueDbError::Busy
        );
        drop(first);
        ReverseOnionQueueDb::open(config, NOW + 2).expect("restart after owner drop");
    }

    #[cfg(unix)]
    #[test]
    fn foreign_schema_is_preserved_and_requires_explicit_migration() {
        let (_directory, config) = fixture();
        let connection = Connection::open(&config.db_path).expect("foreign database");
        connection
            .execute("CREATE TABLE foreign_schema(value BLOB NOT NULL)", [])
            .expect("foreign schema");
        drop(connection);
        private_mode(&config.db_path);

        assert_eq!(
            ReverseOnionQueueDb::open(config.clone(), NOW).unwrap_err(),
            ReverseOnionQueueDbError::MigrationRequired
        );
        let connection = Connection::open(&config.db_path).expect("reopen foreign database");
        let count: i64 = connection
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='foreign_schema'",
                [],
                |row| row.get(0),
            )
            .expect("foreign schema remains");
        assert_eq!(count, 1);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_hardlink_and_permissive_mode_fail_before_sqlite_mutation() {
        use std::os::unix::fs::symlink;

        let (directory, config) = fixture();
        let target = directory.path().join("target.sqlite");
        std::fs::write(&target, []).expect("target");
        private_mode(&target);
        symlink(&target, &config.db_path).expect("symlink fixture");
        assert!(matches!(
            ReverseOnionQueueDb::open(config.clone(), NOW),
            Err(ReverseOnionQueueDbError::Rejected | ReverseOnionQueueDbError::Unavailable)
        ));
        std::fs::remove_file(&config.db_path).expect("remove symlink");

        std::fs::write(&config.db_path, []).expect("hardlink source");
        private_mode(&config.db_path);
        let alias = directory.path().join("alias.sqlite");
        std::fs::hard_link(&config.db_path, &alias).expect("hardlink fixture");
        assert_eq!(
            ReverseOnionQueueDb::open(config.clone(), NOW).unwrap_err(),
            ReverseOnionQueueDbError::Rejected
        );
        std::fs::remove_file(&alias).expect("remove hardlink");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&config.db_path, std::fs::Permissions::from_mode(0o644))
            .expect("permissive mode");
        assert_eq!(
            ReverseOnionQueueDb::open(config, NOW).unwrap_err(),
            ReverseOnionQueueDbError::Rejected
        );
    }

    #[cfg(unix)]
    #[test]
    fn oversized_primary_and_sidecars_are_rejected_without_cleanup() {
        let (directory, config) = fixture();
        std::fs::write(&config.db_path, vec![0u8; 16 * 1024 * 1024 + 1]).expect("large primary");
        private_mode(&config.db_path);
        assert_eq!(
            ReverseOnionQueueDb::open(config.clone(), NOW).unwrap_err(),
            ReverseOnionQueueDbError::Capacity
        );
        assert_eq!(std::fs::metadata(&config.db_path).unwrap().len(), 16 * 1024 * 1024 + 1);

        std::fs::write(&config.db_path, vec![0u8; 12 * 1024 * 1024]).expect("primary budget");
        private_mode(&config.db_path);
        let journal = PathBuf::from(format!("{}-journal", config.db_path.display()));
        std::fs::write(&journal, vec![0u8; 5 * 1024 * 1024]).expect("aggregate journal");
        private_mode(&journal);
        assert_eq!(
            ReverseOnionQueueDb::open(config.clone(), NOW).unwrap_err(),
            ReverseOnionQueueDbError::Capacity
        );
        assert!(journal.exists());

        std::fs::remove_file(&journal).expect("remove journal fixture");
        let wal = PathBuf::from(format!("{}-wal", config.db_path.display()));
        std::fs::write(&wal, [0u8; 1]).expect("wal fixture");
        private_mode(&wal);
        assert_eq!(
            ReverseOnionQueueDb::open(config, NOW).unwrap_err(),
            ReverseOnionQueueDbError::Corrupt
        );
        assert!(directory.path().join("queue.sqlite-wal").exists());
    }

    #[cfg(unix)]
    #[test]
    fn absent_primary_with_oversized_journal_is_rejected_without_creation_or_chmod() {
        use std::os::unix::fs::PermissionsExt;

        let (directory, config) = fixture();
        let journal = PathBuf::from(format!("{}-journal", config.db_path.display()));
        std::fs::write(&journal, vec![0u8; 16 * 1024 * 1024 + 1]).expect("oversized journal");
        private_mode(&journal);
        let parent_mode = std::fs::metadata(directory.path())
            .expect("parent metadata")
            .permissions()
            .mode()
            & 0o777;
        let journal_mode = std::fs::metadata(&journal)
            .expect("journal metadata")
            .permissions()
            .mode()
            & 0o777;

        assert_eq!(
            ReverseOnionQueueDb::open(config.clone(), NOW).unwrap_err(),
            ReverseOnionQueueDbError::Capacity
        );
        assert!(!config.db_path.exists());
        assert_eq!(
            std::fs::metadata(directory.path())
                .expect("parent remains")
                .permissions()
                .mode()
                & 0o777,
            parent_mode
        );
        assert_eq!(
            std::fs::metadata(&journal)
                .expect("journal remains")
                .permissions()
                .mode()
                & 0o777,
            journal_mode
        );
    }

    #[cfg(unix)]
    #[test]
    fn replaced_path_identity_is_detected_without_mutating_replacement() {
        let (directory, config) = fixture();
        std::fs::write(&config.db_path, []).expect("original database");
        private_mode(&config.db_path);
        let original = std::fs::symlink_metadata(&config.db_path).expect("original metadata");
        let expected = InodeIdentity::from_metadata(&original);
        let replacement = directory.path().join("replacement.sqlite");
        std::fs::rename(&config.db_path, &replacement).expect("move original");
        std::fs::write(&config.db_path, []).expect("replacement database");
        private_mode(&config.db_path);
        let observed = std::fs::symlink_metadata(&config.db_path).expect("replacement metadata");
        assert_eq!(
            validate_path_identity(&observed, expected, 16 * 1024 * 1024).unwrap_err(),
            ReverseOnionQueueDbError::Rejected
        );
        assert_eq!(std::fs::metadata(&config.db_path).unwrap().len(), 0);
        assert!(replacement.exists());
    }

    #[cfg(unix)]
    #[test]
    fn pragma_and_page_cap_are_read_back_after_open() {
        let (_directory, config) = fixture();
        let db = ReverseOnionQueueDb::open(config, NOW).expect("queue database");
        let connection = db.connection.lock();
        let busy: i64 = connection
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .expect("busy timeout");
        let synchronous: i64 = connection
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .expect("synchronous");
        let journal: String = connection
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .expect("journal mode");
        let locking: String = connection
            .query_row("PRAGMA locking_mode", [], |row| row.get(0))
            .expect("locking mode");
        let page_size: i64 = connection
            .query_row("PRAGMA page_size", [], |row| row.get(0))
            .expect("page size");
        let page_count: i64 = connection
            .query_row("PRAGMA page_count", [], |row| row.get(0))
            .expect("page count");
        let max_page_count: i64 = connection
            .query_row("PRAGMA max_page_count", [], |row| row.get(0))
            .expect("max page count");
        assert_eq!(busy, 0);
        assert_eq!(synchronous, 3);
        assert_eq!(journal.to_ascii_lowercase(), "delete");
        assert_eq!(locking.to_ascii_lowercase(), "exclusive");
        assert!((SQLITE_MIN_PAGE_SIZE..=SQLITE_MAX_PAGE_SIZE).contains(&page_size));
        assert!(page_count >= 0 && page_count <= max_page_count);
    }

    #[cfg(unix)]
    #[test]
    fn post_fence_failure_poisons_before_admission_or_result_publication() {
        let (_directory, config) = fixture();
        let db = ReverseOnionQueueDb::open(config, NOW).expect("queue database");
        let item = queue_item();
        force_fence(true);
        assert_eq!(
            db.enqueue(&item, NOW).unwrap_err(),
            ReverseOnionQueueDbError::Ambiguous
        );
        force_fence(false);
        assert_eq!(
            db.enqueue(&item, NOW + 1).unwrap_err(),
            ReverseOnionQueueDbError::Unavailable
        );

        let (_directory, config) = fixture();
        let db = ReverseOnionQueueDb::open(config, NOW).expect("queue database");
        force_fence(true);
        assert_eq!(
            db.lookup_result([1; 32], [2; 32], [3; 32], NOW)
                .err(),
            Some(ReverseOnionQueueDbError::Unavailable)
        );
        force_fence(false);
    }

    #[cfg(unix)]
    #[test]
    fn result_context_lookup_is_fenced_and_never_issues_unknown_claim() {
        let (_directory, config) = fixture();
        let db = ReverseOnionQueueDb::open(config, NOW).expect("queue database");
        let outcome = db
            .lookup_result_context([5; 32], [6; 16], [7; 16], [8; 16], NOW + 1)
            .expect("read-only lookup");
        assert!(matches!(
            outcome,
            ReverseOnionQueueResultContext::NoWork
        ));
        force_fence(true);
        assert_eq!(
            db.lookup_result_context([5; 32], [6; 16], [7; 16], [8; 16], NOW + 2)
                .unwrap_err(),
            ReverseOnionQueueDbError::Ambiguous
        );
        force_fence(false);
    }
}
