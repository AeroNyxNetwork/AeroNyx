// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_mailbox.rs
// ============================================
//! # Node-local anonymous mailbox custody
//!
//! ## Creation reason
//! M13B adds the durable repository half of anonymous mailbox custody without
//! wiring any API, peer route, discovery advertisement, or client behavior.
//!
//! ## Security boundary
//! The repository stores only unlinkable mailbox capabilities and opaque
//! sealed envelopes. It never parses chat bytes and has no sender, receiver,
//! wallet, route, or endpoint column. All mutating quota decisions share an
//! immediate SQLite transaction; process-local admission is separately
//! bounded. Cursor authentication uses a stable node-local secret supplied by
//! the future composition root.
//!
//! ## Last modified
//! v1.3.1-PreWriteTicketRejection — Distinguish no-effect target/PoW rejection
//! from ambiguous repository errors for signed terminal completion.
//! v1.3.0-PullItemExpiry — Bind durable Item replay to the signed item's
//! expiry; migrate V3 without inventing expiry for already-ACKed ciphertext.
//! v1.2.0-PullReplayJournal — Persist bounded exact PullOne results so a lost
//! response remains byte-stable across ACK and process restart.
//! v1.1.2-OwnerStorageRestartGuard — Cover capacity release through
//! receiver-bound ACK and bounded expiry cleanup across durable reopen.
//! v1.1.1-LeaseReplayContract — Document and test durable exact replay before
//! admission-ticket freshness.
//! v1.1.0-AnonymousMailboxTicketIssuer — Added durable, target-identity
//! signed, rate-bounded anonymous admission-ticket issuance.
//! v1.0.0-AnonymousMailboxStore — Initial node-local custody repository.

use std::collections::HashMap;
use std::fmt;
#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    AnonymousMailboxAckV1, AnonymousMailboxAdmissionTicketV1, AnonymousMailboxLeaseCreateV1,
    AnonymousMailboxPullOneV1, AnonymousMailboxPutV1, AnonymousMailboxTicketIssueV1,
    MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE, MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES,
};
use hmac::{Hmac, Mac};
#[cfg(unix)]
use nix::fcntl::{openat, OFlag};
#[cfg(unix)]
use nix::sys::stat::{mkdirat, Mode};
use parking_lot::Mutex;
use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use sha2::{Digest, Sha256};

use crate::config_chat_relay::AnonymousMailboxStoreConfig;

use super::chat_relay_backup_certification::verify_sqlite_physical_integrity;
use super::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions,
};

const SCHEMA_VERSION: i64 = 4;
const MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
const CURSOR_VERSION: u8 = 1;
const CURSOR_BODY_BYTES: usize = 1 + 8 + 8 + 8;
const CURSOR_TAG_BYTES: usize = 32;
const CURSOR_BYTES: usize = CURSOR_BODY_BYTES + CURSOR_TAG_BYTES;
// [ANONYMOUS-MAILBOX-PULL-BOUNDS 2026-09-24 by Codex] The V4 SQLite CHECK
// literals in fresh and migration DDL are frozen to these protocol ceilings.
const _: () = assert!(MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES == 162_816 && CURSOR_BYTES == 57);
const CURSOR_TTL_SECS: u64 = 5 * 60;
const PULL_REPLAY_RETENTION_SECS: u64 = 24 * 60 * 60;
const ACK_TOMBSTONE_RETENTION_SECS: u64 = 24 * 60 * 60;
const CURSOR_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/store-cursor/v1\0";

type CursorMac = Hmac<Sha256>;

#[cfg(test)]
thread_local! {
    static FULL_AUDIT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PARENT_DURABILITY_SYNCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FORCE_PARENT_SYNC_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Coarse repository failures. Deliberately contains no path, key, opaque id,
/// commitment, or underlying SQLite text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AnonymousMailboxStoreError {
    /// Feature construction was attempted while disabled.
    #[error("anonymous mailbox store disabled")]
    Disabled,
    /// The process-local operation window is full.
    #[error("anonymous mailbox store busy")]
    Busy,
    /// A signature, freshness window, capability, cursor, or request shape was rejected.
    #[error("anonymous mailbox request rejected")]
    Rejected,
    /// The database belongs to an unknown schema revision.
    #[error("anonymous mailbox schema unsupported")]
    UnsupportedSchema,
    /// Durable bytes or accounting violate the frozen repository invariant.
    #[error("anonymous mailbox store corrupt")]
    Corrupt,
    /// Private storage could not be opened, committed, or rolled back safely.
    #[error("anonymous mailbox store unavailable")]
    Unavailable,
}

impl From<rusqlite::Error> for AnonymousMailboxStoreError {
    fn from(_: rusqlite::Error) -> Self {
        Self::Unavailable
    }
}

/// Durable one-time admission projection.
#[derive(Clone, PartialEq, Eq)]
pub struct AnonymousMailboxTicketProjection {
    pub ticket_id: [u8; 16],
    pub claims_commitment: [u8; 32],
    pub mailbox_id: [u8; 32],
    pub consumed_at: u64,
    pub expires_at: u64,
}

impl fmt::Debug for AnonymousMailboxTicketProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxTicketProjection")
            .field("capabilities", &"<redacted>")
            .field("consumed_at", &self.consumed_at)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Durable lease projection. Verifier keys are unlinkable capabilities, not
/// account or wallet identities.
#[derive(Clone, PartialEq, Eq)]
pub struct AnonymousMailboxLeaseProjection {
    pub mailbox_id: [u8; 32],
    pub deposit_verifier: [u8; 32],
    pub read_verifier: [u8; 32],
    pub max_items: u16,
    pub max_bytes: u64,
    pub current_items: u16,
    pub current_bytes: u64,
    pub next_sequence: u64,
    pub created_at: u64,
    pub expires_at: u64,
}

impl fmt::Debug for AnonymousMailboxLeaseProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxLeaseProjection")
            .field("capabilities", &"<redacted>")
            .field("max_items", &self.max_items)
            .field("max_bytes", &self.max_bytes)
            .field("current_items", &self.current_items)
            .field("current_bytes", &self.current_bytes)
            .field("next_sequence", &self.next_sequence)
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Durable immutable sealed-item projection.
#[derive(Clone, PartialEq, Eq)]
pub struct AnonymousMailboxItemProjection {
    pub mailbox_id: [u8; 32],
    pub item_id: [u8; 16],
    pub sequence: u64,
    pub sealed_commitment: [u8; 32],
    pub sealed_envelope: Vec<u8>,
    pub stored_at: u64,
    pub expires_at: u64,
}

impl fmt::Debug for AnonymousMailboxItemProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxItemProjection")
            .field("capabilities", &"<redacted>")
            .field("sequence", &self.sequence)
            .field("sealed_envelope", &"<redacted>")
            .field("sealed_length", &self.sealed_envelope.len())
            .field("stored_at", &self.stored_at)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Store-local projection for one padded pull result.
///
/// This is deliberately not a terminal-wire payload: the M13A terminal field
/// cannot contain a maximum item plus this metadata. M13C must freeze a
/// separately size-proven codec before any route exposes this projection.
#[derive(Clone, PartialEq, Eq)]
pub struct AnonymousMailboxPulledItem {
    pub item_id: [u8; 16],
    pub sealed_commitment: [u8; 32],
    pub sealed_length: u32,
    pub padded_sealed_envelope: Vec<u8>,
    pub cursor: Vec<u8>,
}

impl fmt::Debug for AnonymousMailboxPulledItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxPulledItem")
            .field("capabilities", &"<redacted>")
            .field("sealed_length", &self.sealed_length)
            .field("padded_sealed_envelope", &"<redacted>")
            .field("cursor", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnonymousMailboxCreateOutcome {
    Created(AnonymousMailboxLeaseProjection),
    Existing(AnonymousMailboxLeaseProjection),
    Conflict,
    AtCapacity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnonymousMailboxPutOutcome {
    Stored(AnonymousMailboxItemProjection),
    Existing(AnonymousMailboxItemProjection),
    Conflict,
    AtCapacity,
    LeaseExpired,
    LeaseNotFound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnonymousMailboxPullOutcome {
    Item(AnonymousMailboxPulledItem),
    Empty,
    LeaseExpired,
    LeaseNotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymousMailboxAckOutcome {
    Acknowledged,
    AlreadyAcknowledged,
    Conflict,
    NotFound,
}

/// Coarse durable result of one target-issued admission-ticket request.
#[derive(Clone, PartialEq, Eq)]
pub enum AnonymousMailboxTicketIssueOutcome {
    /// A newly signed ticket was durably issued.
    Issued(AnonymousMailboxAdmissionTicketV1),
    /// The exact request was replayed and returned its original ticket.
    Existing(AnonymousMailboxAdmissionTicketV1),
    /// The request or ticket id was reused for a different canonical request.
    Conflict,
    /// The global ticket count or fixed issue window is full.
    AtCapacity,
    /// [ANONYMOUS-MAILBOX-POLICY-TERMINAL 2026-09-24 by Codex] The exact
    /// replay lookup missed and target/PoW validation rejected before the
    /// first SQL write. This is the only safe signed no-effect rejection.
    PreWriteRejected,
}

impl fmt::Debug for AnonymousMailboxTicketIssueOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Issued(_) => {
                formatter.write_str("AnonymousMailboxTicketIssueOutcome::Issued(<redacted>)")
            }
            Self::Existing(_) => {
                formatter.write_str("AnonymousMailboxTicketIssueOutcome::Existing(<redacted>)")
            }
            Self::Conflict => formatter.write_str("AnonymousMailboxTicketIssueOutcome::Conflict"),
            Self::AtCapacity => {
                formatter.write_str("AnonymousMailboxTicketIssueOutcome::AtCapacity")
            }
            Self::PreWriteRejected => {
                formatter.write_str("AnonymousMailboxTicketIssueOutcome::PreWriteRejected")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnonymousMailboxCleanupReport {
    pub leases_removed: u64,
    pub items_removed: u64,
    pub bytes_removed: u64,
    pub acknowledgements_removed: u64,
    pub tickets_removed: u64,
    pub issued_tickets_removed: u64,
    /// Expired exact PullOne response journals removed in this transaction.
    pub pull_replays_removed: u64,
}

/// Synchronous local capability boundary. Future async callers must place it
/// behind an explicit blocking boundary.
pub trait AnonymousMailboxCustodyRepository: Send + Sync {
    /// Issues one target-signed ticket through the node-local policy boundary.
    fn issue_ticket(
        &self,
        request: &AnonymousMailboxTicketIssueV1,
        now: u64,
    ) -> Result<AnonymousMailboxTicketIssueOutcome, AnonymousMailboxStoreError>;

    /// [M13J 2026-09-05 by Codex] Resolves a durable exact full-request replay
    /// before freshness; a miss must fully authenticate the target, claims,
    /// signature, and time before any lease or ticket mutation.
    fn create(
        &self,
        request: &AnonymousMailboxLeaseCreateV1,
        now: u64,
    ) -> Result<AnonymousMailboxCreateOutcome, AnonymousMailboxStoreError>;

    fn put(
        &self,
        request: &AnonymousMailboxPutV1,
        now: u64,
    ) -> Result<AnonymousMailboxPutOutcome, AnonymousMailboxStoreError>;

    fn pull_one(
        &self,
        request: &AnonymousMailboxPullOneV1,
        now: u64,
    ) -> Result<AnonymousMailboxPullOutcome, AnonymousMailboxStoreError>;

    fn ack(
        &self,
        request: &AnonymousMailboxAckV1,
        now: u64,
    ) -> Result<AnonymousMailboxAckOutcome, AnonymousMailboxStoreError>;

    fn cleanup(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxCleanupReport, AnonymousMailboxStoreError>;
}

/// SQLite composition for one node-local anonymous mailbox repository.
pub struct SqliteAnonymousMailboxStore {
    config: AnonymousMailboxStoreConfig,
    target_node_id: [u8; 32],
    ticket_issuer: Option<IdentityKeyPair>,
    cursor_secret: [u8; 32],
    connection: Mutex<Connection>,
    in_flight: AtomicUsize,
    #[cfg(unix)]
    _database_parent: File,
}

struct OperationPermit<'a> {
    counter: &'a AtomicUsize,
}

impl Drop for OperationPermit<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug, Clone, Copy)]
struct CursorState {
    snapshot_ceiling: u64,
    next_sequence: u64,
    expires_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StoreTotals {
    leases: u64,
    items: u64,
    bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PullReplayTotals {
    rows: u64,
    bytes: u64,
}

// [ANONYMOUS-MAILBOX-PULL-BOUNDS 2026-09-24 by Codex] SQLite length/typeof
// metadata is checked before any durable replay BLOB is materialized.
struct PullReplayBlobShape {
    kind: String,
    length: Option<i64>,
}

impl PullReplayBlobShape {
    fn from_row(row: &rusqlite::Row<'_>, first: usize) -> rusqlite::Result<Self> {
        Ok(Self {
            kind: row.get(first)?,
            length: row.get(first + 1)?,
        })
    }

    fn is_null(&self) -> bool {
        self.kind == "null" && self.length.is_none()
    }

    fn is_exact_blob(&self, expected: usize) -> bool {
        self.kind == "blob" && self.length == i64::try_from(expected).ok()
    }

    fn bounded_blob_length(&self, max: usize) -> Option<u64> {
        let length = u64::try_from(self.length?).ok()?;
        (self.kind == "blob" && length > 0 && length <= u64::try_from(max).ok()?).then_some(length)
    }
}

struct PullReplayShape {
    request: PullReplayBlobShape,
    item_id: PullReplayBlobShape,
    commitment: PullReplayBlobShape,
    envelope: PullReplayBlobShape,
    cursor: PullReplayBlobShape,
}

impl PullReplayShape {
    fn from_row(row: &rusqlite::Row<'_>, first: usize) -> rusqlite::Result<Self> {
        Ok(Self {
            request: PullReplayBlobShape::from_row(row, first)?,
            item_id: PullReplayBlobShape::from_row(row, first + 2)?,
            commitment: PullReplayBlobShape::from_row(row, first + 4)?,
            envelope: PullReplayBlobShape::from_row(row, first + 6)?,
            cursor: PullReplayBlobShape::from_row(row, first + 8)?,
        })
    }

    fn checked_envelope_bytes(&self, outcome: i64) -> Result<u64, AnonymousMailboxStoreError> {
        if !self.request.is_exact_blob(32) {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        match outcome {
            0 if self.item_id.is_null()
                && self.commitment.is_null()
                && self.envelope.is_null()
                && self.cursor.is_null() =>
            {
                Ok(0)
            }
            1 if self.item_id.is_exact_blob(16)
                && self.commitment.is_exact_blob(32)
                && self.cursor.is_exact_blob(CURSOR_BYTES) =>
            {
                self.envelope
                    .bounded_blob_length(MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)
            }
            _ => Err(AnonymousMailboxStoreError::Corrupt),
        }
    }
}

enum StoredPullReplay {
    Empty,
    Item {
        item_id: [u8; 16],
        sealed_commitment: [u8; 32],
        sealed_envelope: Vec<u8>,
        cursor: Vec<u8>,
        item_expires_at: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TicketIssueMeta {
    outstanding: u64,
    window_started_at: u64,
    issues_in_window: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ExpiredIssuedTicketPurge {
    removed: u64,
    unconsumed: u64,
}

#[derive(Clone)]
struct IssuedTicketRecord {
    request_id: [u8; 16],
    request_commitment: [u8; 32],
    request: AnonymousMailboxTicketIssueV1,
    ticket: AnonymousMailboxAdmissionTicketV1,
    ticket_commitment: [u8; 32],
    consumed_at: Option<u64>,
}

impl fmt::Debug for IssuedTicketRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IssuedTicketRecord")
            .field("capabilities", &"<redacted>")
            .field("consumed", &self.consumed_at.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy)]
struct LeaseCounterAudit {
    stored_items: u64,
    stored_bytes: u64,
    next_sequence: u64,
    observed_items: u64,
    observed_bytes: u64,
    observed_max_sequence: u64,
}

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod custody_flow;
mod custody_rows;
mod pull_replay;
mod schema_audit;
mod store_open;
mod ticket_records;

use custody_rows::as_i64;
use custody_rows::as_u64;
use custody_rows::fixed;
use custody_rows::load_ack_commitment;
use custody_rows::load_item;
use custody_rows::load_lease;
use custody_rows::select_expired_items;
use pull_replay::insert_pull_replay;
use pull_replay::load_pull_replay;
use pull_replay::validate_pull_replay_retention;
use schema_audit::audit_counters;
use schema_audit::audit_issued_tickets;
use schema_audit::audit_pull_replays;
#[cfg(test)]
use schema_audit::count;
use schema_audit::execute_exactly_one;
use schema_audit::initialize_or_verify_schema;
use schema_audit::load_pull_replay_totals;
use schema_audit::load_totals;
use schema_audit::update_pull_replay_totals_exact;
use schema_audit::update_totals_exact;
use schema_audit::validate_pull_replay_totals;
use schema_audit::validate_totals_limits;
use schema_audit::verify_totals_exact;
use ticket_records::load_issued_ticket;
use ticket_records::load_issued_ticket_by_request;
use ticket_records::load_issued_ticket_by_ticket;
use ticket_records::load_ticket_issue_meta;
use ticket_records::purge_expired_issued_tickets;
use ticket_records::update_ticket_issue_meta_exact;
use ticket_records::validate_issued_ticket;
use ticket_records::verify_ticket_issue_meta;

impl SqliteAnonymousMailboxStore {
    /// Opens a private durable store. Disabled construction performs no path
    /// or SQLite operation.
    pub fn open(
        config: AnonymousMailboxStoreConfig,
        target_node_id: [u8; 32],
        cursor_secret: [u8; 32],
    ) -> Result<Self, AnonymousMailboxStoreError> {
        Self::open_inner(config, target_node_id, cursor_secret, None)
    }

    /// Opens a custody store that can issue tickets using the local target identity.
    ///
    /// The older [`Self::open`] remains compatible for read/create/put/pull/ack
    /// callers but deliberately cannot mint new target authority.
    pub fn open_with_ticket_issuer(
        config: AnonymousMailboxStoreConfig,
        ticket_issuer: IdentityKeyPair,
        cursor_secret: [u8; 32],
    ) -> Result<Self, AnonymousMailboxStoreError> {
        let target_node_id = ticket_issuer.public_key_bytes();
        Self::open_inner(config, target_node_id, cursor_secret, Some(ticket_issuer))
    }
}

/// Owner-private SQLite target reserved descriptor-relatively before a caller
/// hands its canonical path to SQLite. This crate-private capability is shared
/// only by node-blind stores with the same no-follow/link-count/durability
/// contract; it exposes no custody schema or mailbox data.
pub(crate) struct PrivateSqliteTarget {
    pub(crate) resolved_path: PathBuf,
    #[cfg(unix)]
    pub(crate) parent: File,
}

#[cfg(unix)]
pub(crate) fn prepare_private_sqlite_target(
    path: &Path,
) -> Result<PrivateSqliteTarget, AnonymousMailboxStoreError> {
    let name = path
        .file_name()
        .ok_or(AnonymousMailboxStoreError::Rejected)?;
    let parent_path = path.parent().unwrap_or_else(|| Path::new("."));
    let mut parent = open_directory_anchor(parent_path)?;
    for component in parent_path.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(name) => parent = open_or_create_directory_at(&parent, name)?,
            Component::ParentDir | Component::Prefix(_) => {
                return Err(AnonymousMailboxStoreError::Rejected);
            }
        }
    }
    let parent_metadata = parent
        .metadata()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    if !parent_metadata.is_dir() || parent_metadata.uid() != effective_user_id() {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    parent
        .set_permissions(std::fs::Permissions::from_mode(0o700))
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let parent_metadata = parent
        .metadata()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    if !parent_metadata.is_dir()
        || parent_metadata.uid() != effective_user_id()
        || parent_metadata.permissions().mode() & 0o077 != 0
    {
        return Err(AnonymousMailboxStoreError::Rejected);
    }

    // [ANONYMOUS-MAILBOX-STORE 2026-09-02 by Codex] Reserve or inspect the
    // exact final inode descriptor-relative before SQLite receives writable
    // access. Existing hardlinks therefore fail without a chmod side effect.
    let (candidate, _created) = open_or_reserve_private_file_at(&parent, name)?;
    verify_private_descriptor(&candidate)?;
    candidate
        .set_permissions(std::fs::Permissions::from_mode(0o600))
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    verify_private_descriptor(&candidate)?;
    if candidate
        .metadata()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        .permissions()
        .mode()
        & 0o777
        != 0o600
    {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    // Synchronize even an existing candidate. A prior activation may have
    // created the dirent and then received an ambiguous parent-fsync error;
    // retry must not skip that durability boundary.
    sync_parent_directory(&parent)?;
    drop(candidate);

    let resolved_parent =
        std::fs::canonicalize(parent_path).map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let resolved_metadata =
        std::fs::metadata(&resolved_parent).map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    if resolved_metadata.dev() != parent_metadata.dev()
        || resolved_metadata.ino() != parent_metadata.ino()
    {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    Ok(PrivateSqliteTarget {
        resolved_path: resolved_parent.join(name),
        parent,
    })
}

#[cfg(unix)]
fn open_directory_anchor(path: &Path) -> Result<File, AnonymousMailboxStoreError> {
    use std::os::unix::fs::OpenOptionsExt;

    let anchor = if path.is_absolute() { "/" } else { "." };
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_DIRECTORY)
        .open(anchor)
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)
}

#[cfg(unix)]
fn open_or_create_directory_at(
    parent: &File,
    name: &std::ffi::OsStr,
) -> Result<File, AnonymousMailboxStoreError> {
    match open_directory_at(parent, name) {
        Ok(directory) => Ok(directory),
        Err(AnonymousMailboxStoreError::Unavailable) => {
            match mkdirat(
                Some(parent.as_raw_fd()),
                name,
                Mode::from_bits_truncate(0o700),
            ) {
                Ok(()) | Err(nix::errno::Errno::EEXIST) => {
                    sync_parent_directory(parent)?;
                    open_directory_at(parent, name)
                }
                Err(nix::errno::Errno::ELOOP | nix::errno::Errno::ENOTDIR) => {
                    Err(AnonymousMailboxStoreError::Rejected)
                }
                Err(_) => Err(AnonymousMailboxStoreError::Unavailable),
            }
        }
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn open_directory_at(
    parent: &File,
    name: &std::ffi::OsStr,
) -> Result<File, AnonymousMailboxStoreError> {
    let raw = openat(
        Some(parent.as_raw_fd()),
        name,
        OFlag::O_RDONLY | OFlag::O_CLOEXEC | OFlag::O_NOFOLLOW | OFlag::O_DIRECTORY,
        Mode::empty(),
    )
    .map_err(|error| match error {
        nix::errno::Errno::ELOOP | nix::errno::Errno::ENOTDIR => {
            AnonymousMailboxStoreError::Rejected
        }
        _ => AnonymousMailboxStoreError::Unavailable,
    })?;
    // SAFETY: openat returned a newly owned descriptor, transferred exactly
    // once into File and closed by its Drop implementation.
    Ok(unsafe { File::from_raw_fd(raw) })
}

#[cfg(unix)]
fn open_or_reserve_private_file_at(
    parent: &File,
    name: &std::ffi::OsStr,
) -> Result<(File, bool), AnonymousMailboxStoreError> {
    let existing = openat(
        Some(parent.as_raw_fd()),
        name,
        OFlag::O_RDONLY | OFlag::O_CLOEXEC | OFlag::O_NOFOLLOW | OFlag::O_NONBLOCK,
        Mode::empty(),
    );
    let (raw, created) = match existing {
        Ok(raw) => (raw, false),
        Err(nix::errno::Errno::ENOENT) => (
            openat(
                Some(parent.as_raw_fd()),
                name,
                OFlag::O_RDWR
                    | OFlag::O_CLOEXEC
                    | OFlag::O_NOFOLLOW
                    | OFlag::O_NONBLOCK
                    | OFlag::O_CREAT
                    | OFlag::O_EXCL,
                Mode::from_bits_truncate(0o600),
            )
            .map_err(|error| match error {
                nix::errno::Errno::ELOOP | nix::errno::Errno::EEXIST => {
                    AnonymousMailboxStoreError::Rejected
                }
                _ => AnonymousMailboxStoreError::Unavailable,
            })?,
            true,
        ),
        Err(nix::errno::Errno::ELOOP) => return Err(AnonymousMailboxStoreError::Rejected),
        Err(_) => return Err(AnonymousMailboxStoreError::Unavailable),
    };
    // SAFETY: openat returned a newly owned descriptor, transferred exactly
    // once into File and closed by its Drop implementation.
    Ok((unsafe { File::from_raw_fd(raw) }, created))
}

#[cfg(unix)]
fn sync_parent_directory(parent: &File) -> Result<(), AnonymousMailboxStoreError> {
    #[cfg(test)]
    if FORCE_PARENT_SYNC_FAILURE.with(std::cell::Cell::get) {
        return Err(AnonymousMailboxStoreError::Unavailable);
    }
    parent
        .sync_all()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    #[cfg(test)]
    PARENT_DURABILITY_SYNCS.with(|calls| calls.set(calls.get().saturating_add(1)));
    Ok(())
}

#[cfg(unix)]
fn verify_private_descriptor(file: &File) -> Result<(), AnonymousMailboxStoreError> {
    let metadata = file
        .metadata()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    if !metadata.is_file() || metadata.nlink() != 1 || metadata.uid() != effective_user_id() {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    Ok(())
}

#[cfg(unix)]
fn effective_user_id() -> u32 {
    // SAFETY: geteuid has no preconditions and performs no memory access.
    unsafe { nix::libc::geteuid() }
}

#[cfg(not(unix))]
pub(crate) fn prepare_private_sqlite_target(
    path: &Path,
) -> Result<PrivateSqliteTarget, AnonymousMailboxStoreError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let resolved_path = std::fs::canonicalize(parent)
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        .join(
            path.file_name()
                .ok_or(AnonymousMailboxStoreError::Rejected)?,
        );
    if !resolved_path.exists() {
        std::fs::File::create(&resolved_path)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    }
    Ok(PrivateSqliteTarget { resolved_path })
}

#[cfg(unix)]
pub(crate) fn verify_private_file(
    path: &Path,
    require_private_mode: bool,
) -> Result<(), AnonymousMailboxStoreError> {
    let metadata =
        std::fs::symlink_metadata(path).map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.nlink() != 1
        || metadata.uid() != effective_user_id()
        || (require_private_mode && metadata.permissions().mode() & 0o777 != 0o600)
    {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn verify_private_file(
    _path: &Path,
    _require_private_mode: bool,
) -> Result<(), AnonymousMailboxStoreError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    mod custody;
    mod other;
    mod replay;

    use std::sync::{Arc, Barrier};

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::anonymous_mailbox::{
        AnonymousMailboxAdmissionTicketV1, AnonymousMailboxLeaseCreateV1,
        AnonymousMailboxTicketIssueV1,
    };
    use tempfile::TempDir;

    use super::*;

    const NOW: u64 = 1_800_000_000;
    const CURSOR_SECRET: [u8; 32] = [0xA5; 32];

    struct TestContext {
        _directory: TempDir,
        config: AnonymousMailboxStoreConfig,
        target: IdentityKeyPair,
        depositor: IdentityKeyPair,
        reader: IdentityKeyPair,
    }

    impl TestContext {
        fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let private_directory = std::fs::canonicalize(directory.path()).unwrap();
            let config = AnonymousMailboxStoreConfig {
                enabled: true,
                db_path: private_directory
                    .join("mailbox.sqlite")
                    .display()
                    .to_string(),
                max_leases_total: 8,
                max_items_total: 2_048,
                max_bytes_total: 1024 * 1024,
                max_in_flight: 4,
                cleanup_batch_size: 8,
                max_outstanding_tickets: 8,
                max_ticket_issues_per_window: 4,
                ticket_issuance_window_secs: 60,
                ticket_issue_work_bits: 1,
            };
            Self {
                _directory: directory,
                config,
                target: IdentityKeyPair::generate(),
                depositor: IdentityKeyPair::generate(),
                reader: IdentityKeyPair::generate(),
            }
        }

        fn open(&self) -> SqliteAnonymousMailboxStore {
            SqliteAnonymousMailboxStore::open(
                self.config.clone(),
                self.target.public_key_bytes(),
                CURSOR_SECRET,
            )
            .unwrap()
        }

        fn open_with_ticket_issuer(&self) -> SqliteAnonymousMailboxStore {
            SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                self.config.clone(),
                self.target.clone(),
                CURSOR_SECRET,
            )
            .unwrap()
        }

        fn lease(
            &self,
            mailbox_id: [u8; 32],
            ticket_id: [u8; 16],
            max_items: u16,
            max_bytes: u64,
            expires_at: u64,
        ) -> AnonymousMailboxLeaseCreateV1 {
            let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
                &mailbox_id,
                &self.depositor.public_key_bytes(),
                &self.reader.public_key_bytes(),
                max_items,
                max_bytes,
                NOW,
                expires_at,
            );
            let ticket = AnonymousMailboxAdmissionTicketV1::issue(
                ticket_id,
                claims,
                NOW,
                NOW + 300,
                &self.target,
            )
            .unwrap();
            AnonymousMailboxLeaseCreateV1::new(
                mailbox_id,
                self.depositor.public_key_bytes(),
                max_items,
                max_bytes,
                NOW,
                expires_at,
                ticket,
                &self.reader,
            )
            .unwrap()
        }

        fn ticket_issue(
            &self,
            request_id: [u8; 16],
            ticket_id: [u8; 16],
            mailbox_id: [u8; 32],
            lease_expires_at: u64,
            ticket_expires_at: u64,
        ) -> AnonymousMailboxTicketIssueV1 {
            let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
                &mailbox_id,
                &self.depositor.public_key_bytes(),
                &self.reader.public_key_bytes(),
                2,
                32,
                NOW,
                lease_expires_at,
            );
            for nonce in 0..u64::MAX {
                let request = AnonymousMailboxTicketIssueV1::new(
                    request_id,
                    ticket_id,
                    self.target.public_key_bytes(),
                    claims,
                    NOW,
                    ticket_expires_at,
                    nonce,
                )
                .unwrap();
                if request.proof_digest().unwrap()[0] & 0x80 == 0 {
                    return request;
                }
            }
            unreachable!("one-bit proof is reachable")
        }

        fn lease_for_issued_ticket(
            &self,
            mailbox_id: [u8; 32],
            lease_expires_at: u64,
            ticket: AnonymousMailboxAdmissionTicketV1,
        ) -> AnonymousMailboxLeaseCreateV1 {
            AnonymousMailboxLeaseCreateV1::new(
                mailbox_id,
                self.depositor.public_key_bytes(),
                2,
                32,
                NOW,
                lease_expires_at,
                ticket,
                &self.reader,
            )
            .unwrap()
        }

        fn put(
            &self,
            mailbox_id: [u8; 32],
            item_id: [u8; 16],
            bytes: &[u8],
            expires_at: u64,
        ) -> AnonymousMailboxPutV1 {
            AnonymousMailboxPutV1::new(
                mailbox_id,
                item_id,
                bytes.to_vec(),
                NOW,
                expires_at,
                &self.depositor,
            )
            .unwrap()
        }

        fn pull(
            &self,
            mailbox_id: [u8; 32],
            cursor: Vec<u8>,
            at: u64,
        ) -> AnonymousMailboxPullOneV1 {
            AnonymousMailboxPullOneV1::new(mailbox_id, [0x51; 16], cursor, at, &self.reader)
                .unwrap()
        }
    }

    fn schema_user_version(connection: &Connection) -> i64 {
        connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap()
    }

    fn anonymous_mailbox_object_count(connection: &Connection) -> i64 {
        connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE name LIKE 'anonymous_mailbox_%'",
                [],
                |row| row.get(0),
            )
            .unwrap()
    }

    // [ANONYMOUS-MAILBOX-OWNER-RESTART-RECOVERY 2026-09-14 by Codex]
    // Tie restart assertions to both durable counters and their backing rows.
    // A stale aggregate must never make released quota appear reusable.
    fn assert_item_accounting(
        store: &SqliteAnonymousMailboxStore,
        mailbox_id: &[u8; 32],
        expected_items: u64,
        expected_bytes: u64,
    ) {
        let connection = store.connection.lock();
        let totals: (i64, i64, i64) = connection
            .query_row(
                "SELECT total_leases, total_items, total_bytes
                 FROM anonymous_mailbox_meta WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        let lease: (i64, i64) = connection
            .query_row(
                "SELECT current_items, current_bytes
                 FROM anonymous_mailbox_leases WHERE mailbox_id = ?1",
                params![&mailbox_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let rows: (i64, i64) = connection
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(length(sealed_envelope)), 0)
                 FROM anonymous_mailbox_items WHERE mailbox_id = ?1",
                params![&mailbox_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let expected_items = as_i64(expected_items).unwrap();
        let expected_bytes = as_i64(expected_bytes).unwrap();
        assert_eq!(totals, (1, expected_items, expected_bytes));
        assert_eq!(lease, (expected_items, expected_bytes));
        assert_eq!(rows, (expected_items, expected_bytes));
    }
}
