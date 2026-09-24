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

const SCHEMA_VERSION: i64 = 3;
const MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
const CURSOR_VERSION: u8 = 1;
const CURSOR_BODY_BYTES: usize = 1 + 8 + 8 + 8;
const CURSOR_TAG_BYTES: usize = 32;
const CURSOR_BYTES: usize = CURSOR_BODY_BYTES + CURSOR_TAG_BYTES;
// [ANONYMOUS-MAILBOX-PULL-BOUNDS 2026-09-24 by Codex] The V3 SQLite CHECK
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

fn validate_pull_replay_retention(
    created_at: i64,
    retain_until: i64,
    lease_expires_at: Option<i64>,
) -> Result<u64, AnonymousMailboxStoreError> {
    let created_at = as_u64(created_at)?;
    let retain_until = as_u64(retain_until)?;
    let lease_expires_at = as_u64(lease_expires_at.ok_or(AnonymousMailboxStoreError::Corrupt)?)?;
    if retain_until != lease_expires_at.min(created_at.saturating_add(PULL_REPLAY_RETENTION_SECS)) {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(retain_until)
}

enum StoredPullReplay {
    Empty,
    Item {
        item_id: [u8; 16],
        sealed_commitment: [u8; 32],
        sealed_envelope: Vec<u8>,
        cursor: Vec<u8>,
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

    fn open_inner(
        config: AnonymousMailboxStoreConfig,
        target_node_id: [u8; 32],
        cursor_secret: [u8; 32],
        ticket_issuer: Option<IdentityKeyPair>,
    ) -> Result<Self, AnonymousMailboxStoreError> {
        if !config.enabled {
            return Err(AnonymousMailboxStoreError::Disabled);
        }
        if config.db_path.is_empty()
            || config.db_path == ":memory:"
            || config.max_leases_total == 0
            || i64::try_from(config.max_leases_total).is_err()
            || config.max_items_total < usize::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE)
            || i64::try_from(config.max_items_total).is_err()
            || config.max_bytes_total == 0
            || config.max_bytes_total > i64::MAX as u64
            || config.max_in_flight == 0
            || config.cleanup_batch_size == 0
            || config.max_outstanding_tickets == 0
            || i64::try_from(config.max_outstanding_tickets).is_err()
            || config.max_ticket_issues_per_window == 0
            || i64::try_from(config.max_ticket_issues_per_window).is_err()
            || config.ticket_issuance_window_secs == 0
            || config.ticket_issue_work_bits == 0
            || config.ticket_issue_work_bits
                > aeronyx_core::protocol::anonymous_mailbox::MAX_ANONYMOUS_MAILBOX_TICKET_ISSUE_WORK_BITS
            || cursor_secret == [0; 32]
        {
            return Err(AnonymousMailboxStoreError::Rejected);
        }

        let target = prepare_private_sqlite_target(Path::new(&config.db_path))?;

        // [ANONYMOUS-MAILBOX-STORE 2026-09-02 by Codex] SQLite NOFOLLOW,
        // owner-only mode, startup quick-check, WAL and FULL durability reuse
        // the existing relay custody activation policy.
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        verify_private_file(&target.resolved_path, false)?;
        restrict_private_sqlite_permissions(&target.resolved_path)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        verify_sqlite_physical_integrity(&connection, "anonymous_mailbox_startup_integrity")
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        configure_full_durability(&connection, MINIMUM_SYNCHRONOUS_LEVEL)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        initialize_or_verify_schema(&mut connection, &target_node_id, &config)?;
        let startup_limits = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        validate_totals_limits(&load_totals(&startup_limits)?, &config)?;
        validate_pull_replay_totals(load_pull_replay_totals(&startup_limits)?, &config)?;
        startup_limits
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        Ok(Self {
            config,
            target_node_id,
            ticket_issuer,
            cursor_secret,
            connection: Mutex::new(connection),
            in_flight: AtomicUsize::new(0),
            #[cfg(unix)]
            _database_parent: target.parent,
        })
    }

    fn acquire(&self) -> Result<OperationPermit<'_>, AnonymousMailboxStoreError> {
        self.in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.config.max_in_flight).then_some(current + 1)
            })
            .map_err(|_| AnonymousMailboxStoreError::Busy)?;
        Ok(OperationPermit {
            counter: &self.in_flight,
        })
    }

    fn encode_cursor(
        &self,
        mailbox_id: &[u8; 32],
        state: CursorState,
    ) -> Result<Vec<u8>, AnonymousMailboxStoreError> {
        let mut body = Vec::with_capacity(CURSOR_BYTES);
        body.push(CURSOR_VERSION);
        body.extend_from_slice(&state.snapshot_ceiling.to_le_bytes());
        body.extend_from_slice(&state.next_sequence.to_le_bytes());
        body.extend_from_slice(&state.expires_at.to_le_bytes());
        let mut mac = CursorMac::new_from_slice(&self.cursor_secret)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        mac.update(CURSOR_DOMAIN);
        mac.update(mailbox_id);
        mac.update(&body);
        body.extend_from_slice(&mac.finalize().into_bytes());
        Ok(body)
    }

    fn decode_cursor(
        &self,
        mailbox_id: &[u8; 32],
        cursor: &[u8],
        now: u64,
    ) -> Result<CursorState, AnonymousMailboxStoreError> {
        if cursor.len() != CURSOR_BYTES || cursor[0] != CURSOR_VERSION {
            return Err(AnonymousMailboxStoreError::Rejected);
        }
        let (body, tag) = cursor.split_at(CURSOR_BODY_BYTES);
        let mut mac = CursorMac::new_from_slice(&self.cursor_secret)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        mac.update(CURSOR_DOMAIN);
        mac.update(mailbox_id);
        mac.update(body);
        mac.verify_slice(tag)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let state = CursorState {
            snapshot_ceiling: u64::from_le_bytes(
                body[1..9]
                    .try_into()
                    .map_err(|_| AnonymousMailboxStoreError::Rejected)?,
            ),
            next_sequence: u64::from_le_bytes(
                body[9..17]
                    .try_into()
                    .map_err(|_| AnonymousMailboxStoreError::Rejected)?,
            ),
            expires_at: u64::from_le_bytes(
                body[17..25]
                    .try_into()
                    .map_err(|_| AnonymousMailboxStoreError::Rejected)?,
            ),
        };
        if state.next_sequence == 0
            || state.next_sequence > state.snapshot_ceiling.saturating_add(1)
            || state.expires_at < now
        {
            return Err(AnonymousMailboxStoreError::Rejected);
        }
        Ok(state)
    }
}

impl AnonymousMailboxCustodyRepository for SqliteAnonymousMailboxStore {
    fn issue_ticket(
        &self,
        request: &AnonymousMailboxTicketIssueV1,
        now: u64,
    ) -> Result<AnonymousMailboxTicketIssueOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        // [ANONYMOUS-MAILBOX-TICKET-ISSUER 2026-09-03 by Codex] Exact
        // durable replay precedes proof-of-work and every capacity check, but
        // an expired authority is never served again.
        if let Some(existing) = load_issued_ticket_by_request(&transaction, &request.request_id)? {
            if existing.request_commitment != request_commitment {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxTicketIssueOutcome::Conflict);
            }
            if existing.ticket.expires_at < now {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Err(AnonymousMailboxStoreError::Rejected);
            }
            validate_issued_ticket(&existing, request, &self.target_node_id)?;
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxTicketIssueOutcome::Existing(
                existing.ticket,
            ));
        }
        if load_issued_ticket_by_ticket(&transaction, &request.ticket_id)?.is_some() {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxTicketIssueOutcome::Conflict);
        }

        request
            .verify_for_target(
                &self.target_node_id,
                now,
                self.config.ticket_issue_work_bits,
            )
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let issuer = self
            .ticket_issuer
            .as_ref()
            .ok_or(AnonymousMailboxStoreError::Unavailable)?;

        let before = load_ticket_issue_meta(&transaction)?;
        let removed = purge_expired_issued_tickets(
            &transaction,
            now,
            u64::try_from(self.config.cleanup_batch_size)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let after_purge = TicketIssueMeta {
            outstanding: before
                .outstanding
                .checked_sub(removed.unconsumed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            ..before
        };
        if after_purge != before {
            update_ticket_issue_meta_exact(&transaction, before, after_purge)?;
        }
        let mut admission = after_purge;
        if now.saturating_sub(admission.window_started_at)
            >= self.config.ticket_issuance_window_secs
        {
            admission.window_started_at = now;
            admission.issues_in_window = 0;
        }
        let maximum_outstanding = u64::try_from(self.config.max_outstanding_tickets)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        let maximum_window = u64::try_from(self.config.max_ticket_issues_per_window)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if admission.outstanding >= maximum_outstanding
            || admission.issues_in_window >= maximum_window
        {
            if admission != after_purge {
                update_ticket_issue_meta_exact(&transaction, after_purge, admission)?;
            }
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxTicketIssueOutcome::AtCapacity);
        }
        let ticket = AnonymousMailboxAdmissionTicketV1::issue(
            request.ticket_id,
            request.lease_claims_commitment,
            request.issued_at,
            request.expires_at,
            issuer,
        )
        .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let ticket_commitment = ticket
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_issued_tickets
                     (request_id, ticket_id, request_commitment, target_node_id,
                      claims_commitment, requested_at, expires_at, proof_nonce,
                      ticket_commitment, ticket_signature, consumed_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL)",
                    params![
                        &request.request_id[..],
                        &request.ticket_id[..],
                        &request_commitment[..],
                        &request.target_node_id[..],
                        &request.lease_claims_commitment[..],
                        as_i64(request.issued_at)?,
                        as_i64(request.expires_at)?,
                        &request.proof_nonce.to_le_bytes()[..],
                        &ticket_commitment[..],
                        &ticket.signature[..],
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let issued = TicketIssueMeta {
            outstanding: admission
                .outstanding
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            issues_in_window: admission
                .issues_in_window
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            ..admission
        };
        update_ticket_issue_meta_exact(&transaction, after_purge, issued)?;
        verify_ticket_issue_meta(&transaction, issued)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxTicketIssueOutcome::Issued(ticket))
    }

    fn create(
        &self,
        request: &AnonymousMailboxLeaseCreateV1,
        now: u64,
    ) -> Result<AnonymousMailboxCreateOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let ticket_commitment = request
            .admission
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let claims_commitment = request.claims_commitment();
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        if let Some((stored_ticket, stored_claims, stored_request, mailbox)) = transaction
            .query_row(
                "SELECT ticket_commitment, claims_commitment, lease_request_commitment, mailbox_id
                   FROM anonymous_mailbox_tickets WHERE ticket_id = ?1",
                params![&request.admission.ticket_id[..]],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        {
            let exact_authority =
                stored_ticket == ticket_commitment && stored_request == request_commitment;
            if !exact_authority {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxCreateOutcome::Conflict);
            }
            if stored_claims != claims_commitment || mailbox != request.mailbox_id {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            let lease = load_lease(&transaction, &request.mailbox_id)?
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            if lease.mailbox_id != request.mailbox_id
                || lease.deposit_verifier != request.deposit_verifier
                || lease.read_verifier != request.read_verifier
                || lease.max_items != request.max_items
                || lease.max_bytes != request.max_bytes
                || lease.created_at != request.issued_at
                || lease.expires_at != request.expires_at
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxCreateOutcome::Existing(lease));
        }

        request
            .verify_for_target(&self.target_node_id, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        if totals.leases >= self.config.max_leases_total as u64 {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxCreateOutcome::AtCapacity);
        }
        if load_lease(&transaction, &request.mailbox_id)?.is_some() {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxCreateOutcome::Conflict);
        }
        let issued_ticket =
            load_issued_ticket_by_ticket(&transaction, &request.admission.ticket_id)?;
        if let Some(record) = &issued_ticket {
            if record.ticket != request.admission
                || record.consumed_at.is_some()
                || record.ticket.expires_at < now
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            validate_issued_ticket(&record, &record.request, &self.target_node_id)?;
        }

        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_leases
                 (mailbox_id, ticket_id, claims_commitment, deposit_verifier, read_verifier,
                  max_items, max_bytes, current_items, current_bytes, next_sequence,
                  created_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 0, 1, ?8, ?9)",
                    params![
                        &request.mailbox_id[..],
                        &request.admission.ticket_id[..],
                        &claims_commitment[..],
                        &request.deposit_verifier[..],
                        &request.read_verifier[..],
                        i64::from(request.max_items),
                        as_i64(request.max_bytes)?,
                        as_i64(request.issued_at)?,
                        as_i64(request.expires_at)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_tickets
                 (ticket_id, ticket_commitment, claims_commitment, lease_request_commitment,
                  mailbox_id, consumed_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        &request.admission.ticket_id[..],
                        &ticket_commitment[..],
                        &claims_commitment[..],
                        &request_commitment[..],
                        &request.mailbox_id[..],
                        as_i64(now)?,
                        as_i64(request.admission.expires_at)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        if let Some(record) = issued_ticket {
            let before = load_ticket_issue_meta(&transaction)?;
            let after = TicketIssueMeta {
                outstanding: before
                    .outstanding
                    .checked_sub(1)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                ..before
            };
            execute_exactly_one(
                &transaction,
                transaction
                    .execute(
                        "UPDATE anonymous_mailbox_issued_tickets SET consumed_at = ?1
                         WHERE ticket_id = ?2 AND consumed_at IS NULL",
                        params![as_i64(now)?, &record.ticket.ticket_id[..]],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
            )?;
            update_ticket_issue_meta_exact(&transaction, before, after)?;
            verify_ticket_issue_meta(&transaction, after)?;
        }
        let new_totals = StoreTotals {
            leases: totals
                .leases
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            ..totals
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        let lease = load_lease(&transaction, &request.mailbox_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if lease.current_items != 0 || lease.current_bytes != 0 || lease.next_sequence != 1 {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxCreateOutcome::Created(lease))
    }

    fn put(
        &self,
        request: &AnonymousMailboxPutV1,
        now: u64,
    ) -> Result<AnonymousMailboxPutOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let sealed_commitment = request.sealed_commitment();
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        if let Some(existing) = load_item(&transaction, &request.mailbox_id, &request.item_id)? {
            let stored_request: Vec<u8> = transaction
                .query_row(
                    "SELECT put_commitment FROM anonymous_mailbox_items
                     WHERE mailbox_id = ?1 AND item_id = ?2",
                    params![&request.mailbox_id[..], &request.item_id[..]],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            let outcome = if stored_request == request_commitment
                && existing.sealed_commitment == sealed_commitment
                && existing.sealed_envelope == request.sealed_envelope
            {
                AnonymousMailboxPutOutcome::Existing(existing)
            } else {
                AnonymousMailboxPutOutcome::Conflict
            };
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(outcome);
        }
        if load_ack_commitment(&transaction, &request.mailbox_id, &request.item_id)?.is_some() {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPutOutcome::Conflict);
        }

        let lease = match load_lease(&transaction, &request.mailbox_id)? {
            Some(lease) => lease,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxPutOutcome::LeaseNotFound);
            }
        };
        request
            .verify_at(&lease.deposit_verifier, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        if lease.expires_at < now || request.expires_at > lease.expires_at {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPutOutcome::LeaseExpired);
        }
        let item_bytes = u64::try_from(request.sealed_envelope.len())
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let new_items = u64::from(lease.current_items)
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let new_lease_bytes = lease
            .current_bytes
            .checked_add(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        let new_total_items = totals
            .items
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let new_total_bytes = totals
            .bytes
            .checked_add(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if new_items > u64::from(lease.max_items)
            || new_lease_bytes > lease.max_bytes
            || new_total_items
                > u64::try_from(self.config.max_items_total)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            || new_total_bytes > self.config.max_bytes_total
        {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPutOutcome::AtCapacity);
        }
        let sequence = lease.next_sequence;
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_items
                 (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                  sealed_envelope, stored_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        &request.mailbox_id[..],
                        &request.item_id[..],
                        as_i64(sequence)?,
                        &request_commitment[..],
                        &sealed_commitment[..],
                        &request.sealed_envelope,
                        as_i64(now)?,
                        as_i64(request.expires_at)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "UPDATE anonymous_mailbox_leases
                 SET current_items = ?1, current_bytes = ?2, next_sequence = ?3
                 WHERE mailbox_id = ?4 AND current_items = ?5
                   AND current_bytes = ?6 AND next_sequence = ?7",
                    params![
                        as_i64(new_items)?,
                        as_i64(new_lease_bytes)?,
                        as_i64(next_sequence)?,
                        &request.mailbox_id[..],
                        i64::from(lease.current_items),
                        as_i64(lease.current_bytes)?,
                        as_i64(lease.next_sequence)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let new_totals = StoreTotals {
            leases: totals.leases,
            items: new_total_items,
            bytes: new_total_bytes,
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        let item = load_item(&transaction, &request.mailbox_id, &request.item_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let updated_lease = load_lease(&transaction, &request.mailbox_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if u64::from(updated_lease.current_items) != new_items
            || updated_lease.current_bytes != new_lease_bytes
            || updated_lease.next_sequence != next_sequence
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxPutOutcome::Stored(item))
    }

    fn pull_one(
        &self,
        request: &AnonymousMailboxPullOneV1,
        now: u64,
    ) -> Result<AnonymousMailboxPullOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        // [ANONYMOUS-MAILBOX-PULL-REPLAY 2026-09-24 by Codex] Resolve the
        // exact signed request before freshness and live-item lookup. This is
        // the durable authority for a response lost immediately before ACK.
        if let Some(outcome) = load_pull_replay(&transaction, request, &request_commitment, now)? {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(outcome);
        }
        let lease = match load_lease(&transaction, &request.mailbox_id)? {
            Some(lease) => lease,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxPullOutcome::LeaseNotFound);
            }
        };
        request
            .verify_at(&lease.read_verifier, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        if lease.expires_at < now {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPullOutcome::LeaseExpired);
        }
        let state = if request.cursor.is_empty() {
            CursorState {
                snapshot_ceiling: lease.next_sequence.saturating_sub(1),
                next_sequence: 1,
                expires_at: lease.expires_at.min(now.saturating_add(CURSOR_TTL_SECS)),
            }
        } else {
            let state = self.decode_cursor(&request.mailbox_id, &request.cursor, now)?;
            if state.snapshot_ceiling >= lease.next_sequence {
                return Err(AnonymousMailboxStoreError::Rejected);
            }
            state
        };
        let replay_retain_until = lease
            .expires_at
            .min(now.saturating_add(PULL_REPLAY_RETENTION_SECS));
        let row = transaction
            .query_row(
                "SELECT item_id, sequence, sealed_commitment, length(sealed_envelope),
                        stored_at, expires_at
                 FROM anonymous_mailbox_items
                 WHERE mailbox_id = ?1 AND sequence >= ?2 AND sequence <= ?3
                   AND expires_at >= ?4
                 ORDER BY sequence ASC LIMIT 1",
                params![
                    &request.mailbox_id[..],
                    as_i64(state.next_sequence)?,
                    as_i64(state.snapshot_ceiling)?,
                    as_i64(now)?,
                ],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let Some((item_id_bytes, sequence_raw, commitment_bytes, length, _, _)) = row else {
            insert_pull_replay(
                &transaction,
                &self.config,
                request,
                &request_commitment,
                StoredPullReplay::Empty,
                now,
                replay_retain_until,
            )?;
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPullOutcome::Empty);
        };
        let sequence = as_u64(sequence_raw)?;
        let admitted_length =
            usize::try_from(length).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if admitted_length == 0 || admitted_length > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        // The length admission above occurs in the same snapshot before this
        // BLOB is materialized.
        let sealed_envelope: Vec<u8> = transaction
            .query_row(
                "SELECT sealed_envelope FROM anonymous_mailbox_items
                 WHERE mailbox_id = ?1 AND item_id = ?2",
                params![&request.mailbox_id[..], &item_id_bytes],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        if sealed_envelope.len() != admitted_length {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        let item_id = fixed::<16>(&item_id_bytes)?;
        let sealed_commitment = fixed::<32>(&commitment_bytes)?;
        let calculated_commitment: [u8; 32] = Sha256::digest(&sealed_envelope).into();
        if sealed_commitment != calculated_commitment {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        let mut padded = vec![0_u8; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES];
        padded[..sealed_envelope.len()].copy_from_slice(&sealed_envelope);
        let cursor = self.encode_cursor(
            &request.mailbox_id,
            CursorState {
                snapshot_ceiling: state.snapshot_ceiling,
                next_sequence: sequence
                    .checked_add(1)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                expires_at: state.expires_at,
            },
        )?;
        insert_pull_replay(
            &transaction,
            &self.config,
            request,
            &request_commitment,
            StoredPullReplay::Item {
                item_id,
                sealed_commitment,
                sealed_envelope: sealed_envelope.clone(),
                cursor: cursor.clone(),
            },
            now,
            replay_retain_until,
        )?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxPullOutcome::Item(
            AnonymousMailboxPulledItem {
                item_id,
                sealed_commitment,
                sealed_length: u32::try_from(sealed_envelope.len())
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
                padded_sealed_envelope: padded,
                cursor,
            },
        ))
    }

    fn ack(
        &self,
        request: &AnonymousMailboxAckV1,
        now: u64,
    ) -> Result<AnonymousMailboxAckOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        // [ANONYMOUS-MAILBOX-STORE 2026-09-02 by Codex] A durable exact ACK
        // replay is decided before request freshness and lease lookup. The
        // tombstone is the retry authority; a restarted caller must not lose
        // idempotency merely because the original signature window elapsed.
        if let Some((stored_commitment, stored_request)) = transaction
            .query_row(
                "SELECT sealed_commitment, ack_request_commitment
                 FROM anonymous_mailbox_acks WHERE mailbox_id = ?1 AND item_id = ?2",
                params![&request.mailbox_id[..], &request.item_id[..]],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(
                if stored_commitment == request.sealed_commitment
                    && stored_request == request_commitment
                {
                    AnonymousMailboxAckOutcome::AlreadyAcknowledged
                } else {
                    AnonymousMailboxAckOutcome::Conflict
                },
            );
        }
        let lease = match load_lease(&transaction, &request.mailbox_id)? {
            Some(lease) => lease,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxAckOutcome::NotFound);
            }
        };
        request
            .verify_at(&lease.read_verifier, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        if lease.expires_at < now {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxAckOutcome::NotFound);
        }
        let item = match load_item(&transaction, &request.mailbox_id, &request.item_id)? {
            Some(item) => item,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxAckOutcome::NotFound);
            }
        };
        if item.sealed_commitment != request.sealed_commitment {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxAckOutcome::Conflict);
        }
        let retain_until = lease.expires_at.min(
            now.saturating_add(ACK_TOMBSTONE_RETENTION_SECS)
                .max(item.expires_at),
        );
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_items WHERE mailbox_id = ?1 AND item_id = ?2",
                    params![&request.mailbox_id[..], &request.item_id[..]],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_acks
                 (mailbox_id, item_id, sealed_commitment, ack_request_commitment,
                  acknowledged_at, retain_until)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        &request.mailbox_id[..],
                        &request.item_id[..],
                        &request.sealed_commitment[..],
                        &request_commitment[..],
                        as_i64(now)?,
                        as_i64(retain_until)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let item_bytes = u64::try_from(item.sealed_envelope.len())
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        let current_items = u64::from(lease.current_items)
            .checked_sub(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let current_bytes = lease
            .current_bytes
            .checked_sub(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let total_items = totals
            .items
            .checked_sub(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let total_bytes = totals
            .bytes
            .checked_sub(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "UPDATE anonymous_mailbox_leases SET current_items = ?1, current_bytes = ?2
                 WHERE mailbox_id = ?3 AND current_items = ?4 AND current_bytes = ?5",
                    params![
                        as_i64(current_items)?,
                        as_i64(current_bytes)?,
                        &request.mailbox_id[..],
                        i64::from(lease.current_items),
                        as_i64(lease.current_bytes)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let new_totals = StoreTotals {
            leases: totals.leases,
            items: total_items,
            bytes: total_bytes,
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        let updated_lease = load_lease(&transaction, &request.mailbox_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if u64::from(updated_lease.current_items) != current_items
            || updated_lease.current_bytes != current_bytes
            || updated_lease.next_sequence != lease.next_sequence
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxAckOutcome::Acknowledged)
    }

    fn cleanup(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxCleanupReport, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        let mut report = AnonymousMailboxCleanupReport::default();
        let mut remaining = u64::try_from(self.config.cleanup_batch_size)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;

        let replay_totals = load_pull_replay_totals(&transaction)?;
        validate_pull_replay_totals(replay_totals, &self.config)?;
        if remaining > 0 {
            let (rows, bytes): (i64, i64) = transaction
                .query_row(
                    "SELECT COUNT(*), COALESCE(SUM(length(sealed_envelope)), 0)
                     FROM anonymous_mailbox_pull_replays WHERE rowid IN
                     (SELECT rowid FROM anonymous_mailbox_pull_replays
                      WHERE retain_until < ?1 ORDER BY retain_until, rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            let rows = as_u64(rows)?;
            let bytes = as_u64(bytes)?;
            if rows > 0 {
                let removed = transaction
                    .execute(
                        "DELETE FROM anonymous_mailbox_pull_replays WHERE rowid IN
                         (SELECT rowid FROM anonymous_mailbox_pull_replays
                          WHERE retain_until < ?1 ORDER BY retain_until, rowid LIMIT ?2)",
                        params![as_i64(now)?, as_i64(remaining)?],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                if u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)? != rows
                {
                    return Err(AnonymousMailboxStoreError::Corrupt);
                }
                let updated = PullReplayTotals {
                    rows: replay_totals
                        .rows
                        .checked_sub(rows)
                        .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                    bytes: replay_totals
                        .bytes
                        .checked_sub(bytes)
                        .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                };
                update_pull_replay_totals_exact(&transaction, replay_totals, updated)?;
                report.pull_replays_removed = rows;
                remaining = remaining
                    .checked_sub(rows)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            }
        }

        let expired_items = select_expired_items(&transaction, now, remaining)?;
        for (mailbox_id, item_id, bytes) in expired_items {
            let mailbox_key = fixed::<32>(&mailbox_id)?;
            let lease = load_lease(&transaction, &mailbox_key)?
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            let updated_items = u64::from(lease.current_items)
                .checked_sub(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            let updated_bytes = lease
                .current_bytes
                .checked_sub(bytes)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            execute_exactly_one(
                &transaction,
                transaction
                    .execute(
                        "DELETE FROM anonymous_mailbox_items
                     WHERE mailbox_id = ?1 AND item_id = ?2",
                        params![&mailbox_id, &item_id],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
            )?;
            execute_exactly_one(
                &transaction,
                transaction
                    .execute(
                        "UPDATE anonymous_mailbox_leases
                     SET current_items = ?1, current_bytes = ?2
                     WHERE mailbox_id = ?3 AND current_items = ?4 AND current_bytes = ?5",
                        params![
                            as_i64(updated_items)?,
                            as_i64(updated_bytes)?,
                            &mailbox_id,
                            i64::from(lease.current_items),
                            as_i64(lease.current_bytes)?,
                        ],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
            )?;
            let updated = load_lease(&transaction, &mailbox_key)?
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            if u64::from(updated.current_items) != updated_items
                || updated.current_bytes != updated_bytes
                || updated.next_sequence != lease.next_sequence
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            report.items_removed = report
                .items_removed
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            report.bytes_removed = report
                .bytes_removed
                .checked_add(bytes)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }

        if remaining > 0 {
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_acks WHERE rowid IN
                     (SELECT rowid FROM anonymous_mailbox_acks
                      WHERE retain_until < ?1 ORDER BY retain_until, rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            report.acknowledgements_removed =
                u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(report.acknowledgements_removed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
        if remaining > 0 {
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_leases WHERE rowid IN
                     (SELECT l.rowid FROM anonymous_mailbox_leases l
                      WHERE l.expires_at < ?1 AND l.current_items = 0 AND l.current_bytes = 0
                        AND NOT EXISTS (SELECT 1 FROM anonymous_mailbox_items i
                                        WHERE i.mailbox_id = l.mailbox_id)
                        AND NOT EXISTS (SELECT 1 FROM anonymous_mailbox_acks a
                                        WHERE a.mailbox_id = l.mailbox_id)
                        AND NOT EXISTS (SELECT 1 FROM anonymous_mailbox_pull_replays r
                                        WHERE r.mailbox_id = l.mailbox_id)
                      ORDER BY l.expires_at, l.rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            report.leases_removed =
                u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(report.leases_removed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
        if remaining > 0 {
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_tickets WHERE rowid IN
                     (SELECT t.rowid FROM anonymous_mailbox_tickets t
                      LEFT JOIN anonymous_mailbox_leases l ON l.mailbox_id = t.mailbox_id
                      WHERE t.expires_at < ?1 AND l.mailbox_id IS NULL
                      ORDER BY t.expires_at, t.rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            report.tickets_removed =
                u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(report.tickets_removed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
        if remaining > 0 {
            let before = load_ticket_issue_meta(&transaction)?;
            let purge = purge_expired_issued_tickets(&transaction, now, remaining)?;
            let after = TicketIssueMeta {
                outstanding: before
                    .outstanding
                    .checked_sub(purge.unconsumed)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                ..before
            };
            if after != before {
                update_ticket_issue_meta_exact(&transaction, before, after)?;
                verify_ticket_issue_meta(&transaction, after)?;
            }
            report.issued_tickets_removed = purge.removed;
        }

        let leases = totals
            .leases
            .checked_sub(report.leases_removed)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let items = totals
            .items
            .checked_sub(report.items_removed)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let bytes = totals
            .bytes
            .checked_sub(report.bytes_removed)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let new_totals = StoreTotals {
            leases,
            items,
            bytes,
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(report)
    }
}

fn initialize_or_verify_schema(
    connection: &mut Connection,
    target_node_id: &[u8; 32],
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let user_version: i64 = transaction
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if user_version == 0 {
        // [ANONYMOUS-MAILBOX-SCHEMA-OWNERSHIP 2026-09-02 by Codex]
        // Claim only an otherwise-empty reserved database so unrelated
        // permanent schema objects never cohabit this node-blind store.
        let foreign_schema_objects: i64 = transaction
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type IN ('table', 'index', 'view', 'trigger')
                   AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if foreign_schema_objects != 0 {
            return Err(AnonymousMailboxStoreError::UnsupportedSchema);
        }
        transaction
            .execute_batch(
                "CREATE TABLE anonymous_mailbox_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL,
                    total_leases INTEGER NOT NULL CHECK (total_leases >= 0),
                    total_items INTEGER NOT NULL CHECK (total_items >= 0),
                    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
                    outstanding_tickets INTEGER NOT NULL CHECK (outstanding_tickets >= 0),
                    issuance_window_started_at INTEGER NOT NULL CHECK (issuance_window_started_at >= 0),
                    issues_in_window INTEGER NOT NULL CHECK (issues_in_window >= 0),
                    pull_replay_rows INTEGER NOT NULL CHECK (pull_replay_rows >= 0),
                    pull_replay_bytes INTEGER NOT NULL CHECK (pull_replay_bytes >= 0)
                 );
                 INSERT INTO anonymous_mailbox_meta VALUES (1, 3, 0, 0, 0, 0, 0, 0, 0, 0);
                 CREATE TABLE anonymous_mailbox_tickets (
                    ticket_id BLOB PRIMARY KEY CHECK (length(ticket_id) = 16),
                    ticket_commitment BLOB NOT NULL CHECK (length(ticket_commitment) = 32),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    lease_request_commitment BLOB NOT NULL CHECK (length(lease_request_commitment) = 32),
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    consumed_at INTEGER NOT NULL CHECK (consumed_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= consumed_at)
                 );
                 CREATE TABLE anonymous_mailbox_leases (
                    mailbox_id BLOB PRIMARY KEY CHECK (length(mailbox_id) = 32),
                    ticket_id BLOB NOT NULL UNIQUE CHECK (length(ticket_id) = 16),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    deposit_verifier BLOB NOT NULL CHECK (length(deposit_verifier) = 32),
                    read_verifier BLOB NOT NULL CHECK (length(read_verifier) = 32),
                    max_items INTEGER NOT NULL CHECK (max_items > 0),
                    max_bytes INTEGER NOT NULL CHECK (max_bytes > 0),
                    current_items INTEGER NOT NULL CHECK (current_items >= 0 AND current_items <= max_items),
                    current_bytes INTEGER NOT NULL CHECK (current_bytes >= 0 AND current_bytes <= max_bytes),
                    next_sequence INTEGER NOT NULL CHECK (next_sequence > 0),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= created_at)
                 );
                 CREATE TABLE anonymous_mailbox_items (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    item_id BLOB NOT NULL CHECK (length(item_id) = 16),
                    sequence INTEGER NOT NULL CHECK (sequence > 0),
                    put_commitment BLOB NOT NULL CHECK (length(put_commitment) = 32),
                    sealed_commitment BLOB NOT NULL CHECK (length(sealed_commitment) = 32),
                    sealed_envelope BLOB NOT NULL CHECK (length(sealed_envelope) > 0),
                    stored_at INTEGER NOT NULL CHECK (stored_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= stored_at),
                    PRIMARY KEY (mailbox_id, item_id),
                    UNIQUE (mailbox_id, sequence),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE
                 );
                 CREATE TABLE anonymous_mailbox_acks (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    item_id BLOB NOT NULL CHECK (length(item_id) = 16),
                    sealed_commitment BLOB NOT NULL CHECK (length(sealed_commitment) = 32),
                    ack_request_commitment BLOB NOT NULL CHECK (length(ack_request_commitment) = 32),
                    acknowledged_at INTEGER NOT NULL CHECK (acknowledged_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= acknowledged_at),
                    PRIMARY KEY (mailbox_id, item_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE
                 );
                 CREATE TABLE anonymous_mailbox_issued_tickets (
                    request_id BLOB PRIMARY KEY CHECK (length(request_id) = 16),
                    ticket_id BLOB NOT NULL UNIQUE CHECK (length(ticket_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    target_node_id BLOB NOT NULL CHECK (length(target_node_id) = 32),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    requested_at INTEGER NOT NULL CHECK (requested_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= requested_at),
                    proof_nonce BLOB NOT NULL CHECK (length(proof_nonce) = 8),
                    ticket_commitment BLOB NOT NULL CHECK (length(ticket_commitment) = 32),
                    ticket_signature BLOB NOT NULL CHECK (length(ticket_signature) = 64),
                    consumed_at INTEGER CHECK (consumed_at >= requested_at)
                 );
                 CREATE TABLE anonymous_mailbox_pull_replays (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    request_id BLOB NOT NULL CHECK (length(request_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    outcome INTEGER NOT NULL CHECK (outcome IN (0, 1)),
                    item_id BLOB CHECK (item_id IS NULL OR length(item_id) = 16),
                    sealed_commitment BLOB CHECK (sealed_commitment IS NULL OR length(sealed_commitment) = 32),
                    sealed_envelope BLOB,
                    cursor BLOB CHECK (cursor IS NULL OR (typeof(cursor) = 'blob' AND length(cursor) = 57)),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= created_at),
                    PRIMARY KEY (mailbox_id, request_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE,
                    CHECK ((outcome = 0 AND item_id IS NULL AND sealed_commitment IS NULL
                            AND sealed_envelope IS NULL AND cursor IS NULL)
                        OR (outcome = 1 AND item_id IS NOT NULL AND sealed_commitment IS NOT NULL
                            AND sealed_envelope IS NOT NULL AND typeof(sealed_envelope) = 'blob'
                            AND length(sealed_envelope) BETWEEN 1 AND 162816
                            AND cursor IS NOT NULL))
                 );
                 CREATE INDEX anonymous_mailbox_lease_expiry
                    ON anonymous_mailbox_leases(expires_at, mailbox_id);
                 CREATE INDEX anonymous_mailbox_item_pull
                    ON anonymous_mailbox_items(mailbox_id, sequence, expires_at);
                 CREATE INDEX anonymous_mailbox_item_expiry
                    ON anonymous_mailbox_items(expires_at, mailbox_id, item_id);
                 CREATE INDEX anonymous_mailbox_ack_expiry
                    ON anonymous_mailbox_acks(retain_until, mailbox_id, item_id);
                 CREATE INDEX anonymous_mailbox_issued_ticket_expiry
                    ON anonymous_mailbox_issued_tickets(expires_at, request_id);
                 CREATE INDEX anonymous_mailbox_pull_replay_expiry
                    ON anonymous_mailbox_pull_replays(retain_until, mailbox_id, request_id);
                 PRAGMA user_version = 3;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    } else if user_version == 1 {
        // [ANONYMOUS-MAILBOX-TICKET-ISSUER 2026-09-03 by Codex] The v2
        // journal is additive: existing consumed tickets and leases retain
        // their frozen rows while only future target-issued authorities gain
        // exact replay evidence.
        transaction
            .execute_batch(
                "ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN outstanding_tickets INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN issuance_window_started_at INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN issues_in_window INTEGER NOT NULL DEFAULT 0;
                 CREATE TABLE anonymous_mailbox_issued_tickets (
                    request_id BLOB PRIMARY KEY CHECK (length(request_id) = 16),
                    ticket_id BLOB NOT NULL UNIQUE CHECK (length(ticket_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    target_node_id BLOB NOT NULL CHECK (length(target_node_id) = 32),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    requested_at INTEGER NOT NULL CHECK (requested_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= requested_at),
                    proof_nonce BLOB NOT NULL CHECK (length(proof_nonce) = 8),
                    ticket_commitment BLOB NOT NULL CHECK (length(ticket_commitment) = 32),
                    ticket_signature BLOB NOT NULL CHECK (length(ticket_signature) = 64),
                    consumed_at INTEGER CHECK (consumed_at >= requested_at)
                 );
                 CREATE INDEX anonymous_mailbox_issued_ticket_expiry
                    ON anonymous_mailbox_issued_tickets(expires_at, request_id);
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_rows INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_bytes INTEGER NOT NULL DEFAULT 0;
                 CREATE TABLE anonymous_mailbox_pull_replays (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    request_id BLOB NOT NULL CHECK (length(request_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    outcome INTEGER NOT NULL CHECK (outcome IN (0, 1)),
                    item_id BLOB CHECK (item_id IS NULL OR length(item_id) = 16),
                    sealed_commitment BLOB CHECK (sealed_commitment IS NULL OR length(sealed_commitment) = 32),
                    sealed_envelope BLOB,
                    cursor BLOB CHECK (cursor IS NULL OR (typeof(cursor) = 'blob' AND length(cursor) = 57)),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= created_at),
                    PRIMARY KEY (mailbox_id, request_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE,
                    CHECK ((outcome = 0 AND item_id IS NULL AND sealed_commitment IS NULL
                            AND sealed_envelope IS NULL AND cursor IS NULL)
                        OR (outcome = 1 AND item_id IS NOT NULL AND sealed_commitment IS NOT NULL
                            AND sealed_envelope IS NOT NULL AND typeof(sealed_envelope) = 'blob'
                            AND length(sealed_envelope) BETWEEN 1 AND 162816
                            AND cursor IS NOT NULL))
                 );
                 CREATE INDEX anonymous_mailbox_pull_replay_expiry
                    ON anonymous_mailbox_pull_replays(retain_until, mailbox_id, request_id);
                 UPDATE anonymous_mailbox_meta SET schema_version = 3 WHERE singleton = 1;
                 PRAGMA user_version = 3;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    } else if user_version == 2 {
        // [ANONYMOUS-MAILBOX-PULL-REPLAY 2026-09-24 by Codex] V3 adds only
        // node-blind, bounded replay evidence; all V1/V2 leases, tickets,
        // items, and ACK tombstones remain byte-for-byte readable.
        transaction
            .execute_batch(
                "ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_rows INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_bytes INTEGER NOT NULL DEFAULT 0;
                 CREATE TABLE anonymous_mailbox_pull_replays (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    request_id BLOB NOT NULL CHECK (length(request_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    outcome INTEGER NOT NULL CHECK (outcome IN (0, 1)),
                    item_id BLOB CHECK (item_id IS NULL OR length(item_id) = 16),
                    sealed_commitment BLOB CHECK (sealed_commitment IS NULL OR length(sealed_commitment) = 32),
                    sealed_envelope BLOB,
                    cursor BLOB CHECK (cursor IS NULL OR (typeof(cursor) = 'blob' AND length(cursor) = 57)),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= created_at),
                    PRIMARY KEY (mailbox_id, request_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE,
                    CHECK ((outcome = 0 AND item_id IS NULL AND sealed_commitment IS NULL
                            AND sealed_envelope IS NULL AND cursor IS NULL)
                        OR (outcome = 1 AND item_id IS NOT NULL AND sealed_commitment IS NOT NULL
                            AND sealed_envelope IS NOT NULL AND typeof(sealed_envelope) = 'blob'
                            AND length(sealed_envelope) BETWEEN 1 AND 162816
                            AND cursor IS NOT NULL))
                 );
                 CREATE INDEX anonymous_mailbox_pull_replay_expiry
                    ON anonymous_mailbox_pull_replays(retain_until, mailbox_id, request_id);
                 UPDATE anonymous_mailbox_meta SET schema_version = 3 WHERE singleton = 1;
                 PRAGMA user_version = 3;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    } else if user_version != SCHEMA_VERSION {
        return Err(AnonymousMailboxStoreError::UnsupportedSchema);
    }
    let version: i64 = transaction
        .query_row(
            "SELECT schema_version FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if version != SCHEMA_VERSION {
        return Err(AnonymousMailboxStoreError::UnsupportedSchema);
    }
    transaction
        .commit()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    audit_counters(&transaction, target_node_id, config)?;
    transaction
        .commit()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)
}

fn load_totals(transaction: &Transaction<'_>) -> Result<StoreTotals, AnonymousMailboxStoreError> {
    let (leases, items, bytes): (i64, i64, i64) = transaction
        .query_row(
            "SELECT total_leases, total_items, total_bytes
             FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    Ok(StoreTotals {
        leases: as_u64(leases)?,
        items: as_u64(items)?,
        bytes: as_u64(bytes)?,
    })
}

fn load_pull_replay_totals(
    transaction: &Transaction<'_>,
) -> Result<PullReplayTotals, AnonymousMailboxStoreError> {
    let (rows, bytes): (i64, i64) = transaction
        .query_row(
            "SELECT pull_replay_rows, pull_replay_bytes
             FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    Ok(PullReplayTotals {
        rows: as_u64(rows)?,
        bytes: as_u64(bytes)?,
    })
}

fn validate_pull_replay_totals(
    totals: PullReplayTotals,
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    if totals.rows
        > u64::try_from(config.max_items_total).map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        || totals.bytes > config.max_bytes_total
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

fn update_pull_replay_totals_exact(
    transaction: &Transaction<'_>,
    old: PullReplayTotals,
    new: PullReplayTotals,
) -> Result<(), AnonymousMailboxStoreError> {
    let affected = transaction
        .execute(
            "UPDATE anonymous_mailbox_meta
             SET pull_replay_rows = ?1, pull_replay_bytes = ?2
             WHERE singleton = 1 AND pull_replay_rows = ?3 AND pull_replay_bytes = ?4",
            params![
                as_i64(new.rows)?,
                as_i64(new.bytes)?,
                as_i64(old.rows)?,
                as_i64(old.bytes)?,
            ],
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    execute_exactly_one(transaction, affected)
}

fn load_pull_replay(
    transaction: &Transaction<'_>,
    request: &AnonymousMailboxPullOneV1,
    request_commitment: &[u8; 32],
    now: u64,
) -> Result<Option<AnonymousMailboxPullOutcome>, AnonymousMailboxStoreError> {
    let metadata = transaction
        .query_row(
            "SELECT r.rowid, r.outcome, r.created_at, r.retain_until, l.expires_at,
                    typeof(r.request_commitment), length(r.request_commitment),
                    typeof(r.item_id), length(r.item_id),
                    typeof(r.sealed_commitment), length(r.sealed_commitment),
                    typeof(r.sealed_envelope), length(r.sealed_envelope),
                    typeof(r.cursor), length(r.cursor)
             FROM anonymous_mailbox_pull_replays r
             LEFT JOIN anonymous_mailbox_leases l ON l.mailbox_id = r.mailbox_id
             WHERE r.mailbox_id = ?1 AND r.request_id = ?2",
            params![&request.mailbox_id[..], &request.request_id[..]],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                    PullReplayShape::from_row(row, 5)?,
                ))
            },
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let Some((rowid, outcome, created_at, retain_until, lease_expires_at, shape)) = metadata else {
        return Ok(None);
    };
    if !shape.request.is_exact_blob(32) {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let stored_request: Vec<u8> = transaction
        .query_row(
            "SELECT request_commitment FROM anonymous_mailbox_pull_replays WHERE rowid = ?1",
            params![rowid],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if stored_request != request_commitment[..] {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    shape.checked_envelope_bytes(outcome)?;
    let retain_until = validate_pull_replay_retention(created_at, retain_until, lease_expires_at)?;
    // [ANONYMOUS-MAILBOX-PULL-BOUNDS 2026-09-24 by Codex] Expired exact
    // replays cannot outlive their lease or the 24-hour journal window merely
    // because cleanup has not reached their row yet.
    if now > retain_until {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    let stored = if outcome == 0 {
        StoredPullReplay::Empty
    } else {
        let (item_id, commitment, sealed_envelope, cursor): (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) =
            transaction
                .query_row(
                    "SELECT item_id, sealed_commitment, sealed_envelope, cursor
                     FROM anonymous_mailbox_pull_replays WHERE rowid = ?1",
                    params![rowid],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        StoredPullReplay::Item {
            item_id: fixed::<16>(&item_id)?,
            sealed_commitment: fixed::<32>(&commitment)?,
            sealed_envelope,
            cursor,
        }
    };
    match stored {
        StoredPullReplay::Empty => Ok(Some(AnonymousMailboxPullOutcome::Empty)),
        StoredPullReplay::Item {
            item_id,
            sealed_commitment,
            sealed_envelope,
            cursor,
        } => {
            if sealed_envelope.is_empty()
                || sealed_envelope.len() > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
                || cursor.len() != CURSOR_BYTES
                || <[u8; 32]>::from(Sha256::digest(&sealed_envelope)) != sealed_commitment
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            let mut padded = vec![0_u8; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES];
            padded[..sealed_envelope.len()].copy_from_slice(&sealed_envelope);
            Ok(Some(AnonymousMailboxPullOutcome::Item(
                AnonymousMailboxPulledItem {
                    item_id,
                    sealed_commitment,
                    sealed_length: u32::try_from(sealed_envelope.len())
                        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
                    padded_sealed_envelope: padded,
                    cursor,
                },
            )))
        }
    }
}

fn insert_pull_replay(
    transaction: &Transaction<'_>,
    config: &AnonymousMailboxStoreConfig,
    request: &AnonymousMailboxPullOneV1,
    request_commitment: &[u8; 32],
    stored: StoredPullReplay,
    now: u64,
    retain_until: u64,
) -> Result<(), AnonymousMailboxStoreError> {
    let old = load_pull_replay_totals(transaction)?;
    validate_pull_replay_totals(old, config)?;
    let bytes = match &stored {
        StoredPullReplay::Empty => 0,
        StoredPullReplay::Item {
            sealed_envelope, ..
        } => {
            u64::try_from(sealed_envelope.len()).map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        }
    };
    let new = PullReplayTotals {
        rows: old
            .rows
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?,
        bytes: old
            .bytes
            .checked_add(bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?,
    };
    if validate_pull_replay_totals(new, config).is_err() {
        return Err(AnonymousMailboxStoreError::Busy);
    }
    let (outcome, item_id, commitment, envelope, cursor) = match stored {
        StoredPullReplay::Empty => (0_i64, None, None, None, None),
        StoredPullReplay::Item {
            item_id,
            sealed_commitment,
            sealed_envelope,
            cursor,
        } => (
            1_i64,
            Some(item_id.to_vec()),
            Some(sealed_commitment.to_vec()),
            Some(sealed_envelope),
            Some(cursor),
        ),
    };
    execute_exactly_one(
        transaction,
        transaction
            .execute(
                "INSERT INTO anonymous_mailbox_pull_replays
                 (mailbox_id, request_id, request_commitment, outcome, item_id,
                  sealed_commitment, sealed_envelope, cursor, created_at, retain_until)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    &request.mailbox_id[..],
                    &request.request_id[..],
                    &request_commitment[..],
                    outcome,
                    item_id,
                    commitment,
                    envelope,
                    cursor,
                    as_i64(now)?,
                    as_i64(retain_until)?,
                ],
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
    )?;
    update_pull_replay_totals_exact(transaction, old, new)?;
    if load_pull_replay_totals(transaction)? != new {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

fn load_ticket_issue_meta(
    transaction: &Transaction<'_>,
) -> Result<TicketIssueMeta, AnonymousMailboxStoreError> {
    let (outstanding, window_started_at, issues_in_window): (i64, i64, i64) = transaction
        .query_row(
            "SELECT outstanding_tickets, issuance_window_started_at, issues_in_window
             FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    Ok(TicketIssueMeta {
        outstanding: as_u64(outstanding)?,
        window_started_at: as_u64(window_started_at)?,
        issues_in_window: as_u64(issues_in_window)?,
    })
}

fn update_ticket_issue_meta_exact(
    transaction: &Transaction<'_>,
    old: TicketIssueMeta,
    new: TicketIssueMeta,
) -> Result<(), AnonymousMailboxStoreError> {
    let affected = transaction
        .execute(
            "UPDATE anonymous_mailbox_meta
             SET outstanding_tickets = ?1, issuance_window_started_at = ?2, issues_in_window = ?3
             WHERE singleton = 1 AND outstanding_tickets = ?4
               AND issuance_window_started_at = ?5 AND issues_in_window = ?6",
            params![
                as_i64(new.outstanding)?,
                as_i64(new.window_started_at)?,
                as_i64(new.issues_in_window)?,
                as_i64(old.outstanding)?,
                as_i64(old.window_started_at)?,
                as_i64(old.issues_in_window)?,
            ],
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    execute_exactly_one(transaction, affected)
}

fn verify_ticket_issue_meta(
    transaction: &Transaction<'_>,
    expected: TicketIssueMeta,
) -> Result<(), AnonymousMailboxStoreError> {
    if load_ticket_issue_meta(transaction)? != expected {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

fn load_issued_ticket_by_request(
    transaction: &Transaction<'_>,
    request_id: &[u8; 16],
) -> Result<Option<IssuedTicketRecord>, AnonymousMailboxStoreError> {
    load_issued_ticket(transaction, "request_id = ?1", params![&request_id[..]])
}

fn load_issued_ticket_by_ticket(
    transaction: &Transaction<'_>,
    ticket_id: &[u8; 16],
) -> Result<Option<IssuedTicketRecord>, AnonymousMailboxStoreError> {
    load_issued_ticket(transaction, "ticket_id = ?1", params![&ticket_id[..]])
}

fn load_issued_ticket<P>(
    transaction: &Transaction<'_>,
    predicate: &str,
    params: P,
) -> Result<Option<IssuedTicketRecord>, AnonymousMailboxStoreError>
where
    P: rusqlite::Params,
{
    let sql = format!(
        "SELECT request_id, ticket_id, request_commitment, target_node_id,
                claims_commitment, requested_at, expires_at, proof_nonce,
                ticket_commitment, ticket_signature, consumed_at
         FROM anonymous_mailbox_issued_tickets WHERE {predicate}"
    );
    let row = transaction
        .query_row(&sql, params, |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, Vec<u8>>(7)?,
                row.get::<_, Vec<u8>>(8)?,
                row.get::<_, Vec<u8>>(9)?,
                row.get::<_, Option<i64>>(10)?,
            ))
        })
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let request = AnonymousMailboxTicketIssueV1 {
        version: aeronyx_core::protocol::anonymous_mailbox::ANONYMOUS_MAILBOX_VERSION_V1,
        request_id: fixed::<16>(&row.0)?,
        ticket_id: fixed::<16>(&row.1)?,
        target_node_id: fixed::<32>(&row.3)?,
        lease_claims_commitment: fixed::<32>(&row.4)?,
        issued_at: as_u64(row.5)?,
        expires_at: as_u64(row.6)?,
        proof_nonce: u64::from_le_bytes(fixed::<8>(&row.7)?),
    };
    let ticket = AnonymousMailboxAdmissionTicketV1 {
        version: aeronyx_core::protocol::anonymous_mailbox::ANONYMOUS_MAILBOX_VERSION_V1,
        ticket_id: request.ticket_id,
        target_node_id: request.target_node_id,
        lease_claims_commitment: request.lease_claims_commitment,
        issued_at: request.issued_at,
        expires_at: request.expires_at,
        signature: fixed::<64>(&row.9)?,
    };
    Ok(Some(IssuedTicketRecord {
        request_id: request.request_id,
        request_commitment: fixed::<32>(&row.2)?,
        request,
        ticket,
        ticket_commitment: fixed::<32>(&row.8)?,
        consumed_at: row.10.map(as_u64).transpose()?,
    }))
}

fn validate_issued_ticket(
    record: &IssuedTicketRecord,
    request: &AnonymousMailboxTicketIssueV1,
    target_node_id: &[u8; 32],
) -> Result<(), AnonymousMailboxStoreError> {
    if record.request_id != request.request_id
        || record.request != *request
        || record.ticket.target_node_id != *target_node_id
        || record
            .ticket
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            != record.ticket_commitment
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    record
        .ticket
        .verify_at(
            target_node_id,
            &request.lease_claims_commitment,
            request.issued_at,
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)
}

fn purge_expired_issued_tickets(
    transaction: &Transaction<'_>,
    now: u64,
    limit: u64,
) -> Result<ExpiredIssuedTicketPurge, AnonymousMailboxStoreError> {
    let mut statement = transaction
        .prepare(
            "SELECT request_id, consumed_at FROM anonymous_mailbox_issued_tickets
             WHERE expires_at < ?1 ORDER BY expires_at, request_id LIMIT ?2",
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let rows = statement
        .query_map(params![as_i64(now)?, as_i64(limit)?], |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Option<i64>>(1)?))
        })
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let mut result = ExpiredIssuedTicketPurge::default();
    for row in rows {
        let (request_id, consumed_at) = row.map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let request_id = fixed::<16>(&request_id)?;
        execute_exactly_one(
            transaction,
            transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_issued_tickets WHERE request_id = ?1",
                    params![&request_id[..]],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        result.removed = result
            .removed
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if consumed_at.is_none() {
            result.unconsumed = result
                .unconsumed
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
    }
    Ok(result)
}

fn validate_totals_limits(
    totals: &StoreTotals,
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let maximum_leases =
        u64::try_from(config.max_leases_total).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let maximum_items =
        u64::try_from(config.max_items_total).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if totals.leases > maximum_leases
        || totals.items > maximum_items
        || totals.bytes > config.max_bytes_total
        || (totals.leases == 0 && (totals.items != 0 || totals.bytes != 0))
        || (totals.items == 0) != (totals.bytes == 0)
        || totals.bytes < totals.items
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

fn update_totals_exact(
    transaction: &Transaction<'_>,
    old: StoreTotals,
    new: StoreTotals,
) -> Result<(), AnonymousMailboxStoreError> {
    let affected = transaction
        .execute(
            "UPDATE anonymous_mailbox_meta
             SET total_leases = ?1, total_items = ?2, total_bytes = ?3
             WHERE singleton = 1 AND total_leases = ?4
               AND total_items = ?5 AND total_bytes = ?6",
            params![
                as_i64(new.leases)?,
                as_i64(new.items)?,
                as_i64(new.bytes)?,
                as_i64(old.leases)?,
                as_i64(old.items)?,
                as_i64(old.bytes)?,
            ],
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    execute_exactly_one(transaction, affected)
}

fn verify_totals_exact(
    transaction: &Transaction<'_>,
    expected: StoreTotals,
) -> Result<(), AnonymousMailboxStoreError> {
    if load_totals(transaction)? != expected {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

fn execute_exactly_one(
    _transaction: &Transaction<'_>,
    affected: usize,
) -> Result<(), AnonymousMailboxStoreError> {
    if affected != 1 {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

fn audit_counters(
    transaction: &Transaction<'_>,
    target_node_id: &[u8; 32],
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    #[cfg(test)]
    FULL_AUDIT_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    let totals = load_totals(transaction)?;

    // [ANONYMOUS-MAILBOX-STORE 2026-09-02 by Codex] Recompute every
    // aggregate with checked Rust arithmetic. SQLite SUM overflow and stale
    // counters are both corruption; cleanup must never turn either into an
    // implicit repair.
    let mut leases = HashMap::new();
    let mut lease_rows = transaction
        .prepare(
            "SELECT mailbox_id, current_items, current_bytes, next_sequence
             FROM anonymous_mailbox_leases",
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut lease_query = lease_rows
        .query([])
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut lease_count = 0_u64;
    while let Some(row) = lease_query
        .next()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        let mailbox_id = fixed::<32>(
            &row.get::<_, Vec<u8>>(0)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let audit = LeaseCounterAudit {
            stored_items: as_u64(
                row.get::<_, i64>(1)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            )?,
            stored_bytes: as_u64(
                row.get::<_, i64>(2)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            )?,
            next_sequence: as_u64(
                row.get::<_, i64>(3)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            )?,
            observed_items: 0,
            observed_bytes: 0,
            observed_max_sequence: 0,
        };
        if audit.next_sequence == 0 || leases.insert(mailbox_id, audit).is_some() {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        lease_count = lease_count
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
    }
    drop(lease_query);
    drop(lease_rows);

    let mut item_rows = transaction
        .prepare(
            "SELECT mailbox_id, item_id, sequence, length(sealed_envelope)
             FROM anonymous_mailbox_items",
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut item_query = item_rows
        .query([])
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut item_count = 0_u64;
    let mut byte_count = 0_u64;
    let mut item_keys = Vec::new();
    let maximum_item_bytes = u64::try_from(MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES)
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    while let Some(row) = item_query
        .next()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        let mailbox_id = fixed::<32>(
            &row.get::<_, Vec<u8>>(0)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let item_id = fixed::<16>(
            &row.get::<_, Vec<u8>>(1)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let sequence = as_u64(
            row.get::<_, i64>(2)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let length = as_u64(
            row.get::<_, i64>(3)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        if sequence == 0 || length == 0 || length > maximum_item_bytes {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        let lease = leases
            .get_mut(&mailbox_id)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        lease.observed_items = lease
            .observed_items
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        lease.observed_bytes = lease
            .observed_bytes
            .checked_add(length)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        lease.observed_max_sequence = lease.observed_max_sequence.max(sequence);
        item_count = item_count
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        byte_count = byte_count
            .checked_add(length)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        item_keys.push((mailbox_id, item_id));
    }
    drop(item_query);
    drop(item_rows);

    if totals.leases != lease_count || totals.items != item_count || totals.bytes != byte_count {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    for lease in leases.values() {
        if lease.stored_items != lease.observed_items
            || lease.stored_bytes != lease.observed_bytes
            || lease.next_sequence <= lease.observed_max_sequence
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
    }
    for mailbox_id in leases.keys() {
        load_lease(transaction, mailbox_id)?.ok_or(AnonymousMailboxStoreError::Corrupt)?;
    }
    for (mailbox_id, item_id) in item_keys {
        load_item(transaction, &mailbox_id, &item_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
    }
    audit_pull_replays(transaction, config)?;
    audit_issued_tickets(transaction, target_node_id, config)?;
    Ok(())
}

fn audit_pull_replays(
    transaction: &Transaction<'_>,
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let stored = load_pull_replay_totals(transaction)?;
    validate_pull_replay_totals(stored, config)?;
    let mut statement = transaction
        .prepare(
            "SELECT r.rowid, r.outcome, r.created_at, r.retain_until, l.expires_at,
                    typeof(r.request_commitment), length(r.request_commitment),
                    typeof(r.item_id), length(r.item_id),
                    typeof(r.sealed_commitment), length(r.sealed_commitment),
                    typeof(r.sealed_envelope), length(r.sealed_envelope),
                    typeof(r.cursor), length(r.cursor)
             FROM anonymous_mailbox_pull_replays r
             LEFT JOIN anonymous_mailbox_leases l ON l.mailbox_id = r.mailbox_id",
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut rows = statement
        .query([])
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut observed = PullReplayTotals { rows: 0, bytes: 0 };
    let mut item_rows = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        let rowid = row
            .get::<_, i64>(0)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        let outcome = row
            .get::<_, i64>(1)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        validate_pull_replay_retention(
            row.get::<_, i64>(2)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            row.get::<_, i64>(3)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            row.get::<_, Option<i64>>(4)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let bytes = PullReplayShape::from_row(row, 5)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            .checked_envelope_bytes(outcome)?;
        if outcome == 1 {
            item_rows.push((rowid, bytes));
        }
        observed.rows = observed
            .rows
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        observed.bytes = observed
            .bytes
            .checked_add(bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if observed.rows > stored.rows || observed.bytes > stored.bytes {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
    }
    drop(rows);
    drop(statement);
    if observed != stored {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    // The metadata scan admitted every BLOB length before the second pass.
    // A transaction snapshot keeps these reads tied to the audited rows.
    for (rowid, admitted_bytes) in item_rows {
        let (commitment, envelope): (Vec<u8>, Vec<u8>) = transaction
            .query_row(
                "SELECT sealed_commitment, sealed_envelope
                 FROM anonymous_mailbox_pull_replays WHERE rowid = ?1",
                params![rowid],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if u64::try_from(envelope.len()).map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            != admitted_bytes
            || <[u8; 32]>::from(Sha256::digest(&envelope)) != fixed::<32>(&commitment)?
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
    }
    Ok(())
}

fn audit_issued_tickets(
    transaction: &Transaction<'_>,
    target_node_id: &[u8; 32],
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let meta = load_ticket_issue_meta(transaction)?;
    if meta.outstanding
        > u64::try_from(config.max_outstanding_tickets)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        || meta.issues_in_window
            > u64::try_from(config.max_ticket_issues_per_window)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let mut statement = transaction
        .prepare("SELECT request_id FROM anonymous_mailbox_issued_tickets")
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let rows = statement
        .query_map([], |row| row.get::<_, Vec<u8>>(0))
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let request_ids = rows
        .map(|row| {
            let bytes = row.map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            fixed::<16>(&bytes)
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(statement);
    let mut outstanding = 0_u64;
    for request_id in request_ids {
        let record = load_issued_ticket_by_request(transaction, &request_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        // The configured difficulty may increase after issuance. Requiring
        // one bit here proves a target-bound non-zero work image while durable
        // issuance admission, not startup policy drift, remains authoritative.
        record
            .request
            .verify_for_target(target_node_id, record.request.issued_at, 1)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        validate_issued_ticket(&record, &record.request, target_node_id)?;
        if record.consumed_at.is_none() {
            outstanding = outstanding
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
    }
    if outstanding != meta.outstanding {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

#[cfg(test)]
fn count(transaction: &Transaction<'_>, sql: &str) -> Result<u64, AnonymousMailboxStoreError> {
    let raw: i64 = transaction
        .query_row(sql, [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    as_u64(raw)
}

fn load_lease(
    transaction: &Transaction<'_>,
    mailbox_id: &[u8; 32],
) -> Result<Option<AnonymousMailboxLeaseProjection>, AnonymousMailboxStoreError> {
    let row = transaction
        .query_row(
            "SELECT mailbox_id, ticket_id, claims_commitment, deposit_verifier, read_verifier,
                    max_items, max_bytes, current_items, current_bytes, next_sequence,
                    created_at, expires_at
             FROM anonymous_mailbox_leases WHERE mailbox_id = ?1",
            params![&mailbox_id[..]],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                ))
            },
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let ticket_id = fixed::<16>(&row.1)?;
    let stored_claims = fixed::<32>(&row.2)?;
    let lease = AnonymousMailboxLeaseProjection {
        mailbox_id: fixed::<32>(&row.0)?,
        deposit_verifier: fixed::<32>(&row.3)?,
        read_verifier: fixed::<32>(&row.4)?,
        max_items: u16::try_from(as_u64(row.5)?)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        max_bytes: as_u64(row.6)?,
        current_items: u16::try_from(as_u64(row.7)?)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        current_bytes: as_u64(row.8)?,
        next_sequence: as_u64(row.9)?,
        created_at: as_u64(row.10)?,
        expires_at: as_u64(row.11)?,
    };
    let calculated_claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
        &lease.mailbox_id,
        &lease.deposit_verifier,
        &lease.read_verifier,
        lease.max_items,
        lease.max_bytes,
        lease.created_at,
        lease.expires_at,
    );
    if stored_claims != calculated_claims {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let ticket_link = transaction
        .query_row(
            "SELECT mailbox_id, claims_commitment FROM anonymous_mailbox_tickets
             WHERE ticket_id = ?1",
            params![&ticket_id[..]],
            |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        .ok_or(AnonymousMailboxStoreError::Corrupt)?;
    if fixed::<32>(&ticket_link.0)? != lease.mailbox_id
        || fixed::<32>(&ticket_link.1)? != stored_claims
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(Some(lease))
}

fn load_item(
    transaction: &Transaction<'_>,
    mailbox_id: &[u8; 32],
    item_id: &[u8; 16],
) -> Result<Option<AnonymousMailboxItemProjection>, AnonymousMailboxStoreError> {
    let metadata = transaction
        .query_row(
            "SELECT mailbox_id, item_id, sequence, sealed_commitment,
                    length(sealed_envelope), stored_at, expires_at
             FROM anonymous_mailbox_items WHERE mailbox_id = ?1 AND item_id = ?2",
            params![&mailbox_id[..], &item_id[..]],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            },
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let admitted_length =
        usize::try_from(metadata.4).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if admitted_length == 0 || admitted_length > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let sealed_envelope: Vec<u8> = transaction
        .query_row(
            "SELECT sealed_envelope FROM anonymous_mailbox_items
             WHERE mailbox_id = ?1 AND item_id = ?2",
            params![&mailbox_id[..], &item_id[..]],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    if sealed_envelope.len() != admitted_length {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let sealed_commitment = fixed::<32>(&metadata.3)?;
    let calculated_commitment: [u8; 32] = Sha256::digest(&sealed_envelope).into();
    if sealed_commitment != calculated_commitment {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(Some(AnonymousMailboxItemProjection {
        mailbox_id: fixed::<32>(&metadata.0)?,
        item_id: fixed::<16>(&metadata.1)?,
        sequence: as_u64(metadata.2)?,
        sealed_commitment,
        sealed_envelope,
        stored_at: as_u64(metadata.5)?,
        expires_at: as_u64(metadata.6)?,
    }))
}

fn load_ack_commitment(
    transaction: &Transaction<'_>,
    mailbox_id: &[u8; 32],
    item_id: &[u8; 16],
) -> Result<Option<[u8; 32]>, AnonymousMailboxStoreError> {
    transaction
        .query_row(
            "SELECT sealed_commitment FROM anonymous_mailbox_acks
             WHERE mailbox_id = ?1 AND item_id = ?2",
            params![&mailbox_id[..], &item_id[..]],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        .map(|bytes| fixed::<32>(&bytes))
        .transpose()
}

fn select_expired_items(
    transaction: &Transaction<'_>,
    now: u64,
    limit: u64,
) -> Result<Vec<(Vec<u8>, Vec<u8>, u64)>, AnonymousMailboxStoreError> {
    let mut statement = transaction
        .prepare(
            "SELECT mailbox_id, item_id, length(sealed_envelope)
             FROM anonymous_mailbox_items WHERE expires_at < ?1
             ORDER BY expires_at, mailbox_id, item_id LIMIT ?2",
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let rows = statement
        .query_map(params![as_i64(now)?, as_i64(limit)?], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let mut result = Vec::new();
    for row in rows {
        let (mailbox_id, item_id, bytes) =
            row.map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let bytes = as_u64(bytes)?;
        if bytes == 0 || bytes > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES as u64 {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        result.push((mailbox_id, item_id, bytes));
    }
    Ok(result)
}

fn as_i64(value: u64) -> Result<i64, AnonymousMailboxStoreError> {
    i64::try_from(value).map_err(|_| AnonymousMailboxStoreError::Rejected)
}

fn as_u64(value: i64) -> Result<u64, AnonymousMailboxStoreError> {
    u64::try_from(value).map_err(|_| AnonymousMailboxStoreError::Corrupt)
}

fn fixed<const N: usize>(bytes: &[u8]) -> Result<[u8; N], AnonymousMailboxStoreError> {
    bytes
        .try_into()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)
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

    #[test]
    fn exact_pull_replay_survives_ack_and_restart_without_advancing_to_next_item() {
        let context = TestContext::new();
        let mailbox = [0xC1; 32];
        let lease = context.lease(mailbox, [0xC2; 16], 2, 32, NOW + 1_000);
        let first_put = context.put(mailbox, [0xC3; 16], b"opaque-a", NOW + 900);
        let second_put = context.put(mailbox, [0xC4; 16], b"opaque-b", NOW + 900);
        let pull = context.pull(mailbox, Vec::new(), NOW);

        let store = context.open();
        store.create(&lease, NOW).unwrap();
        store.put(&first_put, NOW).unwrap();
        store.put(&second_put, NOW).unwrap();
        let first = store.pull_one(&pull, NOW).unwrap();
        let AnonymousMailboxPullOutcome::Item(first_item) = &first else {
            panic!("first pull must return an opaque item")
        };
        assert_eq!(first_item.item_id, first_put.item_id);
        let ack = AnonymousMailboxAckV1::new(
            mailbox,
            [0xC5; 16],
            first_item.item_id,
            first_item.sealed_commitment,
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        assert_eq!(
            store.ack(&ack, NOW + 1).unwrap(),
            AnonymousMailboxAckOutcome::Acknowledged
        );
        drop(store);

        let reopened = context.open();
        assert_eq!(reopened.pull_one(&pull, NOW + 600).unwrap(), first);
        let changed_same_id = context.pull(mailbox, Vec::new(), NOW + 1);
        assert_eq!(
            reopened.pull_one(&changed_same_id, NOW + 1),
            Err(AnonymousMailboxStoreError::Rejected)
        );
        let fresh_id = AnonymousMailboxPullOneV1::new(
            mailbox,
            [0xC6; 16],
            Vec::new(),
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        assert!(matches!(
            reopened.pull_one(&fresh_id, NOW + 1).unwrap(),
            AnonymousMailboxPullOutcome::Item(ref item) if item.item_id == second_put.item_id
        ));
    }

    #[test]
    fn exact_empty_pull_replay_survives_later_put_and_restart() {
        let context = TestContext::new();
        let mailbox = [0xD2; 32];
        let lease = context.lease(mailbox, [0xD3; 16], 1, 32, NOW + 1_000);
        let pull = context.pull(mailbox, Vec::new(), NOW);
        let put = context.put(mailbox, [0xD4; 16], b"opaque-later", NOW + 900);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        assert_eq!(
            store.pull_one(&pull, NOW).unwrap(),
            AnonymousMailboxPullOutcome::Empty
        );
        store.put(&put, NOW + 1).unwrap();
        drop(store);

        let reopened = context.open();
        assert_eq!(
            reopened.pull_one(&pull, NOW + 600).unwrap(),
            AnonymousMailboxPullOutcome::Empty
        );
        let fresh = AnonymousMailboxPullOneV1::new(
            mailbox,
            [0xD5; 16],
            Vec::new(),
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        assert!(matches!(
            reopened.pull_one(&fresh, NOW + 1).unwrap(),
            AnonymousMailboxPullOutcome::Item(ref item) if item.item_id == put.item_id
        ));
    }

    #[test]
    fn pull_replay_retention_boundary_fails_closed_before_cleanup() {
        let context = TestContext::new();
        let item_mailbox = [0xE1; 32];
        let empty_mailbox = [0xE2; 32];
        let expires_at = NOW + 1_000;
        let item_pull = context.pull(item_mailbox, Vec::new(), NOW);
        let empty_pull = context.pull(empty_mailbox, Vec::new(), NOW);
        let store = context.open();
        store
            .create(
                &context.lease(item_mailbox, [0xE3; 16], 1, 32, expires_at),
                NOW,
            )
            .unwrap();
        store
            .create(
                &context.lease(empty_mailbox, [0xE4; 16], 1, 32, expires_at),
                NOW,
            )
            .unwrap();
        let put = context.put(item_mailbox, [0xE5; 16], b"opaque", expires_at);
        store.put(&put, NOW).unwrap();
        let first = store.pull_one(&item_pull, NOW).unwrap();
        assert_eq!(
            store.pull_one(&empty_pull, NOW).unwrap(),
            AnonymousMailboxPullOutcome::Empty
        );
        let AnonymousMailboxPullOutcome::Item(item) = &first else {
            panic!("item pull must return an opaque item")
        };
        let ack = AnonymousMailboxAckV1::new(
            item_mailbox,
            [0xE6; 16],
            item.item_id,
            item.sealed_commitment,
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        assert_eq!(
            store.ack(&ack, NOW + 1).unwrap(),
            AnonymousMailboxAckOutcome::Acknowledged
        );
        drop(store);

        let reopened = context.open();
        assert_eq!(reopened.pull_one(&item_pull, expires_at).unwrap(), first);
        assert_eq!(
            reopened.pull_one(&empty_pull, expires_at).unwrap(),
            AnonymousMailboxPullOutcome::Empty
        );
        for pull in [&item_pull, &empty_pull] {
            assert_eq!(
                reopened.pull_one(pull, expires_at + 1),
                Err(AnonymousMailboxStoreError::Rejected),
                "retention must end even before cleanup removes the row"
            );
        }
        let changed_same_id = context.pull(item_mailbox, Vec::new(), NOW + 1);
        assert_eq!(
            reopened.pull_one(&changed_same_id, expires_at),
            Err(AnonymousMailboxStoreError::Rejected)
        );
    }

    #[test]
    fn pull_replay_budget_and_cleanup_are_bounded() {
        let context = TestContext::new();
        let mailbox = [0xC7; 32];
        let lease = context.lease(mailbox, [0xC8; 16], 2, 32, NOW + 1_000);
        let mut store = context.open();
        store.config.max_items_total = 2;
        store.config.cleanup_batch_size = 1;
        store.create(&lease, NOW).unwrap();
        for request_id in [[0xC9; 16], [0xCA; 16]] {
            let request = AnonymousMailboxPullOneV1::new(
                mailbox,
                request_id,
                Vec::new(),
                NOW,
                &context.reader,
            )
            .unwrap();
            assert_eq!(
                store.pull_one(&request, NOW).unwrap(),
                AnonymousMailboxPullOutcome::Empty
            );
        }
        let over_cap =
            AnonymousMailboxPullOneV1::new(mailbox, [0xCB; 16], Vec::new(), NOW, &context.reader)
                .unwrap();
        assert_eq!(
            store.pull_one(&over_cap, NOW),
            Err(AnonymousMailboxStoreError::Busy)
        );
        let report = store.cleanup(NOW + PULL_REPLAY_RETENTION_SECS + 1).unwrap();
        assert_eq!(report.pull_replays_removed, 1);
        let connection = store.connection.lock();
        assert_eq!(
            connection
                .query_row(
                    "SELECT pull_replay_rows FROM anonymous_mailbox_meta WHERE singleton = 1",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn pull_replay_ciphertext_corruption_fails_closed_on_restart() {
        let context = TestContext::new();
        let mailbox = [0xCC; 32];
        let lease = context.lease(mailbox, [0xCD; 16], 1, 32, NOW + 1_000);
        let put = context.put(mailbox, [0xCE; 16], b"opaque", NOW + 900);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        store.put(&put, NOW).unwrap();
        store
            .pull_one(&context.pull(mailbox, Vec::new(), NOW), NOW)
            .unwrap();
        drop(store);

        let connection = Connection::open(&context.config.db_path).unwrap();
        connection
            .execute(
                "UPDATE anonymous_mailbox_pull_replays SET sealed_envelope = ?1",
                params![b"tampered-opaque".as_slice()],
            )
            .unwrap();
        drop(connection);
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                context.config.clone(),
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::Corrupt)
        ));
    }

    #[test]
    fn oversized_replay_blobs_fail_closed_on_lookup_and_startup() {
        for oversize_cursor in [false, true] {
            let context = TestContext::new();
            let mailbox = [0xE7; 32];
            let pull = context.pull(mailbox, Vec::new(), NOW);
            let store = context.open();
            store
                .create(&context.lease(mailbox, [0xE8; 16], 1, 32, NOW + 1_000), NOW)
                .unwrap();
            store
                .put(&context.put(mailbox, [0xE9; 16], b"opaque", NOW + 900), NOW)
                .unwrap();
            assert!(matches!(
                store.pull_one(&pull, NOW).unwrap(),
                AnonymousMailboxPullOutcome::Item(_)
            ));

            let connection = Connection::open(&context.config.db_path).unwrap();
            // Simulate a physically valid but corrupt older V3 file. Current
            // fresh-schema CHECK constraints reject this normal write.
            connection
                .execute_batch("PRAGMA ignore_check_constraints = ON;")
                .unwrap();
            if oversize_cursor {
                connection
                    .execute(
                        "UPDATE anonymous_mailbox_pull_replays SET cursor = zeroblob(?1)",
                        params![i64::try_from(CURSOR_BYTES + 1).unwrap()],
                    )
                    .unwrap();
            } else {
                connection
                    .execute(
                        "UPDATE anonymous_mailbox_pull_replays
                         SET sealed_envelope = zeroblob(?1)",
                        params![i64::try_from(MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES + 1)
                            .unwrap()],
                    )
                    .unwrap();
            }
            drop(connection);
            assert_eq!(
                store.pull_one(&pull, NOW + 1),
                Err(AnonymousMailboxStoreError::Corrupt)
            );
            drop(store);
            assert!(matches!(
                SqliteAnonymousMailboxStore::open(
                    context.config.clone(),
                    context.target.public_key_bytes(),
                    CURSOR_SECRET,
                ),
                Err(AnonymousMailboxStoreError::Corrupt)
            ));
        }
    }

    #[test]
    fn disabled_open_has_no_filesystem_side_effect() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("absent").join("mailbox.sqlite");
        let config = AnonymousMailboxStoreConfig {
            db_path: path.display().to_string(),
            ..Default::default()
        };
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(config, [1; 32], CURSOR_SECRET),
            Err(AnonymousMailboxStoreError::Disabled)
        ));
        assert!(!path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn new_database_dirent_is_synchronized_before_open_returns() {
        let context = TestContext::new();
        let before = PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get);
        let store = context.open();
        let after = PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get);
        assert_eq!(after, before + 1);
        drop(store);
    }

    #[cfg(unix)]
    #[test]
    fn parent_sync_failure_is_fail_closed_and_retry_resynchronizes() {
        let context = TestContext::new();
        FORCE_PARENT_SYNC_FAILURE.with(|forced| forced.set(true));
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                context.config.clone(),
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::Unavailable)
        ));
        FORCE_PARENT_SYNC_FAILURE.with(|forced| forced.set(false));
        let before = PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get);
        let store = context.open();
        assert_eq!(
            PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get),
            before + 1
        );
        drop(store);
    }

    #[test]
    fn create_consumes_ticket_once_and_exact_retry_survives_restart() {
        let context = TestContext::new();
        let mailbox = [0x11; 32];
        let request = context.lease(mailbox, [0x21; 16], 2, 32, NOW + 1_000);
        let store = context.open();
        assert!(matches!(
            store.create(&request, NOW).unwrap(),
            AnonymousMailboxCreateOutcome::Created(_)
        ));
        assert!(matches!(
            store.create(&request, NOW + 1).unwrap(),
            AnonymousMailboxCreateOutcome::Existing(_)
        ));
        drop(store);
        let reopened = context.open();
        assert!(matches!(
            reopened.create(&request, NOW + 301).unwrap(),
            AnonymousMailboxCreateOutcome::Existing(_)
        ));

        let conflicting = context.lease([0x12; 32], [0x21; 16], 2, 32, NOW + 1_000);
        assert_eq!(
            reopened.create(&conflicting, NOW + 301).unwrap(),
            AnonymousMailboxCreateOutcome::Conflict
        );
    }

    #[test]
    fn ticket_issue_replays_exactly_conflicts_and_survives_restart() {
        let context = TestContext::new();
        let request =
            context.ticket_issue([0x81; 16], [0x82; 16], [0x83; 32], NOW + 1_000, NOW + 300);
        let store = context.open_with_ticket_issuer();
        let issued = match store.issue_ticket(&request, NOW).unwrap() {
            AnonymousMailboxTicketIssueOutcome::Issued(ticket) => ticket,
            other => panic!("unexpected issue result: {other:?}"),
        };
        assert!(matches!(
            store.issue_ticket(&request, NOW + 1).unwrap(),
            AnonymousMailboxTicketIssueOutcome::Existing(ticket) if ticket == issued
        ));
        let conflict = context.ticket_issue(
            request.request_id,
            [0x84; 16],
            [0x83; 32],
            NOW + 1_000,
            NOW + 300,
        );
        assert_eq!(
            store.issue_ticket(&conflict, NOW + 1).unwrap(),
            AnonymousMailboxTicketIssueOutcome::Conflict
        );
        drop(store);
        let reopened = context.open_with_ticket_issuer();
        assert!(matches!(
            reopened.issue_ticket(&request, NOW + 2).unwrap(),
            AnonymousMailboxTicketIssueOutcome::Existing(ticket) if ticket == issued
        ));
    }

    #[test]
    fn issued_ticket_is_consumed_once_and_capacity_replay_precedes_limits() {
        let mut context = TestContext::new();
        context.config.max_outstanding_tickets = 2;
        context.config.max_ticket_issues_per_window = 1;
        let first =
            context.ticket_issue([0x91; 16], [0x92; 16], [0x93; 32], NOW + 1_000, NOW + 300);
        let second =
            context.ticket_issue([0x94; 16], [0x95; 16], [0x96; 32], NOW + 1_000, NOW + 300);
        let store = context.open_with_ticket_issuer();
        let ticket = match store.issue_ticket(&first, NOW).unwrap() {
            AnonymousMailboxTicketIssueOutcome::Issued(ticket) => ticket,
            other => panic!("unexpected issue result: {other:?}"),
        };
        assert_eq!(
            store.issue_ticket(&second, NOW + 1).unwrap(),
            AnonymousMailboxTicketIssueOutcome::AtCapacity
        );
        assert!(matches!(
            store.issue_ticket(&first, NOW + 1).unwrap(),
            AnonymousMailboxTicketIssueOutcome::Existing(existing) if existing == ticket
        ));
        let lease = context.lease_for_issued_ticket([0x93; 32], NOW + 1_000, ticket.clone());
        assert!(matches!(
            store.create(&lease, NOW + 2).unwrap(),
            AnonymousMailboxCreateOutcome::Created(_)
        ));
        assert!(matches!(
            store.create(&lease, NOW + 3).unwrap(),
            AnonymousMailboxCreateOutcome::Existing(_)
        ));
        assert!(matches!(
            store.issue_ticket(&first, NOW + 3).unwrap(),
            AnonymousMailboxTicketIssueOutcome::Existing(existing) if existing == ticket
        ));
        assert!(matches!(
            store.issue_ticket(&second, NOW + 60).unwrap(),
            AnonymousMailboxTicketIssueOutcome::Issued(_)
        ));
    }

    #[test]
    fn expired_issued_ticket_is_not_served_and_cleanup_is_bounded() {
        let context = TestContext::new();
        let request =
            context.ticket_issue([0xa1; 16], [0xa2; 16], [0xa3; 32], NOW + 1_000, NOW + 10);
        let store = context.open_with_ticket_issuer();
        assert!(matches!(
            store.issue_ticket(&request, NOW).unwrap(),
            AnonymousMailboxTicketIssueOutcome::Issued(_)
        ));
        assert!(matches!(
            store.issue_ticket(&request, NOW + 11),
            Err(AnonymousMailboxStoreError::Rejected)
        ));
        let report = store.cleanup(NOW + 11).unwrap();
        assert_eq!(report.issued_tickets_removed, 1);
    }

    #[test]
    fn invalid_ticket_issue_proof_is_rejected_without_persistence() {
        let context = TestContext::new();
        let valid =
            context.ticket_issue([0xb1; 16], [0xb2; 16], [0xb3; 32], NOW + 1_000, NOW + 300);
        let invalid = (valid.proof_nonce.saturating_add(1)..u64::MAX)
            .find_map(|nonce| {
                let candidate = AnonymousMailboxTicketIssueV1::new(
                    valid.request_id,
                    valid.ticket_id,
                    valid.target_node_id,
                    valid.lease_claims_commitment,
                    valid.issued_at,
                    valid.expires_at,
                    nonce,
                )
                .unwrap();
                (candidate.proof_digest().unwrap()[0] & 0x80 != 0).then_some(candidate)
            })
            .expect("invalid one-bit proof");
        let store = context.open_with_ticket_issuer();
        assert!(matches!(
            store.issue_ticket(&invalid, NOW),
            Err(AnonymousMailboxStoreError::Rejected)
        ));
        let connection = store.connection.lock();
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM anonymous_mailbox_issued_tickets",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn v1_store_migrates_additively_and_reopens_with_ticket_journal() {
        let context = TestContext::new();
        drop(context.open());
        let connection = Connection::open(&context.config.db_path).unwrap();
        connection
            .execute_batch(
                "DROP INDEX anonymous_mailbox_pull_replay_expiry;
                 DROP TABLE anonymous_mailbox_pull_replays;
                 DROP INDEX anonymous_mailbox_issued_ticket_expiry;
                 DROP TABLE anonymous_mailbox_issued_tickets;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_rows;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_bytes;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN outstanding_tickets;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN issuance_window_started_at;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN issues_in_window;
                 UPDATE anonymous_mailbox_meta SET schema_version = 1;
                 PRAGMA user_version = 1;",
            )
            .unwrap();
        drop(connection);

        let reopened = context.open_with_ticket_issuer();
        let connection = reopened.connection.lock();
        assert_eq!(schema_user_version(&connection), 3);
        assert_eq!(
            connection
                .query_row(
                    "SELECT schema_version FROM anonymous_mailbox_meta WHERE singleton = 1",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            3
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'table' AND name = 'anonymous_mailbox_issued_tickets'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn v2_store_migrates_additively_without_rewriting_custody_rows() {
        let context = TestContext::new();
        let mailbox = [0xCF; 32];
        let lease = context.lease(mailbox, [0xD0; 16], 1, 32, NOW + 1_000);
        let put = context.put(mailbox, [0xD1; 16], b"opaque-v2", NOW + 900);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        store.put(&put, NOW).unwrap();
        drop(store);

        let connection = Connection::open(&context.config.db_path).unwrap();
        connection
            .execute_batch(
                "DROP INDEX anonymous_mailbox_pull_replay_expiry;
                 DROP TABLE anonymous_mailbox_pull_replays;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_rows;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_bytes;
                 UPDATE anonymous_mailbox_meta SET schema_version = 2 WHERE singleton = 1;
                 PRAGMA user_version = 2;",
            )
            .unwrap();
        drop(connection);

        let reopened = context.open();
        let connection = reopened.connection.lock();
        assert_eq!(schema_user_version(&connection), 3);
        assert_eq!(
            connection
                .query_row(
                    "SELECT sealed_envelope FROM anonymous_mailbox_items
                     WHERE mailbox_id = ?1 AND item_id = ?2",
                    params![&mailbox[..], &put.item_id[..]],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .unwrap(),
            b"opaque-v2"
        );
    }

    #[test]
    fn put_exact_retry_precedes_quota_and_changed_bytes_conflict() {
        let context = TestContext::new();
        let mailbox = [0x31; 32];
        let lease = context.lease(mailbox, [0x32; 16], 1, 4, NOW + 1_000);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        let put = context.put(mailbox, [0x33; 16], b"four", NOW + 100);
        assert!(matches!(
            store.put(&put, NOW).unwrap(),
            AnonymousMailboxPutOutcome::Stored(_)
        ));
        assert!(matches!(
            store.put(&put, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::Existing(_)
        ));
        let changed = context.put(mailbox, [0x33; 16], b"diff", NOW + 100);
        assert_eq!(
            store.put(&changed, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::Conflict
        );
        let over = context.put(mailbox, [0x34; 16], b"x", NOW + 100);
        assert_eq!(
            store.put(&over, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::AtCapacity
        );
    }

    #[test]
    fn pull_is_single_padded_snapshot_and_cursor_is_restart_stable_and_authenticated() {
        let context = TestContext::new();
        let mailbox = [0x41; 32];
        let lease = context.lease(mailbox, [0x42; 16], 3, 64, NOW + 1_000);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        store
            .put(&context.put(mailbox, [1; 16], b"one", NOW + 100), NOW)
            .unwrap();
        store
            .put(&context.put(mailbox, [2; 16], b"two", NOW + 100), NOW)
            .unwrap();
        let first = match store
            .pull_one(&context.pull(mailbox, Vec::new(), NOW), NOW)
            .unwrap()
        {
            AnonymousMailboxPullOutcome::Item(item) => item,
            other => panic!("unexpected pull outcome: {other:?}"),
        };
        assert_eq!(first.item_id, [1; 16]);
        assert_eq!(first.sealed_length, 3);
        assert_eq!(
            first.padded_sealed_envelope.len(),
            MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
        );
        assert_eq!(&first.padded_sealed_envelope[..3], b"one");
        assert_eq!(first.cursor.len(), CURSOR_BYTES);

        let mut tampered = first.cursor.clone();
        tampered[10] ^= 1;
        assert_eq!(
            store.pull_one(&context.pull(mailbox, tampered, NOW + 1), NOW + 1),
            Err(AnonymousMailboxStoreError::Rejected)
        );
        drop(store);

        let reopened = context.open();
        let second = reopened
            .pull_one(
                &context.pull(mailbox, first.cursor.clone(), NOW + 1),
                NOW + 1,
            )
            .unwrap();
        match second {
            AnonymousMailboxPullOutcome::Item(item) => assert_eq!(item.item_id, [2; 16]),
            other => panic!("unexpected pull outcome: {other:?}"),
        }
        drop(reopened);

        let wrong_key = SqliteAnonymousMailboxStore::open(
            context.config.clone(),
            context.target.public_key_bytes(),
            [0x5A; 32],
        )
        .unwrap();
        assert_eq!(
            wrong_key.pull_one(&context.pull(mailbox, first.cursor, NOW + 1), NOW + 1,),
            Err(AnonymousMailboxStoreError::Rejected)
        );
    }

    #[test]
    fn expired_items_are_never_pulled() {
        let context = TestContext::new();
        let mailbox = [0x45; 32];
        let lease = context.lease(mailbox, [0x46; 16], 2, 64, NOW + 1_000);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        store
            .put(&context.put(mailbox, [0x47; 16], b"opaque", NOW + 2), NOW)
            .unwrap();
        assert_eq!(
            store
                .pull_one(&context.pull(mailbox, Vec::new(), NOW + 3), NOW + 3)
                .unwrap(),
            AnonymousMailboxPullOutcome::Empty
        );
    }

    #[test]
    fn ack_binds_commitment_and_exact_tombstone_is_restart_idempotent() {
        let context = TestContext::new();
        let mailbox = [0x51; 32];
        let lease = context.lease(mailbox, [0x52; 16], 2, 64, NOW + 200_000);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        let put = context.put(mailbox, [0x53; 16], b"opaque", NOW + 100_000);
        store.put(&put, NOW).unwrap();
        let ack = AnonymousMailboxAckV1::new(
            mailbox,
            [0x54; 16],
            put.item_id,
            put.sealed_commitment(),
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        assert_eq!(
            store.ack(&ack, NOW + 1).unwrap(),
            AnonymousMailboxAckOutcome::Acknowledged
        );
        drop(store);
        let reopened = context.open();
        let boundary = reopened.cleanup(put.expires_at).unwrap();
        assert_eq!(boundary.acknowledgements_removed, 0);
        assert_eq!(
            reopened.ack(&ack, put.expires_at + 1).unwrap(),
            AnonymousMailboxAckOutcome::AlreadyAcknowledged
        );
        let wrong = AnonymousMailboxAckV1::new(
            mailbox,
            [0x55; 16],
            put.item_id,
            [0xEE; 32],
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        assert_eq!(
            reopened.ack(&wrong, put.expires_at + 1).unwrap(),
            AnonymousMailboxAckOutcome::Conflict
        );
        let retain_until: i64 = reopened
            .connection
            .lock()
            .query_row(
                "SELECT retain_until FROM anonymous_mailbox_acks
                 WHERE mailbox_id = ?1 AND item_id = ?2",
                params![&mailbox[..], &put.item_id[..]],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(as_u64(retain_until).unwrap(), put.expires_at);
        let cleanup = reopened.cleanup(put.expires_at + 1).unwrap();
        assert_eq!(cleanup.acknowledgements_removed, 1);
        assert_eq!(cleanup.leases_removed, 0);
    }

    #[test]
    fn lease_and_ticket_projection_tampering_fails_closed_without_mutation() {
        let context = TestContext::new();
        let mailbox = [0x57; 32];
        let request = context.lease(mailbox, [0x58; 16], 2, 64, NOW + 1_000);
        let store = context.open();
        store.create(&request, NOW).unwrap();
        let baseline: (i64, i64, i64) = store
            .connection
            .lock()
            .query_row(
                "SELECT total_leases, total_items, total_bytes FROM anonymous_mailbox_meta",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        for mutation in [
            "UPDATE anonymous_mailbox_leases SET deposit_verifier = zeroblob(32)",
            "UPDATE anonymous_mailbox_leases SET read_verifier = zeroblob(32)",
            "UPDATE anonymous_mailbox_leases SET max_items = max_items + 1",
            "UPDATE anonymous_mailbox_leases SET max_bytes = max_bytes + 1",
            "UPDATE anonymous_mailbox_leases SET expires_at = expires_at + 1",
            "UPDATE anonymous_mailbox_leases SET claims_commitment = zeroblob(32)",
            "UPDATE anonymous_mailbox_leases SET ticket_id = zeroblob(16)",
            "UPDATE anonymous_mailbox_tickets SET mailbox_id = zeroblob(32)",
            "UPDATE anonymous_mailbox_tickets SET claims_commitment = zeroblob(32)",
        ] {
            store.connection.lock().execute(mutation, []).unwrap();
            assert_eq!(
                store.create(&request, NOW + 1),
                Err(AnonymousMailboxStoreError::Corrupt)
            );
            let claims = request.claims_commitment();
            store
                .connection
                .lock()
                .execute(
                    "UPDATE anonymous_mailbox_leases
                     SET ticket_id = ?1, claims_commitment = ?2, deposit_verifier = ?3,
                         read_verifier = ?4, max_items = ?5, max_bytes = ?6,
                         created_at = ?7, expires_at = ?8",
                    params![
                        &request.admission.ticket_id[..],
                        &claims[..],
                        &request.deposit_verifier[..],
                        &request.read_verifier[..],
                        i64::from(request.max_items),
                        as_i64(request.max_bytes).unwrap(),
                        as_i64(request.issued_at).unwrap(),
                        as_i64(request.expires_at).unwrap(),
                    ],
                )
                .unwrap();
            store
                .connection
                .lock()
                .execute(
                    "UPDATE anonymous_mailbox_tickets
                     SET mailbox_id = ?1, claims_commitment = ?2
                     WHERE ticket_id = ?3",
                    params![
                        &request.mailbox_id[..],
                        &claims[..],
                        &request.admission.ticket_id[..],
                    ],
                )
                .unwrap();
            let observed: (i64, i64, i64) = store
                .connection
                .lock()
                .query_row(
                    "SELECT total_leases, total_items, total_bytes FROM anonymous_mailbox_meta",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
            assert_eq!(observed, baseline);
        }
    }

    #[test]
    fn sealed_item_tampering_is_rejected_by_hot_read_and_startup_audit() {
        let context = TestContext::new();
        let mailbox = [0x59; 32];
        let store = context.open();
        store
            .create(&context.lease(mailbox, [0x5A; 16], 2, 64, NOW + 1_000), NOW)
            .unwrap();
        let put = context.put(mailbox, [0x5B; 16], b"opaque", NOW + 100);
        store.put(&put, NOW).unwrap();
        store
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_items SET sealed_envelope = ?1",
                params![&b"change"[..]],
            )
            .unwrap();
        assert_eq!(
            store.pull_one(&context.pull(mailbox, Vec::new(), NOW + 1), NOW + 1),
            Err(AnonymousMailboxStoreError::Corrupt)
        );
        assert_eq!(
            store.put(&put, NOW + 1),
            Err(AnonymousMailboxStoreError::Corrupt)
        );
        store
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_items
                 SET sealed_envelope = ?1, sealed_commitment = zeroblob(32)",
                params![&b"opaque"[..]],
            )
            .unwrap();
        assert_eq!(
            store.pull_one(&context.pull(mailbox, Vec::new(), NOW + 1), NOW + 1),
            Err(AnonymousMailboxStoreError::Corrupt)
        );
        drop(store);
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                context.config.clone(),
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::Corrupt)
        ));
    }

    #[test]
    fn global_capacity_is_atomic_across_connections() {
        let mut context = TestContext::new();
        context.config.max_leases_total = 1;
        let first = Arc::new(context.open());
        let second = Arc::new(context.open());
        let request_a = context.lease([0x61; 32], [0x62; 16], 1, 8, NOW + 1_000);
        let request_b = context.lease([0x63; 32], [0x64; 16], 1, 8, NOW + 1_000);
        let barrier = Arc::new(Barrier::new(3));
        let run = |store: Arc<SqliteAnonymousMailboxStore>,
                   request: AnonymousMailboxLeaseCreateV1,
                   barrier: Arc<Barrier>| {
            std::thread::spawn(move || {
                barrier.wait();
                store.create(&request, NOW).unwrap()
            })
        };
        let a = run(Arc::clone(&first), request_a, Arc::clone(&barrier));
        let b = run(Arc::clone(&second), request_b, Arc::clone(&barrier));
        barrier.wait();
        let outcomes = [a.join().unwrap(), b.join().unwrap()];
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, AnonymousMailboxCreateOutcome::Created(_)))
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, AnonymousMailboxCreateOutcome::AtCapacity))
                .count(),
            1
        );
    }

    #[test]
    fn in_flight_gate_is_shared_by_all_repository_methods() {
        let mut context = TestContext::new();
        context.config.max_in_flight = 1;
        let store = context.open();
        let _held = store.acquire().unwrap();
        assert_eq!(store.cleanup(NOW), Err(AnonymousMailboxStoreError::Busy));
    }

    #[test]
    fn mutation_hot_paths_do_not_invoke_full_database_audit() {
        let mut context = TestContext::new();
        context.config.max_leases_total = 64;
        let store = context.open();
        let audits_after_startup = FULL_AUDIT_CALLS.with(std::cell::Cell::get);
        for discriminator in 1_u8..=48 {
            let lease = context.lease([discriminator; 32], [discriminator; 16], 2, 64, NOW + 1_000);
            assert!(matches!(
                store.create(&lease, NOW).unwrap(),
                AnonymousMailboxCreateOutcome::Created(_)
            ));
        }
        let mailbox = [48; 32];
        let put = context.put(mailbox, [0xF0; 16], b"opaque", NOW + 100);
        store.put(&put, NOW).unwrap();
        let ack = AnonymousMailboxAckV1::new(
            mailbox,
            [0xF1; 16],
            put.item_id,
            put.sealed_commitment(),
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        store.ack(&ack, NOW + 1).unwrap();
        store.cleanup(NOW + 2).unwrap();
        assert_eq!(
            FULL_AUDIT_CALLS.with(std::cell::Cell::get),
            audits_after_startup
        );
    }

    #[test]
    fn global_item_row_cap_preserves_exact_put_retry_priority() {
        let mut context = TestContext::new();
        context.config.max_items_total = usize::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
        context.config.max_bytes_total = 2 * 1024 * 1024;
        let store = context.open();
        let full_mailbox = [0xA1; 32];
        let fallback_mailbox = [0xA2; 32];
        store
            .create(
                &context.lease(
                    full_mailbox,
                    [0xA3; 16],
                    MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE,
                    2 * 1024,
                    NOW + 1_000,
                ),
                NOW,
            )
            .unwrap();
        store
            .create(
                &context.lease(fallback_mailbox, [0xA4; 16], 2, 64, NOW + 1_000),
                NOW,
            )
            .unwrap();
        let exact = context.put(full_mailbox, [0xA5; 16], b"x", NOW + 100);
        store.put(&exact, NOW).unwrap();

        // One transaction installs a commitment-valid large fixture without
        // turning this resource-bound regression into 1,023 FULL fsyncs.
        let commitment: [u8; 32] = Sha256::digest(b"x").into();
        let mut connection = store.connection.lock();
        let transaction = connection.transaction().unwrap();
        for sequence in 2_u64..=u64::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE) {
            let mut item_id = [0_u8; 16];
            item_id[..8].copy_from_slice(&sequence.to_le_bytes());
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_items
                     (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                      sealed_envelope, stored_at, expires_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        &full_mailbox[..],
                        &item_id[..],
                        as_i64(sequence).unwrap(),
                        &[0_u8; 32][..],
                        &commitment[..],
                        &b"x"[..],
                        as_i64(NOW).unwrap(),
                        as_i64(NOW + 100).unwrap(),
                    ],
                )
                .unwrap();
        }
        let item_cap = u64::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
        transaction
            .execute(
                "UPDATE anonymous_mailbox_leases
                 SET current_items = ?1, current_bytes = ?1, next_sequence = ?2
                 WHERE mailbox_id = ?3",
                params![
                    as_i64(item_cap).unwrap(),
                    as_i64(item_cap + 1).unwrap(),
                    &full_mailbox[..],
                ],
            )
            .unwrap();
        transaction
            .execute(
                "UPDATE anonymous_mailbox_meta SET total_items = ?1, total_bytes = ?1",
                params![as_i64(item_cap).unwrap()],
            )
            .unwrap();
        transaction.commit().unwrap();
        drop(connection);

        assert!(matches!(
            store.put(&exact, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::Existing(_)
        ));
        let blocked = context.put(fallback_mailbox, [0xA6; 16], b"y", NOW + 100);
        assert_eq!(
            store.put(&blocked, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::AtCapacity
        );
        let blocked_request = blocked.request_commitment().unwrap();
        let blocked_sealed = blocked.sealed_commitment();
        let mut connection = store.connection.lock();
        let transaction = connection.transaction().unwrap();
        transaction
            .execute(
                "INSERT INTO anonymous_mailbox_items
                 (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                  sealed_envelope, stored_at, expires_at)
                 VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7)",
                params![
                    &fallback_mailbox[..],
                    &blocked.item_id[..],
                    &blocked_request[..],
                    &blocked_sealed[..],
                    &blocked.sealed_envelope,
                    as_i64(NOW).unwrap(),
                    as_i64(blocked.expires_at).unwrap(),
                ],
            )
            .unwrap();
        transaction
            .execute(
                "UPDATE anonymous_mailbox_leases
                 SET current_items = 1, current_bytes = 1, next_sequence = 2
                 WHERE mailbox_id = ?1",
                params![&fallback_mailbox[..]],
            )
            .unwrap();
        transaction
            .execute(
                "UPDATE anonymous_mailbox_meta
                 SET total_items = total_items + 1, total_bytes = total_bytes + 1",
                [],
            )
            .unwrap();
        transaction.commit().unwrap();
        drop(connection);
        drop(store);
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                context.config.clone(),
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::Corrupt)
        ));
    }

    #[test]
    fn full_owner_storage_reopens_and_releases_capacity_without_counter_drift() {
        let mut context = TestContext::new();
        let item_cap = u64::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
        context.config.max_items_total = usize::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
        context.config.max_bytes_total = item_cap;
        context.config.cleanup_batch_size = 1;

        let mailbox = [0xB1; 32];
        let lease = context.lease(
            mailbox,
            [0xB2; 16],
            MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE,
            item_cap,
            NOW + 1_000,
        );
        let store = context.open();
        assert!(matches!(
            store.create(&lease, NOW).unwrap(),
            AnonymousMailboxCreateOutcome::Created(_)
        ));
        let exact = context.put(mailbox, [0xB3; 16], b"x", NOW + 100);
        assert!(matches!(
            store.put(&exact, NOW).unwrap(),
            AnonymousMailboxPutOutcome::Stored(_)
        ));

        // Production Put establishes the lease and first opaque row. Populate
        // the remaining one-byte rows in one transaction so this bounded
        // restart regression does not perform 1,023 FULL fsyncs.
        let sealed_commitment: [u8; 32] = Sha256::digest(b"x").into();
        let mut connection = store.connection.lock();
        let transaction = connection.transaction().unwrap();
        for sequence in 2..=item_cap {
            let mut item_id = [0_u8; 16];
            item_id[..8].copy_from_slice(&sequence.to_le_bytes());
            let expires_at = if sequence == 2 { NOW + 2 } else { NOW + 100 };
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_items
                     (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                      sealed_envelope, stored_at, expires_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        &mailbox[..],
                        &item_id[..],
                        as_i64(sequence).unwrap(),
                        &[0_u8; 32][..],
                        &sealed_commitment[..],
                        &b"x"[..],
                        as_i64(NOW).unwrap(),
                        as_i64(expires_at).unwrap(),
                    ],
                )
                .unwrap();
        }
        transaction
            .execute(
                "UPDATE anonymous_mailbox_leases
                 SET current_items = ?1, current_bytes = ?1, next_sequence = ?2
                 WHERE mailbox_id = ?3",
                params![
                    as_i64(item_cap).unwrap(),
                    as_i64(item_cap + 1).unwrap(),
                    &mailbox[..],
                ],
            )
            .unwrap();
        transaction
            .execute(
                "UPDATE anonymous_mailbox_meta SET total_items = ?1, total_bytes = ?1",
                params![as_i64(item_cap).unwrap()],
            )
            .unwrap();
        transaction.commit().unwrap();
        drop(connection);
        assert_item_accounting(&store, &mailbox, item_cap, item_cap);
        drop(store);

        let reopened = context.open();
        assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);
        assert!(matches!(
            reopened.put(&exact, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::Existing(_)
        ));
        let changed = context.put(mailbox, exact.item_id, b"y", exact.expires_at);
        assert_eq!(
            reopened.put(&changed, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::Conflict
        );
        let blocked = context.put(mailbox, [0xB4; 16], b"z", NOW + 100);
        assert_eq!(
            reopened.put(&blocked, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::AtCapacity
        );
        assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);

        let wrong_reader = IdentityKeyPair::generate();
        let wrong_ack = AnonymousMailboxAckV1::new(
            mailbox,
            [0xB5; 16],
            exact.item_id,
            exact.sealed_commitment(),
            NOW + 1,
            &wrong_reader,
        )
        .unwrap();
        assert_eq!(
            reopened.ack(&wrong_ack, NOW + 1),
            Err(AnonymousMailboxStoreError::Rejected)
        );
        assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);

        let ack = AnonymousMailboxAckV1::new(
            mailbox,
            [0xB6; 16],
            exact.item_id,
            exact.sealed_commitment(),
            NOW + 1,
            &context.reader,
        )
        .unwrap();
        assert_eq!(
            reopened.ack(&ack, NOW + 1).unwrap(),
            AnonymousMailboxAckOutcome::Acknowledged
        );
        assert_item_accounting(&reopened, &mailbox, item_cap - 1, item_cap - 1);
        assert!(matches!(
            reopened.put(&blocked, NOW + 1).unwrap(),
            AnonymousMailboxPutOutcome::Stored(_)
        ));
        assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);
        drop(reopened);

        let after_ack_restart = context.open();
        assert_item_accounting(&after_ack_restart, &mailbox, item_cap, item_cap);
        let first_visible = after_ack_restart
            .pull_one(&context.pull(mailbox, Vec::new(), NOW + 3), NOW + 3)
            .unwrap();
        let mut first_unexpired_item_id = [0_u8; 16];
        first_unexpired_item_id[..8].copy_from_slice(&3_u64.to_le_bytes());
        assert!(matches!(
            first_visible,
            AnonymousMailboxPullOutcome::Item(ref item)
                if item.item_id == first_unexpired_item_id
        ));
        let cleanup = after_ack_restart.cleanup(NOW + 3).unwrap();
        assert_eq!(cleanup.items_removed, 1);
        assert_eq!(cleanup.bytes_removed, 1);
        assert_eq!(cleanup.acknowledgements_removed, 0);
        assert_item_accounting(&after_ack_restart, &mailbox, item_cap - 1, item_cap - 1);

        let after_expiry = context.put(mailbox, [0xB7; 16], b"q", NOW + 100);
        assert!(matches!(
            after_ack_restart.put(&after_expiry, NOW + 3).unwrap(),
            AnonymousMailboxPutOutcome::Stored(_)
        ));
        assert_item_accounting(&after_ack_restart, &mailbox, item_cap, item_cap);
        drop(after_ack_restart);

        let final_reopen = context.open();
        assert_item_accounting(&final_reopen, &mailbox, item_cap, item_cap);
        assert!(matches!(
            final_reopen.put(&after_expiry, NOW + 4).unwrap(),
            AnonymousMailboxPutOutcome::Existing(_)
        ));
    }

    #[test]
    fn cleanup_is_bounded_and_failure_rolls_back_without_counter_repair() {
        let mut context = TestContext::new();
        context.config.cleanup_batch_size = 1;
        let mailbox = [0x71; 32];
        let lease = context.lease(mailbox, [0x72; 16], 2, 64, NOW + 1_000);
        let store = context.open();
        store.create(&lease, NOW).unwrap();
        store
            .put(&context.put(mailbox, [0x73; 16], b"one", NOW + 2), NOW)
            .unwrap();
        store
            .put(&context.put(mailbox, [0x74; 16], b"two", NOW + 2), NOW)
            .unwrap();
        let report = store.cleanup(NOW + 3).unwrap();
        assert_eq!(report.items_removed, 1);
        assert_eq!(
            count(
                &store.connection.lock().transaction().unwrap(),
                "SELECT COUNT(*) FROM anonymous_mailbox_items"
            )
            .unwrap(),
            1
        );

        store
            .connection
            .lock()
            .execute_batch(
                "CREATE TRIGGER anonymous_mailbox_test_cleanup_abort
                 BEFORE DELETE ON anonymous_mailbox_items
                 BEGIN SELECT RAISE(ABORT, 'test'); END;",
            )
            .unwrap();
        assert_eq!(
            store.cleanup(NOW + 3),
            Err(AnonymousMailboxStoreError::Unavailable)
        );
        let remaining: i64 = store
            .connection
            .lock()
            .query_row("SELECT COUNT(*) FROM anonymous_mailbox_items", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(remaining, 1);

        store
            .connection
            .lock()
            .execute_batch(
                "DROP TRIGGER anonymous_mailbox_test_cleanup_abort;
                 UPDATE anonymous_mailbox_meta SET total_bytes = total_bytes + 1;",
            )
            .unwrap();
        assert_eq!(
            store.cleanup(NOW + 3),
            Err(AnonymousMailboxStoreError::Corrupt)
        );
        let after_corrupt: i64 = store
            .connection
            .lock()
            .query_row("SELECT COUNT(*) FROM anonymous_mailbox_items", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(after_corrupt, 1);
    }

    #[test]
    fn schema_has_no_identity_or_routing_columns_and_unknown_version_fails_closed() {
        let context = TestContext::new();
        let store = context.open();
        let connection = store.connection.lock();
        let mut statement = connection
            .prepare(
                "SELECT m.name, p.name FROM sqlite_master m, pragma_table_info(m.name) p
                 WHERE m.type = 'table' AND m.name LIKE 'anonymous_mailbox_%'",
            )
            .unwrap();
        let columns: Vec<String> = statement
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        for forbidden in [
            "sender", "receiver", "wallet", "route", "endpoint", "identity",
        ] {
            assert!(columns.iter().all(|column| !column.contains(forbidden)));
        }
        drop(statement);
        drop(connection);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for path in [
                std::path::PathBuf::from(&context.config.db_path),
                std::path::PathBuf::from(format!("{}-wal", context.config.db_path)),
                std::path::PathBuf::from(format!("{}-shm", context.config.db_path)),
            ] {
                if path.exists() {
                    let mode = std::fs::symlink_metadata(path)
                        .unwrap()
                        .permissions()
                        .mode()
                        & 0o777;
                    assert_eq!(mode, 0o600);
                }
            }
        }
        drop(store);
        let connection = Connection::open(&context.config.db_path).unwrap();
        connection.pragma_update(None, "user_version", 99).unwrap();
        drop(connection);
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                context.config.clone(),
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::UnsupportedSchema)
        ));
    }

    #[test]
    fn foreign_table_on_user_version_zero_fails_without_mailbox_mutation() {
        let context = TestContext::new();
        let connection = Connection::open(&context.config.db_path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE pending_messages (
                    id INTEGER PRIMARY KEY,
                    body TEXT NOT NULL
                 );
                 INSERT INTO pending_messages (body) VALUES ('still-here');",
            )
            .unwrap();
        drop(connection);

        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                context.config.clone(),
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::UnsupportedSchema)
        ));

        let connection = Connection::open(&context.config.db_path).unwrap();
        assert_eq!(schema_user_version(&connection), 0);
        assert_eq!(anonymous_mailbox_object_count(&connection), 0);
        assert_eq!(
            connection
                .query_row(
                    "SELECT body FROM pending_messages WHERE id = 1",
                    [],
                    |row| { row.get::<_, String>(0) }
                )
                .unwrap(),
            "still-here"
        );
    }

    #[test]
    fn foreign_view_and_trigger_on_user_version_zero_fail_without_mailbox_mutation() {
        let context = TestContext::new();
        let connection = Connection::open(&context.config.db_path).unwrap();
        connection
            .execute_batch(
                "CREATE VIEW pending_view AS SELECT name FROM sqlite_master;
                 CREATE TRIGGER pending_view_block
                 INSTEAD OF INSERT ON pending_view
                 BEGIN
                    SELECT RAISE(ABORT, 'blocked');
                 END;",
            )
            .unwrap();
        drop(connection);

        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                context.config.clone(),
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::UnsupportedSchema)
        ));

        let connection = Connection::open(&context.config.db_path).unwrap();
        assert_eq!(schema_user_version(&connection), 0);
        assert_eq!(anonymous_mailbox_object_count(&connection), 0);
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE name IN ('pending_view', 'pending_view_block')",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            2
        );
    }

    #[test]
    fn empty_reserved_database_initializes_and_reopens() {
        let context = TestContext::new();
        std::fs::write(&context.config.db_path, []).unwrap();

        let store = context.open();
        drop(store);

        let connection = Connection::open(&context.config.db_path).unwrap();
        assert_eq!(schema_user_version(&connection), SCHEMA_VERSION);
        assert!(anonymous_mailbox_object_count(&connection) > 0);
        drop(connection);

        let reopened = context.open();
        drop(reopened);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_database_path_fails_closed() {
        use std::os::unix::fs::symlink;

        let context = TestContext::new();
        let target_path = context._directory.path().join("target.sqlite");
        Connection::open(&target_path).unwrap();
        let link_path = context._directory.path().join("alias.sqlite");
        symlink(&target_path, &link_path).unwrap();
        let mut config = context.config.clone();
        config.db_path = link_path.display().to_string();
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                config,
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::Rejected)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn hardlinked_database_is_rejected_without_mode_side_effect() {
        use std::os::unix::fs::PermissionsExt;

        let context = TestContext::new();
        let parent = std::fs::canonicalize(context._directory.path()).unwrap();
        let external = parent.join("external.sqlite");
        std::fs::write(&external, b"external").unwrap();
        std::fs::set_permissions(&external, std::fs::Permissions::from_mode(0o640)).unwrap();
        let alias = parent.join("mailbox-hardlink.sqlite");
        std::fs::hard_link(&external, &alias).unwrap();
        let mode_before = std::fs::metadata(&external).unwrap().permissions().mode() & 0o777;
        let mut config = context.config.clone();
        config.db_path = alias.display().to_string();
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                config,
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::Rejected)
        ));
        let mode_after = std::fs::metadata(&external).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode_after, mode_before);
    }

    #[cfg(unix)]
    #[test]
    fn existing_owned_database_mode_is_normalized_before_sqlite_activation() {
        use std::os::unix::fs::PermissionsExt;

        let context = TestContext::new();
        let database = PathBuf::from(&context.config.db_path);
        std::fs::write(&database, []).unwrap();
        std::fs::set_permissions(&database, std::fs::Permissions::from_mode(0o640)).unwrap();
        let store = context.open();
        let mode = std::fs::metadata(&database).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        drop(store);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_parent_component_is_rejected_without_database_creation() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let context = TestContext::new();
        let parent = std::fs::canonicalize(context._directory.path()).unwrap();
        let actual = parent.join("actual-private");
        std::fs::create_dir(&actual).unwrap();
        std::fs::set_permissions(&actual, std::fs::Permissions::from_mode(0o700)).unwrap();
        let alias = parent.join("parent-alias");
        symlink(&actual, &alias).unwrap();
        let mut config = context.config.clone();
        config.db_path = alias.join("mailbox.sqlite").display().to_string();
        assert!(matches!(
            SqliteAnonymousMailboxStore::open(
                config,
                context.target.public_key_bytes(),
                CURSOR_SECRET,
            ),
            Err(AnonymousMailboxStoreError::Rejected)
        ));
        assert!(!actual.join("mailbox.sqlite").exists());
    }
}
