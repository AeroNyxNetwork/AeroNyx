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

use aeronyx_core::protocol::anonymous_mailbox::{
    AnonymousMailboxAckV1, AnonymousMailboxLeaseCreateV1, AnonymousMailboxPullOneV1,
    AnonymousMailboxPutV1, MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE,
    MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES,
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

const SCHEMA_VERSION: i64 = 1;
const MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
const CURSOR_VERSION: u8 = 1;
const CURSOR_BODY_BYTES: usize = 1 + 8 + 8 + 8;
const CURSOR_TAG_BYTES: usize = 32;
const CURSOR_BYTES: usize = CURSOR_BODY_BYTES + CURSOR_TAG_BYTES;
const CURSOR_TTL_SECS: u64 = 5 * 60;
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnonymousMailboxCleanupReport {
    pub leases_removed: u64,
    pub items_removed: u64,
    pub bytes_removed: u64,
    pub acknowledgements_removed: u64,
    pub tickets_removed: u64,
}

/// Synchronous local capability boundary. Future async callers must place it
/// behind an explicit blocking boundary.
pub trait AnonymousMailboxCustodyRepository: Send + Sync {
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
        initialize_or_verify_schema(&mut connection)?;
        let startup_limits = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        validate_totals_limits(&load_totals(&startup_limits)?, &config)?;
        startup_limits
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        Ok(Self {
            config,
            target_node_id,
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
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
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
) -> Result<(), AnonymousMailboxStoreError> {
    let user_version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if user_version != 0 && user_version != SCHEMA_VERSION {
        return Err(AnonymousMailboxStoreError::UnsupportedSchema);
    }
    if user_version == 0 {
        let managed_tables: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name LIKE 'anonymous_mailbox_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if managed_tables != 0 {
            return Err(AnonymousMailboxStoreError::UnsupportedSchema);
        }
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        transaction
            .execute_batch(
                "CREATE TABLE anonymous_mailbox_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL,
                    total_leases INTEGER NOT NULL CHECK (total_leases >= 0),
                    total_items INTEGER NOT NULL CHECK (total_items >= 0),
                    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0)
                 );
                 INSERT INTO anonymous_mailbox_meta VALUES (1, 1, 0, 0, 0);
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
                 CREATE INDEX anonymous_mailbox_lease_expiry
                    ON anonymous_mailbox_leases(expires_at, mailbox_id);
                 CREATE INDEX anonymous_mailbox_item_pull
                    ON anonymous_mailbox_items(mailbox_id, sequence, expires_at);
                 CREATE INDEX anonymous_mailbox_item_expiry
                    ON anonymous_mailbox_items(expires_at, mailbox_id, item_id);
                 CREATE INDEX anonymous_mailbox_ack_expiry
                    ON anonymous_mailbox_acks(retain_until, mailbox_id, item_id);
                 PRAGMA user_version = 1;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    }
    let version: i64 = connection
        .query_row(
            "SELECT schema_version FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if version != SCHEMA_VERSION {
        return Err(AnonymousMailboxStoreError::UnsupportedSchema);
    }
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    audit_counters(&transaction)?;
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

fn audit_counters(transaction: &Transaction<'_>) -> Result<(), AnonymousMailboxStoreError> {
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

struct PrivateSqliteTarget {
    resolved_path: PathBuf,
    #[cfg(unix)]
    parent: File,
}

#[cfg(unix)]
fn prepare_private_sqlite_target(
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
fn prepare_private_sqlite_target(
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
fn verify_private_file(
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
fn verify_private_file(
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
            reopened.create(&request, NOW + 2).unwrap(),
            AnonymousMailboxCreateOutcome::Existing(_)
        ));

        let conflicting = context.lease([0x12; 32], [0x21; 16], 2, 32, NOW + 1_000);
        assert_eq!(
            reopened.create(&conflicting, NOW + 2).unwrap(),
            AnonymousMailboxCreateOutcome::Conflict
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
