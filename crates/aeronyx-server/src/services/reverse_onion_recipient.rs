// ============================================
// File: crates/aeronyx-server/src/services/reverse_onion_recipient.rs
// ============================================
//! Private reverse-onion recipient recovery journal; no network or execution.
//!
//! [REVERSE-ONION-RECIPIENT 2026-10-04 by Codex] All methods are synchronous:
//! runtime callers must move an owned Arc into spawn_blocking, then await the
//! durable result before POST/dispatch. Cancellation never undoes Armed. An
//! Armed row without a stored Result is ambiguous on restart, never executable.
//! This module does not verify source replies, invent wire, or log identifiers.
//!
//! Last Modified: v1.1.0 — Nonmutating preflight and observed aggregate fence.
//! [REVERSE-ONION-RECIPIENT-DB-BOUNDARY 2026-10-04 by Codex] Phase/schema
//! semantics unchanged; a sampled physical bound is not an OS hard quota.
//! [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Typed relay admission
//! stores R's execution expiry, not an independently authenticated source route.

use std::path::Path;
use std::sync::Mutex;

use aeronyx_core::protocol::chat::BlindRelayEnvelope;
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, VerifiedRecipientLease, MAX_REVERSE_ONION_FRAME_BYTES,
    REVERSE_ONION_ENVELOPE_LIFETIME_SECS, REVERSE_ONION_RESULT_RETENTION_SECS,
};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior};

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::unix::{fs::{MetadataExt, OpenOptionsExt}, io::AsRawFd};

const MAX_ENTRIES: usize = 1024;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
const UNLEASED_EVIDENCE_SECS: u64 =
    REVERSE_ONION_ENVELOPE_LIFETIME_SECS + REVERSE_ONION_RESULT_RETENTION_SECS;
const APPLICATION_ID: i64 = 0x41585250;
const META_SQL: &str = "CREATE TABLE recipient_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), relay BLOB NOT NULL, recipient BLOB NOT NULL, clock INTEGER NOT NULL)";
const ROW_SQL: &str = "CREATE TABLE recipient_jobs (claim_id BLOB PRIMARY KEY NOT NULL, route_id BLOB UNIQUE, claim BLOB NOT NULL, lease BLOB, result BLOB, route_deadline INTEGER NOT NULL, retain_until INTEGER NOT NULL, phase INTEGER NOT NULL)";

/// Fixed ceilings also constrain caller-supplied configuration.
pub(crate) struct RecipientJournalLimits {
    pub(crate) max_entries: usize,
    pub(crate) max_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RecipientJournalError {
    #[error("recipient journal rejected")]
    Rejected,
    #[error("recipient journal busy")]
    Busy,
    #[error("recipient journal conflict")]
    Conflict,
    #[error("recipient journal at capacity")]
    Capacity,
    #[error("recipient journal expired")]
    Expired,
    #[error("recipient journal ambiguous")]
    Ambiguous,
    #[error("recipient journal corrupt")]
    Corrupt,
    #[error("recipient journal unavailable")]
    Unavailable,
}

type Result<T> = std::result::Result<T, RecipientJournalError>;

/// Intentionally no Debug: caller must not log frames or opaque identifiers.
pub(crate) enum RecipientRecovery {
    Poll { claim_id: [u8; 16], exact_bytes: Vec<u8> },
    LeaseReady { claim_id: [u8; 16] },
    Result { claim_id: [u8; 16], exact_bytes: Vec<u8> },
    Ambiguous { claim_id: [u8; 16] },
}

/// Advance `next_after` until None, then begin another bounded recovery pass.
/// This avoids old unacknowledged results permanently starving later polls.
pub(crate) struct RecipientRecoveryPage {
    pub(crate) items: Vec<RecipientRecovery>,
    pub(crate) next_after: Option<[u8; 16]>,
}

/// Returned only after the one-time Armed commit, with the exact reply-binding
/// material needed by the worker to construct the existing core Result frame.
/// It is NOT permission to dispatch twice or to reconstruct after restart.
pub(crate) struct RecipientDispatch {
    pub(crate) envelope: BlindRelayEnvelope,
    pub(crate) claim: ReverseOnionFrameV1,
    pub(crate) lease: ReverseOnionFrameV1,
    // [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] On record_relay_lease
    // admission this legacy-named field is R's signed execution bound only.
    // It does not attest hidden source admission; terminal must not infer that.
    pub(crate) route_deadline: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
enum Phase { Poll = 1, Lease = 2, Armed = 3, Result = 4, Ambiguous = 5 }

struct Record {
    claim: ReverseOnionFrameV1,
    lease: Option<ReverseOnionFrameV1>,
    result: Option<ReverseOnionFrameV1>,
    deadline: u64,
    retain_until: u64,
    phase: Phase,
}

struct Inner {
    connection: Connection,
    poisoned: bool,
}

/// One journal is bound to one adjacent pair. No endpoint is persisted here.
/// A process-lifetime inode flock and SQLite EXCLUSIVE mode exclude a second
/// cooperating process/handle. Same-euid hostile rename is not eliminated.
pub(crate) struct ReverseOnionRecipientJournal {
    inner: Mutex<Inner>,
    relay: [u8; 32],
    recipient: [u8; 32],
    limits: RecipientJournalLimits,
    #[cfg(unix)]
    _inode_lock: File,
    #[cfg(unix)]
    _parent: File,
    #[cfg(unix)]
    db_path: std::path::PathBuf,
    #[cfg(unix)]
    physical_limit: u64,
    #[cfg(unix)]
    inode_identity: (u64, u64),
    #[cfg(unix)]
    parent_identity: (u64, u64),
    #[cfg(test)]
    post_commit_sidecar_bytes: std::sync::atomic::AtomicU64,
}

impl ReverseOnionRecipientJournal {
    /// Explicit construction only; disabled startup must never call this API.
    /// No test/in-memory bypass constructor, and non-Unix is fail-closed.
    #[cfg(unix)]
    pub(crate) fn open(
        path: &Path,
        relay: [u8; 32],
        recipient: [u8; 32],
        limits: RecipientJournalLimits,
        now: u64,
    ) -> Result<Self> {
        use super::chat_relay_mailbox::{prepare_private_sqlite_target, verify_private_file};
        if limits.max_entries == 0 || limits.max_entries > MAX_ENTRIES
            || limits.max_bytes == 0 || limits.max_bytes > MAX_BYTES
            || relay == recipient || relay == [0; 32] || recipient == [0; 32]
            || path == Path::new(":memory:")
        {
            return Err(RecipientJournalError::Rejected);
        }
        to_sql(now)?;
        // [REVERSE-ONION-RECIPIENT-DB-BOUNDARY 2026-10-04 by Codex]
        // Check known existing boundaries before the helper creates/chmods.
        let physical_limit = recipient_physical_limit(&limits)?;
        preflight_recipient_files(path, physical_limit)?;
        let target = prepare_private_sqlite_target(path)
            .map_err(|_| RecipientJournalError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| RecipientJournalError::Rejected)?;
        let inode = std::fs::OpenOptions::new().read(true).write(true)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC | nix::libc::O_NONBLOCK)
            .open(&target.resolved_path).map_err(|_| RecipientJournalError::Unavailable)?;
        let metadata = inode.metadata().map_err(|_| RecipientJournalError::Unavailable)?;
        // SAFETY: geteuid has no preconditions or pointer arguments.
        let uid = unsafe { nix::libc::geteuid() };
        if !metadata.is_file() || metadata.uid() != uid || metadata.nlink() != 1
            || metadata.mode() & 0o777 != 0o600
        {
            return Err(RecipientJournalError::Rejected);
        }
        // Bound even the integrity scan before SQLite touches an existing DB.
        // This is a logical/physical repository cap, not an OS hard quota.
        if metadata.len() > physical_limit { return Err(RecipientJournalError::Capacity); }
        // SAFETY: inode owns a live fd retained for the complete journal lifetime.
        if unsafe { nix::libc::flock(inode.as_raw_fd(), nix::libc::LOCK_EX | nix::libc::LOCK_NB) } != 0 {
            return Err(RecipientJournalError::Busy);
        }
        // Existing rollback journal is allowed only after private-file checks.
        // WAL is not this schema's durability mode; do not silently convert it.
        audit_recipient_sidecars(&target.resolved_path, physical_limit, metadata.len())?;
        let parent_metadata = target.parent.metadata().map_err(|_| RecipientJournalError::Unavailable)?;
        validate_recipient_parent(&parent_metadata)?;
        let mut connection = Connection::open_with_flags(&target.resolved_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW)
            .map_err(|_| RecipientJournalError::Unavailable)?;
        let after = std::fs::symlink_metadata(&target.resolved_path)
            .map_err(|_| RecipientJournalError::Unavailable)?;
        if metadata.dev() != after.dev() || metadata.ino() != after.ino() {
            return Err(RecipientJournalError::Rejected);
        }
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| RecipientJournalError::Rejected)?;
        connection.execute_batch("PRAGMA busy_timeout=0; PRAGMA trusted_schema=OFF; PRAGMA temp_store=MEMORY; PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=DELETE; PRAGMA synchronous=EXTRA; PRAGMA fullfsync=ON; PRAGMA foreign_keys=ON;")
            .map_err(|_| RecipientJournalError::Unavailable)?;
        let page_size: i64 = connection.query_row("PRAGMA page_size", [], |r| r.get(0)).map_err(unavailable)?;
        if !(512..=65536).contains(&page_size) || !(page_size as u64).is_power_of_two() {
            return Err(RecipientJournalError::Corrupt);
        }
        connection.pragma_update(None, "max_page_count", (physical_limit / page_size as u64) as i64)
            .map_err(unavailable)?;
        audit_recipient_pragmas(&connection, physical_limit)?;
        let integrity: String = connection.query_row("PRAGMA quick_check", [], |row| row.get(0))
            .map_err(|_| RecipientJournalError::Corrupt)?;
        if integrity != "ok" { return Err(RecipientJournalError::Corrupt); }
        initialize_schema(&mut connection, relay, recipient, now)?;
        let journal = Self {
            inner: Mutex::new(Inner { connection, poisoned: false }),
            relay, recipient, limits, _inode_lock: inode, _parent: target.parent,
            db_path: target.resolved_path, physical_limit,
            inode_identity: (metadata.dev(), metadata.ino()),
            parent_identity: (parent_metadata.dev(), parent_metadata.ino()),
            #[cfg(test)]
            post_commit_sidecar_bytes: std::sync::atomic::AtomicU64::new(0),
        };
        // [REVERSE-ONION-RECIPIENT 2026-10-04 by Codex] Crash barrier: audit
        // all bounded rows first, then atomically retire unresolved dispatches.
        journal.transaction(now, |tx| {
            let ids = all_ids(tx)?;
            for id in ids { journal.load(tx, id)?.ok_or(RecipientJournalError::Corrupt)?; }
            tx.execute("UPDATE recipient_jobs SET phase=5 WHERE phase=3", [])
                .map_err(unavailable)?;
            Ok(())
        })?;
        Ok(journal)
    }

    #[cfg(not(unix))]
    pub(crate) fn open(
        _path: &Path, _relay: [u8; 32], _recipient: [u8; 32],
        _limits: RecipientJournalLimits, _now: u64,
    ) -> Result<Self> { Err(RecipientJournalError::Rejected) }

    /// Commit exact poll bytes before any POST. Exact retries precede quotas,
    /// but expired credentials are never returned for transmission.
    pub(crate) fn prepare_poll(&self, claim: &ReverseOnionFrameV1, now: u64) -> Result<Vec<u8>> {
        claim.verify_claim(self.relay, self.recipient, now)
            .map_err(|_| RecipientJournalError::Rejected)?;
        let bytes = claim.encode();
        self.transaction(now, |tx| {
            if let Some(row) = self.load(tx, claim.claim_id())? {
                row.claim.require_exact_retry(claim).map_err(|_| RecipientJournalError::Conflict)?;
                if row.phase != Phase::Poll { return Err(RecipientJournalError::Conflict); }
                return Ok(bytes.clone());
            }
            let retention = claim.expires_at().checked_add(UNLEASED_EVIDENCE_SECS).ok_or(RecipientJournalError::Rejected)?;
            tx.execute("INSERT INTO recipient_jobs VALUES(?1,NULL,?2,NULL,NULL,0,?3,1)",
                params![claim.claim_id().as_slice(), bytes, to_sql(retention)?]).map_err(unavailable)?;
            Ok(bytes.clone())
        })
    }

    /// The deadline is already authenticated by route admission, not taken
    /// from an untrusted HTTP query. It becomes immutable with the exact lease.
    pub(crate) fn record_lease(&self, id: [u8; 16], lease: &ReverseOnionFrameV1,
        authenticated_route_deadline: u64, now: u64) -> Result<()> {
        self.record_bound_lease(id, lease, authenticated_route_deadline, now)
    }

    /// [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Consume pinned relay
    /// authority, recheck against this journal's identities and durable Claim,
    /// and store the signed expiry immutably. No caller scalar can extend it.
    /// Existing exact replay may recover evidence but never rearms execution.
    pub(crate) fn record_relay_lease(&self, proof: VerifiedRecipientLease<'_>, now: u64) -> Result<()> {
        let lease = proof.lease();
        if lease.relay() != self.relay || lease.immediate_recipient() != self.recipient {
            return Err(RecipientJournalError::Rejected);
        }
        self.record_bound_lease(lease.claim_id(), lease, proof.relay_execution_expiry(), now)
    }

    // Shared persistence only; the two entry points retain distinct authority
    // contracts. Existing schema, exact retries, retention and phase CAS remain.
    fn record_bound_lease(&self, id: [u8; 16], lease: &ReverseOnionFrameV1,
        authenticated_route_deadline: u64, now: u64) -> Result<()> {
        self.transaction(now, |tx| {
            let row = self.load(tx, id)?.ok_or(RecipientJournalError::Rejected)?;
            if let Some(existing) = &row.lease {
                existing.require_exact_retry(lease).map_err(|_| RecipientJournalError::Conflict)?;
                return if row.deadline == authenticated_route_deadline { Ok(()) }
                    else { Err(RecipientJournalError::Conflict) };
            }
            if row.phase != Phase::Poll { return Err(RecipientJournalError::Conflict); }
            lease.verify_lease(&row.claim, authenticated_route_deadline, now)
                .map_err(|_| RecipientJournalError::Rejected)?;
            let duplicate: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM recipient_jobs WHERE route_id=?1)",
                params![lease.route_id().as_slice()], |r| r.get(0)).map_err(unavailable)?;
            if duplicate { return Err(RecipientJournalError::Conflict); }
            changed(tx.execute("UPDATE recipient_jobs SET route_id=?1,lease=?2,route_deadline=?3,retain_until=?4,phase=2 WHERE claim_id=?5 AND phase=1",
                params![lease.route_id().as_slice(), lease.encode(), to_sql(authenticated_route_deadline)?,
                    to_sql(lease.replay_evidence_deadline().map_err(|_| RecipientJournalError::Rejected)?)?, id.as_slice()]).map_err(unavailable)?)
        })
    }

    /// Commit Armed before yielding the envelope for ONE local dispatch.
    /// An Armed retry is ambiguous even in the same process; no dedup assumption.
    pub(crate) fn arm(&self, id: [u8; 16], now: u64) -> Result<RecipientDispatch> {
        self.transaction(now, |tx| {
            let row = self.load(tx, id)?.ok_or(RecipientJournalError::Rejected)?;
            if row.phase == Phase::Armed || row.phase == Phase::Ambiguous {
                return Err(RecipientJournalError::Ambiguous);
            }
            if row.phase != Phase::Lease { return Err(RecipientJournalError::Conflict); }
            let lease = row.lease.ok_or(RecipientJournalError::Corrupt)?;
            let envelope = lease.verify_lease(&row.claim, row.deadline, now)
                .map_err(|_| RecipientJournalError::Expired)?;
            changed(tx.execute("UPDATE recipient_jobs SET phase=3 WHERE claim_id=?1 AND phase=2",
                params![id.as_slice()]).map_err(unavailable)?)?;
            Ok(RecipientDispatch { envelope, claim: row.claim, lease, route_deadline: row.deadline })
        })
    }

    /// The first exact Result is committed before submission. If capacity or
    /// storage fails, leave Armed/non-reexecutable; never rerun the terminal.
    pub(crate) fn record_result(&self, id: [u8; 16], result: &ReverseOnionFrameV1, now: u64) -> Result<()> {
        self.transaction(now, |tx| {
            let row = self.load(tx, id)?.ok_or(RecipientJournalError::Rejected)?;
            if let Some(existing) = &row.result {
                return existing.require_exact_retry(result).map_err(|_| RecipientJournalError::Conflict);
            }
            if row.phase != Phase::Armed { return Err(RecipientJournalError::Ambiguous); }
            let lease = row.lease.as_ref().ok_or(RecipientJournalError::Corrupt)?;
            result.verify_result(&row.claim, lease, row.deadline, now)
                .map_err(|_| RecipientJournalError::Rejected)?;
            changed(tx.execute("UPDATE recipient_jobs SET result=?1,phase=4 WHERE claim_id=?2 AND phase=3",
                params![result.encode(), id.as_slice()]).map_err(unavailable)?)
        })
    }

    /// Bounded restart work list. Armed is reported ambiguous, not executable.
    /// Result replay stops at result expiry; HTTP 2xx never deletes evidence.
    pub(crate) fn resume(&self, after: Option<[u8; 16]>, limit: usize, now: u64) -> Result<RecipientRecoveryPage> {
        if limit == 0 || limit > 64 { return Err(RecipientJournalError::Rejected); }
        self.transaction(now, |tx| {
            let mut out = Vec::new();
            let mut scanned = 0;
            let mut last = None;
            let mut has_more = false;
            for id in all_ids(tx)? {
                if after.is_some_and(|cursor| id <= cursor) { continue; }
                if scanned == limit { has_more = true; break; }
                scanned += 1;
                last = Some(id);
                let row = self.load(tx, id)?.ok_or(RecipientJournalError::Corrupt)?;
                let next = match row.phase {
                    // [RECIPIENT-HISTORICAL-POLL 2026-10-04 by Codex]
                    // Claim expiry does not prove R issued no Lease. Preserve
                    // exact recovery through the entire immutable horizon.
                    Phase::Poll if now < row.retain_until => Some(RecipientRecovery::Poll {
                        claim_id: id, exact_bytes: row.claim.encode(),
                    }),
                    Phase::Lease if now < row.lease.as_ref().ok_or(RecipientJournalError::Corrupt)?.expires_at() =>
                        Some(RecipientRecovery::LeaseReady { claim_id: id }),
                    Phase::Result if now < row.result.as_ref().ok_or(RecipientJournalError::Corrupt)?.expires_at() =>
                        Some(RecipientRecovery::Result { claim_id: id, exact_bytes: row.result.as_ref().ok_or(RecipientJournalError::Corrupt)?.encode() }),
                    Phase::Armed | Phase::Ambiguous => Some(RecipientRecovery::Ambiguous { claim_id: id }),
                    _ => None,
                };
                if let Some(next) = next { out.push(next); }
            }
            Ok(RecipientRecoveryPage { items: out, next_after: if has_more { last } else { None } })
        })
    }

    /// Bounded transactional evidence cleanup, never before its replay bound.
    pub(crate) fn cleanup(&self, limit: usize, now: u64) -> Result<usize> {
        if limit == 0 || limit > 64 { return Err(RecipientJournalError::Rejected); }
        self.transaction(now, |tx| {
            let mut removed = 0;
            for id in all_ids(tx)? {
                let row = self.load(tx, id)?.ok_or(RecipientJournalError::Corrupt)?;
                if now >= row.retain_until {
                    changed(tx.execute("DELETE FROM recipient_jobs WHERE claim_id=?1 AND retain_until<=?2",
                        params![id.as_slice(), to_sql(now)?]).map_err(unavailable)?)?;
                    removed += 1;
                    if removed == limit { break; }
                }
            }
            Ok(removed)
        })
    }

    // [REVERSE-ONION-RECIPIENT 2026-10-04 by Codex] One transaction serializes
    // time, actual byte/count bounds and mutation. Any ambiguous DB error poisons
    // this handle; a reopen audits and retires Armed before returning work.
    fn transaction<T>(&self, now: u64, action: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<T> {
        let mut inner = self.inner.try_lock().map_err(|_| RecipientJournalError::Busy)?;
        if inner.poisoned { return Err(RecipientJournalError::Unavailable); }
        let outcome = (|| {
            let tx = inner.connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(unavailable)?;
            let operation = (|| {
                let (relay, recipient, clock): (Vec<u8>, Vec<u8>, i64) = tx.query_row(
                    "SELECT relay,recipient,clock FROM recipient_meta WHERE singleton=1 AND length(relay)=32 AND length(recipient)=32",
                    [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).map_err(|_| RecipientJournalError::Corrupt)?;
                if relay.as_slice() != self.relay.as_slice()
                    || recipient.as_slice() != self.recipient.as_slice() || clock < 0
                {
                    return Err(RecipientJournalError::Corrupt);
                }
                if to_sql(now)? < clock { return Err(RecipientJournalError::Rejected); }
                self.audit_bounds(&tx)?;
                let value = action(&tx)?;
                self.audit_bounds(&tx)?;
                changed(tx.execute("UPDATE recipient_meta SET clock=?1 WHERE singleton=1 AND clock=?2",
                    params![to_sql(now)?, clock]).map_err(unavailable)?)?;
                Ok(value)
            })();
            match operation {
                Ok(value) => {
                    tx.commit().map_err(unavailable)?;
                    // A failed post-commit durability fence is ambiguous, not
                    // permission for a second dispatch. Poison before returning.
                    self.post_operation_fence(&inner.connection)
                        .map_err(|_| RecipientJournalError::Unavailable)?;
                    Ok(value)
                }
                Err(error) => {
                    tx.rollback().map_err(unavailable)?;
                    self.post_operation_fence(&inner.connection)
                        .map_err(|_| RecipientJournalError::Unavailable)?;
                    Err(error)
                }
            }
        })();
        if matches!(&outcome, Err(RecipientJournalError::Corrupt | RecipientJournalError::Unavailable)) {
            inner.poisoned = true;
        }
        outcome
    }

    // [REVERSE-ONION-RECIPIENT-DB-BOUNDARY 2026-10-04 by Codex] Called
    // while the transaction's original mutex is still held. No value escapes
    // a failed fence; Unavailable poisons the handle before releasing the lock.
    #[cfg(unix)]
    fn post_operation_fence(&self, connection: &Connection) -> Result<()> {
        // Test-only growth after SQLite commit, before any dispatch/result is
        // published. This never changes production durability or admission.
        #[cfg(test)] {
            let bytes = self.post_commit_sidecar_bytes.swap(0, std::sync::atomic::Ordering::SeqCst);
            if bytes != 0 {
                std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600)
                    .open(recipient_sidecar(&self.db_path, "-journal"))
                    .and_then(|file| file.set_len(bytes)).map_err(|_| RecipientJournalError::Unavailable)?;
            }
        }
        let held = self._inode_lock.metadata().map_err(|_| RecipientJournalError::Unavailable)?;
        validate_recipient_file(&held)?;
        if (held.dev(), held.ino()) != self.inode_identity { return Err(RecipientJournalError::Rejected); }
        let parent = self._parent.metadata().map_err(|_| RecipientJournalError::Unavailable)?;
        validate_recipient_parent(&parent)?;
        if (parent.dev(), parent.ino()) != self.parent_identity { return Err(RecipientJournalError::Rejected); }
        let parent_path = self.db_path.parent().ok_or(RecipientJournalError::Rejected)?;
        let observed_parent = std::fs::symlink_metadata(parent_path).map_err(|_| RecipientJournalError::Unavailable)?;
        validate_recipient_parent(&observed_parent)?;
        if (observed_parent.dev(), observed_parent.ino()) != self.parent_identity {
            return Err(RecipientJournalError::Rejected);
        }
        let observed = std::fs::symlink_metadata(&self.db_path).map_err(|_| RecipientJournalError::Unavailable)?;
        validate_recipient_file(&observed)?;
        if (observed.dev(), observed.ino()) != self.inode_identity || observed.len() != held.len() {
            return Err(RecipientJournalError::Rejected);
        }
        audit_recipient_sidecars(&self.db_path, self.physical_limit, observed.len())?;
        audit_recipient_pragmas(connection, self.physical_limit)?;
        self._parent.sync_all().map_err(|_| RecipientJournalError::Unavailable)
    }

    #[cfg(not(unix))]
    fn post_operation_fence(&self, _connection: &Connection) -> Result<()> { Err(RecipientJournalError::Rejected) }

    fn audit_bounds(&self, tx: &Transaction<'_>) -> Result<()> {
        let (count, bytes): (i64, i64) = tx.query_row(
            "SELECT count(*),coalesce(sum(c+l+r),0) FROM (SELECT length(claim) AS c,coalesce(length(lease),0) AS l,coalesce(length(result),0) AS r FROM recipient_jobs LIMIT ?1)",
            params![(MAX_ENTRIES + 1) as i64], |r| Ok((r.get(0)?, r.get(1)?))).map_err(|_| RecipientJournalError::Corrupt)?;
        if count < 0 || bytes < 0 { return Err(RecipientJournalError::Corrupt); }
        if count as usize > self.limits.max_entries || bytes as u64 > self.limits.max_bytes {
            return Err(RecipientJournalError::Capacity);
        }
        Ok(())
    }

    fn load(&self, tx: &Transaction<'_>, id: [u8; 16]) -> Result<Option<Record>> {
        let lengths: Option<(i64, Option<i64>, Option<i64>, Option<i64>)> = tx.query_row(
            "SELECT length(claim),length(lease),length(result),length(route_id) FROM recipient_jobs WHERE claim_id=?1",
            params![id.as_slice()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))
            .optional().map_err(|_| RecipientJournalError::Corrupt)?;
        let Some((claim_len, lease_len, result_len, route_len)) = lengths else { return Ok(None); };
        if claim_len != 234 || [lease_len, result_len].into_iter().flatten()
            .any(|n| n < 234 || n > MAX_REVERSE_ONION_FRAME_BYTES as i64)
            || route_len.is_some_and(|n| n != 16)
        { return Err(RecipientJournalError::Corrupt); }
        let (claim, lease, result, route, deadline, retain, phase):
            (Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>, i64, i64, i64) = tx.query_row(
                "SELECT claim,lease,result,route_id,route_deadline,retain_until,phase FROM recipient_jobs WHERE claim_id=?1",
                params![id.as_slice()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?)))
                .map_err(|_| RecipientJournalError::Corrupt)?;
        let decode = |bytes: &[u8]| ReverseOnionFrameV1::decode_for_recovery(bytes)
            .map_err(|_| RecipientJournalError::Corrupt);
        let claim = decode(&claim)?;
        claim.verify_claim(self.relay, self.recipient, claim.issued_at())
            .map_err(|_| RecipientJournalError::Corrupt)?;
        if claim.claim_id() != id || deadline < 0 || retain < 0 { return Err(RecipientJournalError::Corrupt); }
        let lease = lease.as_deref().map(decode).transpose()?;
        let result = result.as_deref().map(decode).transpose()?;
        let phase = match phase {
            1 => Phase::Poll, 2 => Phase::Lease, 3 => Phase::Armed,
            4 => Phase::Result, 5 => Phase::Ambiguous,
            _ => return Err(RecipientJournalError::Corrupt),
        };
        if let Some(lease) = &lease {
            lease.verify_lease(&claim, deadline as u64, lease.issued_at())
                .map_err(|_| RecipientJournalError::Corrupt)?;
            if route.as_deref() != Some(lease.route_id().as_slice()) || phase == Phase::Poll
                || retain as u64 != lease.replay_evidence_deadline().map_err(|_| RecipientJournalError::Corrupt)?
            { return Err(RecipientJournalError::Corrupt); }
            if let Some(result) = &result {
                result.verify_result(&claim, lease, deadline as u64, result.issued_at())
                    .map_err(|_| RecipientJournalError::Corrupt)?;
            }
        } else if phase != Phase::Poll || route.is_some() || deadline != 0
            || retain as u64 != claim.expires_at().checked_add(UNLEASED_EVIDENCE_SECS).ok_or(RecipientJournalError::Corrupt)?
        { return Err(RecipientJournalError::Corrupt); }
        if result.is_some() != (phase == Phase::Result) { return Err(RecipientJournalError::Corrupt); }
        Ok(Some(Record { claim, lease, result, deadline: deadline as u64, retain_until: retain as u64, phase }))
    }
}

// [REVERSE-ONION-RECIPIENT-DB-BOUNDARY 2026-10-04 by Codex] Local-only
// policy composition; reuse the private target helper without editing it.
fn recipient_physical_limit(limits: &RecipientJournalLimits) -> Result<u64> {
    limits.max_bytes.checked_mul(2)
        .and_then(|n| n.checked_add((MAX_ENTRIES as u64 + 256).checked_mul(4096)?))
        .ok_or(RecipientJournalError::Rejected)
}

#[cfg(unix)]
fn validate_recipient_file(metadata: &std::fs::Metadata) -> Result<()> {
    // SAFETY: geteuid has no pointer arguments or preconditions.
    if !metadata.is_file() || metadata.uid() != unsafe { nix::libc::geteuid() }
        || metadata.nlink() != 1 || metadata.mode() & 0o777 != 0o600
    { return Err(RecipientJournalError::Rejected); }
    Ok(())
}

#[cfg(unix)]
fn validate_recipient_parent(metadata: &std::fs::Metadata) -> Result<()> {
    // SAFETY: geteuid has no pointer arguments or preconditions.
    if !metadata.is_dir() || metadata.uid() != unsafe { nix::libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    { return Err(RecipientJournalError::Rejected); }
    Ok(())
}

#[cfg(unix)]
fn preflight_recipient_files(path: &Path, physical_limit: u64) -> Result<()> {
    let parent_path = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    match std::fs::symlink_metadata(parent_path) {
        Ok(parent) => validate_recipient_parent(&parent)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(RecipientJournalError::Unavailable),
    }
    let primary = match std::fs::symlink_metadata(path) {
        Ok(metadata) => { validate_recipient_file(&metadata)?; metadata.len() }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(_) => return Err(RecipientJournalError::Unavailable),
    };
    audit_recipient_sidecars(path, physical_limit, primary)
}

fn recipient_checked_aggregate(primary: u64, rollback: u64, limit: u64) -> Result<u64> {
    let total = primary.checked_add(rollback).ok_or(RecipientJournalError::Capacity)?;
    if total > limit { return Err(RecipientJournalError::Capacity); }
    Ok(total)
}

#[cfg(unix)]
fn recipient_sidecar(path: &Path, suffix: &str) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_os_string(); name.push(suffix); name.into()
}

#[cfg(unix)]
fn audit_recipient_sidecars(path: &Path, physical_limit: u64, primary: u64) -> Result<()> {
    let mut total = recipient_checked_aggregate(primary, 0, physical_limit)?;
    for suffix in ["-journal", "-wal", "-shm"] {
        match std::fs::symlink_metadata(recipient_sidecar(path, suffix)) {
            Ok(_) if suffix != "-journal" => return Err(RecipientJournalError::Corrupt),
            Ok(metadata) => {
                validate_recipient_file(&metadata)?;
                total = recipient_checked_aggregate(total, metadata.len(), physical_limit)?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(RecipientJournalError::Unavailable),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn audit_recipient_pragmas(connection: &Connection, physical_limit: u64) -> Result<()> {
    for (query, expected) in [
        ("PRAGMA busy_timeout", 0i64), ("PRAGMA trusted_schema", 0),
        ("PRAGMA temp_store", 2), ("PRAGMA synchronous", 3),
        ("PRAGMA fullfsync", 1), ("PRAGMA foreign_keys", 1),
    ] {
        let actual: i64 = connection.query_row(query, [], |r| r.get(0)).map_err(unavailable)?;
        if actual != expected { return Err(RecipientJournalError::Unavailable); }
    }
    for (query, expected) in [("PRAGMA journal_mode", "delete"), ("PRAGMA locking_mode", "exclusive")] {
        let actual: String = connection.query_row(query, [], |r| r.get(0)).map_err(unavailable)?;
        if !actual.eq_ignore_ascii_case(expected) { return Err(RecipientJournalError::Unavailable); }
    }
    let page_size: i64 = connection.query_row("PRAGMA page_size", [], |r| r.get(0)).map_err(unavailable)?;
    let pages: i64 = connection.query_row("PRAGMA page_count", [], |r| r.get(0)).map_err(unavailable)?;
    let maximum: i64 = connection.query_row("PRAGMA max_page_count", [], |r| r.get(0)).map_err(unavailable)?;
    if !(512..=65536).contains(&page_size) || !(page_size as u64).is_power_of_two() { return Err(RecipientJournalError::Corrupt); }
    if pages < 0 || maximum <= 0 || pages > maximum || maximum as u64 > physical_limit / page_size as u64
        || (pages as u64).checked_mul(page_size as u64).map_or(true, |n| n > physical_limit)
    { return Err(RecipientJournalError::Capacity); }
    Ok(())
}

// [REVERSE-ONION-RECIPIENT 2026-10-04 by Codex] Unknown schema is never
// silently adopted, migrated, reset, or repaired by recovery.
fn initialize_schema(connection: &mut Connection, relay: [u8; 32], recipient: [u8; 32], now: u64) -> Result<()> {
    let tx = connection.transaction_with_behavior(TransactionBehavior::Exclusive).map_err(unavailable)?;
    let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(unavailable)?;
    let app: i64 = tx.query_row("PRAGMA application_id", [], |r| r.get(0)).map_err(unavailable)?;
    let count: i64 = tx.query_row("SELECT count(*) FROM (SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 3)", [], |r| r.get(0)).map_err(unavailable)?;
    if version == 0 && app == 0 && count == 0 {
        tx.execute_batch(META_SQL).map_err(unavailable)?;
        tx.execute_batch(ROW_SQL).map_err(unavailable)?;
        tx.execute("INSERT INTO recipient_meta VALUES(1,?1,?2,?3)",
            params![relay.as_slice(), recipient.as_slice(), to_sql(now)?]).map_err(unavailable)?;
        tx.pragma_update(None, "user_version", 1).map_err(unavailable)?;
        tx.pragma_update(None, "application_id", APPLICATION_ID).map_err(unavailable)?;
    } else if version != 1 || app != APPLICATION_ID || count != 2 {
        return Err(RecipientJournalError::Corrupt);
    }
    for (name, expected) in [("recipient_meta", META_SQL), ("recipient_jobs", ROW_SQL)] {
        let sql: String = tx.query_row("SELECT CASE WHEN length(sql)=?2 THEN sql ELSE NULL END FROM sqlite_master WHERE type='table' AND name=?1", params![name, expected.len() as i64], |r| r.get(0)).map_err(|_| RecipientJournalError::Corrupt)?;
        if sql != expected { return Err(RecipientJournalError::Corrupt); }
    }
    tx.commit().map_err(unavailable)
}

fn all_ids(tx: &Transaction<'_>) -> Result<Vec<[u8; 16]>> {
    let mut statement = tx.prepare("SELECT CASE WHEN typeof(claim_id)='blob' AND length(claim_id)=16 THEN claim_id ELSE NULL END FROM recipient_jobs ORDER BY claim_id LIMIT ?1").map_err(unavailable)?;
    let rows = statement.query_map(params![(MAX_ENTRIES + 1) as i64], |r| r.get::<_, Option<Vec<u8>>>(0)).map_err(unavailable)?;
    let mut ids = Vec::new();
    for row in rows {
        let bytes = row.map_err(|_| RecipientJournalError::Corrupt)?.ok_or(RecipientJournalError::Corrupt)?;
        ids.push(bytes.try_into().map_err(|_| RecipientJournalError::Corrupt)?);
        if ids.len() > MAX_ENTRIES { return Err(RecipientJournalError::Capacity); }
    }
    Ok(ids)
}

fn to_sql(value: u64) -> Result<i64> { i64::try_from(value).map_err(|_| RecipientJournalError::Rejected) }
fn unavailable(_: rusqlite::Error) -> RecipientJournalError { RecipientJournalError::Unavailable }
fn changed(rows: usize) -> Result<()> {
    if rows == 1 { Ok(()) } else { Err(RecipientJournalError::Corrupt) }
}

// [REVERSE-ONION-RECOVERY-TESTS 2026-10-04 by Codex] Source-only authoring;
// no execution claimed. Every filesystem fixture is a unique TempDir beneath
// the explicitly approved external-volume test root, never default /tmp.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
    use aeronyx_core::protocol::onion_reply::{
        encode_onion_sealed_response, seal_onion_reply, OnionReplySession,
        ONION_REPLY_RESPONSE_SIZE_CLASSES,
    };

    const NOW: u64 = 1_800_000_000;

    struct Fixture {
        directory: tempfile::TempDir,
        relay: IdentityKeyPair,
        recipient: IdentityKeyPair,
        claim: ReverseOnionFrameV1,
        lease: ReverseOnionFrameV1,
        deadline: u64,
    }

    impl Fixture {
        fn new(deadline: u64) -> Self {
            let directory = tempfile::Builder::new()
                .prefix("r1-recipient-test-")
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let relay = IdentityKeyPair::from_bytes(&[31; 32]).unwrap();
            let recipient = IdentityKeyPair::from_bytes(&[32; 32]).unwrap();
            let claim = ReverseOnionFrameV1::claim(
                relay.public_key_bytes(), [1; 16], NOW, NOW + 30, &recipient,
            ).unwrap();
            let (_, kem) = recipient.to_x25519();
            let envelope = build_onion_envelope(
                &[OnionHop { node_id: recipient.public_key_bytes(), kem_pub: kem.to_bytes() }],
                b"request", [2; 16], 1, NOW, &relay,
            ).unwrap();
            let lease = ReverseOnionFrameV1::lease(
                &claim, &envelope, [3; 16], deadline, NOW, &relay,
            ).unwrap();
            Self { directory, relay, recipient, claim, lease, deadline }
        }

        fn path(&self) -> std::path::PathBuf { self.directory.path().join("recipient.sqlite") }

        fn open(&self, now: u64) -> Result<ReverseOnionRecipientJournal> {
            self.open_with_limits(now, 8, 16 * 1024 * 1024)
        }

        fn open_with_limits(&self, now: u64, max_entries: usize, max_bytes: u64) -> Result<ReverseOnionRecipientJournal> {
            ReverseOnionRecipientJournal::open(&self.path(), self.relay.public_key_bytes(),
                self.recipient.public_key_bytes(), RecipientJournalLimits { max_entries, max_bytes }, now)
        }

        fn ready(&self, journal: &ReverseOnionRecipientJournal) {
            assert_eq!(journal.prepare_poll(&self.claim, NOW).unwrap(), self.claim.encode());
            journal.record_lease(self.claim.claim_id(), &self.lease, self.deadline, NOW + 1).unwrap();
        }

        fn result(&self, now: u64) -> ReverseOnionFrameV1 {
            let (request, _source) = OnionReplySession::prepare_source_sealed(
                self.lease.route_id(), self.recipient.public_key_bytes(),
                ONION_REPLY_RESPONSE_SIZE_CLASSES[0], b"operation".to_vec(),
            ).unwrap();
            let sealed = seal_onion_reply(self.lease.route_id(), &request, b"reply", &self.recipient).unwrap();
            ReverseOnionFrameV1::result(&self.claim, &self.lease,
                &encode_onion_sealed_response(&sealed).unwrap(), self.deadline, now, &self.recipient).unwrap()
        }
    }

    #[test]
    fn exact_poll_replay_survives_restart_and_full_quota_conflicts_do_not_overwrite() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open_with_limits(NOW, 1, 234).unwrap();
        let exact = journal.prepare_poll(&f.claim, NOW).unwrap();
        assert_eq!(journal.prepare_poll(&f.claim, NOW + 1).unwrap(), exact);
        let changed_claim = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
            f.claim.claim_id(), NOW, NOW + 29, &f.recipient).unwrap();
        assert_eq!(journal.prepare_poll(&changed_claim, NOW + 1).err(), Some(RecipientJournalError::Conflict));
        let other = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
            [9; 16], NOW, NOW + 30, &f.recipient).unwrap();
        assert_eq!(journal.prepare_poll(&other, NOW + 1).err(), Some(RecipientJournalError::Capacity));
        drop(journal);
        let journal = f.open_with_limits(NOW + 2, 1, 234).unwrap();
        let page = journal.resume(None, 64, NOW + 2).unwrap();
        assert_eq!(page.items.len(), 1);
        match &page.items[0] {
            RecipientRecovery::Poll { claim_id, exact_bytes } => {
                assert_eq!(*claim_id, f.claim.claim_id());
                assert_eq!(*exact_bytes, exact);
            }
            _ => panic!("expected exact durable poll"),
        }
    }

    #[test]
    fn arm_is_once_only_and_restart_without_result_is_ambiguous() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        let dispatch = journal.arm(f.claim.claim_id(), NOW + 2).unwrap();
        assert_eq!(dispatch.envelope.route_id, f.lease.route_id());
        assert_eq!(dispatch.claim.encode(), f.claim.encode());
        assert_eq!(dispatch.lease.encode(), f.lease.encode());
        assert_eq!(dispatch.route_deadline, f.deadline);
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 2).err(), Some(RecipientJournalError::Ambiguous));
        drop(journal);
        let journal = f.open(NOW + 3).unwrap();
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 3).err(), Some(RecipientJournalError::Ambiguous));
        assert!(matches!(journal.resume(None, 64, NOW + 3).unwrap().items.as_slice(),
            [RecipientRecovery::Ambiguous { .. }]));
        assert_eq!(journal.record_result(f.claim.claim_id(), &f.result(NOW + 3), NOW + 3).err(),
            Some(RecipientJournalError::Ambiguous));
    }

    #[test]
    fn unarmed_lease_restart_uses_execution_deadline_not_claim_freshness() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        drop(journal);
        let journal = f.open(NOW + 31).unwrap();
        journal.record_lease(f.claim.claim_id(), &f.lease, f.deadline, NOW + 31).unwrap();
        assert_eq!(journal.record_lease(f.claim.claim_id(), &f.lease, f.deadline + 1, NOW + 31).err(),
            Some(RecipientJournalError::Conflict));
        assert!(matches!(journal.resume(None, 64, NOW + 31).unwrap().items.as_slice(),
            [RecipientRecovery::LeaseReady { .. }]));
        assert!(journal.arm(f.claim.claim_id(), NOW + 32).is_ok());
    }

    #[test]
    fn stored_result_replays_exact_bytes_after_restart_but_not_after_grace() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        journal.arm(f.claim.claim_id(), NOW + 2).unwrap();
        let result = f.result(NOW + 610);
        journal.record_result(f.claim.claim_id(), &result, NOW + 610).unwrap();
        let exact = result.encode();
        drop(journal);
        let journal = f.open(NOW + 611).unwrap();
        journal.record_result(f.claim.claim_id(), &result, NOW + 611).unwrap();
        match journal.resume(None, 64, NOW + 611).unwrap().items.as_slice() {
            [RecipientRecovery::Result { exact_bytes, .. }] => assert_eq!(*exact_bytes, exact),
            _ => panic!("expected retained exact result"),
        }
        let different = f.result(NOW + 612);
        assert_eq!(journal.record_result(f.claim.claim_id(), &different, NOW + 612).err(),
            Some(RecipientJournalError::Conflict));
        assert!(journal.resume(None, 64, NOW + 900).unwrap().items.is_empty());
    }

    #[test]
    fn clock_rollback_rejects_without_reopening_dispatch() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        assert_eq!(journal.arm(f.claim.claim_id(), NOW).err(), Some(RecipientJournalError::Rejected));
        assert!(journal.arm(f.claim.claim_id(), NOW + 2).is_ok());
        drop(journal);
        assert_eq!(f.open(NOW + 1).err(), Some(RecipientJournalError::Rejected));
        let journal = f.open(NOW + 3).unwrap();
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 3).err(), Some(RecipientJournalError::Ambiguous));
    }

    #[test]
    fn shortened_execution_does_not_shorten_replay_evidence_cleanup() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        assert_eq!(journal.cleanup(64, NOW + 310).unwrap(), 0);
        assert!(journal.arm(f.claim.claim_id(), NOW + 310).is_err());
        assert_eq!(journal.cleanup(64, NOW + 599).unwrap(), 0);
        assert_eq!(journal.cleanup(64, NOW + 600).unwrap(), 1);
        assert!(journal.resume(None, 64, NOW + 600).unwrap().items.is_empty());
    }

    #[test]
    fn recovery_cursor_advances_past_unacknowledged_polls() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        for value in 1..=3 {
            let claim = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
                [value; 16], NOW, NOW + 30, &f.recipient).unwrap();
            journal.prepare_poll(&claim, NOW).unwrap();
        }
        let mut cursor = None;
        let mut ids = Vec::new();
        for _ in 0..3 {
            let page = journal.resume(cursor, 1, NOW).unwrap();
            match page.items.as_slice() {
                [RecipientRecovery::Poll { claim_id, .. }] => ids.push(*claim_id),
                _ => panic!("expected one recovery item"),
            }
            cursor = page.next_after;
        }
        assert_eq!(ids, vec![[1; 16], [2; 16], [3; 16]]);
        assert!(cursor.is_none());
        assert!(journal.resume(None, 65, NOW).is_err());
    }

    #[test]
    fn exclusive_handle_and_oversized_corruption_fail_closed_without_cleanup() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        assert_eq!(f.open(NOW + 1).err(), Some(RecipientJournalError::Busy));
        drop(journal);
        let connection = Connection::open(f.path()).unwrap();
        connection.execute("UPDATE recipient_jobs SET claim=zeroblob(?1)",
            params![(MAX_REVERSE_ONION_FRAME_BYTES + 1) as i64]).unwrap();
        drop(connection);
        assert_eq!(f.open(NOW + 2).err(), Some(RecipientJournalError::Corrupt));
        let connection = Connection::open(f.path()).unwrap();
        let (count, phase): (i64, i64) = connection.query_row(
            "SELECT count(*),min(phase) FROM recipient_jobs", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!((count, phase), (1, Phase::Lease as i64));
    }

    // [REVERSE-ONION-RECIPIENT-DB-BOUNDARY 2026-10-04 by Codex] Authored
    // only. Sparse files avoid allocating the physical byte ceiling in memory.
    fn private_sparse(path: &Path, bytes: u64) {
        std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600)
            .open(path).unwrap().set_len(bytes).unwrap();
    }

    #[test]
    fn recipient_preflight_counts_primary_and_rollback_as_one_observed_budget() {
        let f = Fixture::new(NOW + 600);
        let logical = 1024 * 1024;
        let bound = recipient_physical_limit(&RecipientJournalLimits { max_entries: 8, max_bytes: logical }).unwrap();
        let primary = bound * 3 / 4; let rollback = bound / 2;
        let sidecar = recipient_sidecar(&f.path(), "-journal");
        private_sparse(&f.path(), primary); private_sparse(&sidecar, rollback);
        let parent_mode = std::fs::metadata(f.directory.path()).unwrap().mode();
        assert!(primary < bound && rollback < bound);
        assert_eq!(f.open_with_limits(NOW, 8, logical).err(), Some(RecipientJournalError::Capacity));
        assert_eq!(std::fs::metadata(f.path()).unwrap().len(), primary);
        assert_eq!(std::fs::metadata(&sidecar).unwrap().len(), rollback);
        assert_eq!(std::fs::metadata(f.directory.path()).unwrap().mode(), parent_mode);
        assert_eq!(std::fs::metadata(f.path()).unwrap().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(sidecar).unwrap().mode() & 0o777, 0o600);
    }

    // [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Authored, unexecuted.
    #[test]
    fn relay_authority_survives_restart_without_reviving_execution() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        journal.prepare_poll(&f.claim, NOW).unwrap();
        let proof = f.lease.verify_recipient_lease(&f.claim,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW + 31).unwrap();
        journal.record_relay_lease(proof, NOW + 31).unwrap();
        drop(journal);
        let journal = f.open(NOW + 32).unwrap();
        let armed = journal.arm(f.claim.claim_id(), NOW + 32).unwrap();
        assert_eq!(armed.route_deadline, f.lease.expires_at());
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 33).err(), Some(RecipientJournalError::Ambiguous));
        journal.record_result(f.claim.claim_id(), &f.result(NOW + 601), NOW + 601).unwrap();
        drop(journal);
        let journal = f.open(NOW + 602).unwrap();
        let page = journal.resume(None, 1, NOW + 602).unwrap();
        assert!(matches!(page.items.first(), Some(RecipientRecovery::Result { .. })));
        assert!(journal.arm(f.claim.claim_id(), NOW + 602).is_err());
    }

    #[test]
    fn retained_relay_proof_cannot_admit_expired_execution() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        journal.prepare_poll(&f.claim, NOW).unwrap();
        let proof = f.lease.verify_recipient_lease(&f.claim,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW).unwrap();
        assert_eq!(journal.record_relay_lease(proof, NOW + 10).err(), Some(RecipientJournalError::Rejected));
        let page = journal.resume(None, 1, NOW + 10).unwrap();
        assert!(matches!(page.items.first(), Some(RecipientRecovery::Poll { .. })));
    }

    // [RECIPIENT-HISTORICAL-POLL 2026-10-04 by Codex] Authored, unexecuted.
    #[test]
    fn historical_poll_remains_exact_until_evidence_horizon_not_claim_expiry() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        let bytes = journal.prepare_poll(&f.claim, NOW).unwrap();
        drop(journal);
        let journal = f.open(NOW + 31).unwrap();
        let horizon = f.claim.expires_at() + UNLEASED_EVIDENCE_SECS;
        for now in [NOW + 31, horizon - 1] {
            let page = journal.resume(None, 1, now).unwrap();
            assert_eq!(page.items.len(), 1);
            match &page.items[0] {
                RecipientRecovery::Poll { exact_bytes, .. } => assert_eq!(exact_bytes, &bytes),
                _ => panic!("expected retained poll"),
            }
        }
        assert!(f.lease.verify_recipient_lease(&f.claim, f.relay.public_key_bytes(),
            f.recipient.public_key_bytes(), horizon - 1).is_err());
        assert!(journal.arm(f.claim.claim_id(), horizon - 1).is_err());
        assert!(journal.resume(None, 1, horizon).unwrap().items.is_empty());
        assert_eq!(journal.cleanup(1, horizon).unwrap(), 1);
    }

    #[test]
    fn recipient_absent_primary_oversized_sidecar_refusal_has_no_create_or_chmod() {
        let f = Fixture::new(NOW + 600); let logical = 1024 * 1024;
        let bound = recipient_physical_limit(&RecipientJournalLimits { max_entries: 8, max_bytes: logical }).unwrap();
        let sidecar = recipient_sidecar(&f.path(), "-journal"); private_sparse(&sidecar, bound + 1);
        let parent = std::fs::metadata(f.directory.path()).unwrap();
        let before = std::fs::metadata(&sidecar).unwrap();
        let names_before: Vec<_> = std::fs::read_dir(f.directory.path()).unwrap()
            .map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(f.open_with_limits(NOW, 8, logical).err(), Some(RecipientJournalError::Capacity));
        assert!(!f.path().exists());
        let names_after: Vec<_> = std::fs::read_dir(f.directory.path()).unwrap()
            .map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(names_before, names_after);
        assert_eq!(std::fs::metadata(f.directory.path()).unwrap().mode(), parent.mode());
        let after = std::fs::metadata(sidecar).unwrap();
        assert_eq!(after.mode(), before.mode()); assert_eq!(after.len(), before.len());
    }

    #[test]
    fn recipient_post_commit_aggregate_failure_never_publishes_dispatch_and_poisons() {
        let f = Fixture::new(NOW + 600); let journal = f.open(NOW).unwrap(); f.ready(&journal);
        let primary = std::fs::metadata(f.path()).unwrap().len();
        assert!(primary > 1 && primary < journal.physical_limit);
        journal.post_commit_sidecar_bytes.store(journal.physical_limit - 1, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 2).err(), Some(RecipientJournalError::Unavailable));
        let sidecar = recipient_sidecar(&f.path(), "-journal");
        assert_eq!(std::fs::metadata(&sidecar).unwrap().len(), journal.physical_limit - 1);
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 3).err(), Some(RecipientJournalError::Unavailable));
        assert!(sidecar.exists());
    }

    #[test]
    fn recipient_critical_pragma_drift_is_read_back_before_dispatch_publication() {
        for (pragma, value) in [("fullfsync", 0), ("trusted_schema", 1), ("temp_store", 1), ("busy_timeout", 3)] {
            let f = Fixture::new(NOW + 600); let journal = f.open(NOW).unwrap(); f.ready(&journal);
            {
                let inner = journal.inner.lock().unwrap();
                audit_recipient_pragmas(&inner.connection, journal.physical_limit).unwrap();
                inner.connection.pragma_update(None, pragma, value).unwrap();
            }
            assert_eq!(journal.arm(f.claim.claim_id(), NOW + 2).err(), Some(RecipientJournalError::Unavailable));
            assert_eq!(journal.resume(None, 1, NOW + 2).err(), Some(RecipientJournalError::Unavailable));
            drop(journal);
            let journal = f.open(NOW + 3).unwrap();
            assert_eq!(journal.arm(f.claim.claim_id(), NOW + 3).err(), Some(RecipientJournalError::Ambiguous));
        }
    }

    #[test]
    fn recipient_overflow_and_forbidden_sidecars_fail_closed_without_creation() {
        assert_eq!(recipient_checked_aggregate(u64::MAX, 1, u64::MAX).err(), Some(RecipientJournalError::Capacity));
        assert_eq!(recipient_checked_aggregate(4, 5, 9).unwrap(), 9);
        for suffix in ["-wal", "-shm"] {
            let f = Fixture::new(NOW + 600); let sidecar = recipient_sidecar(&f.path(), suffix);
            private_sparse(&sidecar, 1);
            assert_eq!(f.open(NOW).err(), Some(RecipientJournalError::Corrupt));
            assert!(!f.path().exists()); assert_eq!(std::fs::metadata(sidecar).unwrap().len(), 1);
        }
    }
}
