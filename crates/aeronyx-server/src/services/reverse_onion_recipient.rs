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

use std::path::Path;
use std::sync::Mutex;

use aeronyx_core::protocol::chat::BlindRelayEnvelope;
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, MAX_REVERSE_ONION_FRAME_BYTES,
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
        let physical_limit = limits.max_bytes.checked_mul(2)
            .and_then(|n| n.checked_add((MAX_ENTRIES as u64 + 256) * 4096))
            .ok_or(RecipientJournalError::Rejected)?;
        if metadata.len() > physical_limit { return Err(RecipientJournalError::Capacity); }
        // SAFETY: inode owns a live fd retained for the complete journal lifetime.
        if unsafe { nix::libc::flock(inode.as_raw_fd(), nix::libc::LOCK_EX | nix::libc::LOCK_NB) } != 0 {
            return Err(RecipientJournalError::Busy);
        }
        // Existing rollback journal is allowed only after private-file checks.
        // WAL is not this schema's durability mode; do not silently convert it.
        for suffix in ["-journal", "-wal", "-shm"] {
            let mut name = target.resolved_path.as_os_str().to_os_string();
            name.push(suffix);
            let sidecar = std::path::PathBuf::from(name);
            match std::fs::symlink_metadata(&sidecar) {
                Ok(_) if suffix != "-journal" => return Err(RecipientJournalError::Corrupt),
                Ok(meta) => {
                    verify_private_file(&sidecar, true).map_err(|_| RecipientJournalError::Rejected)?;
                    if meta.len() > physical_limit { return Err(RecipientJournalError::Capacity); }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(RecipientJournalError::Unavailable),
            }
        }
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
        connection.execute_batch("PRAGMA busy_timeout=0; PRAGMA trusted_schema=OFF; PRAGMA temp_store=MEMORY; PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=DELETE; PRAGMA synchronous=EXTRA; PRAGMA fullfsync=ON;")
            .map_err(|_| RecipientJournalError::Unavailable)?;
        let sync: i64 = connection.query_row("PRAGMA synchronous", [], |r| r.get(0)).map_err(unavailable)?;
        let mode: String = connection.query_row("PRAGMA journal_mode", [], |r| r.get(0)).map_err(unavailable)?;
        let locking: String = connection.query_row("PRAGMA locking_mode", [], |r| r.get(0)).map_err(unavailable)?;
        if sync != 3 || mode != "delete" || locking != "exclusive" {
            return Err(RecipientJournalError::Unavailable);
        }
        let page_size: i64 = connection.query_row("PRAGMA page_size", [], |r| r.get(0)).map_err(unavailable)?;
        if !(512..=65536).contains(&page_size) || !(page_size as u64).is_power_of_two() {
            return Err(RecipientJournalError::Corrupt);
        }
        connection.pragma_update(None, "max_page_count", (physical_limit / page_size as u64) as i64)
            .map_err(unavailable)?;
        let integrity: String = connection.query_row("PRAGMA quick_check", [], |row| row.get(0))
            .map_err(|_| RecipientJournalError::Corrupt)?;
        if integrity != "ok" { return Err(RecipientJournalError::Corrupt); }
        initialize_schema(&mut connection, relay, recipient, now)?;
        let journal = Self {
            inner: Mutex::new(Inner { connection, poisoned: false }),
            relay, recipient, limits, _inode_lock: inode, _parent: target.parent,
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
                    Phase::Poll if now < row.claim.expires_at() => Some(RecipientRecovery::Poll {
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
                    #[cfg(unix)]
                    self._parent.sync_all().map_err(|_| RecipientJournalError::Unavailable)?;
                    Ok(value)
                }
                Err(error) => { tx.rollback().map_err(unavailable)?; Err(error) }
            }
        })();
        if matches!(&outcome, Err(RecipientJournalError::Corrupt | RecipientJournalError::Unavailable)) {
            inner.poisoned = true;
        }
        outcome
    }

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
