// ============================================
// File: crates/aeronyx-server/src/services/reverse_onion_recipient.rs
// ============================================
//! Private reverse-onion recipient recovery journal; no network or execution.
//!
//! [REVERSE-ONION-RECIPIENT 2026-10-04 by Codex] Database methods are synchronous.
//! [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Runtime callers use
//! `run_blocking` to serialize owned blocking work, then await its durable
//! result before POST/dispatch. Cancellation never undoes Armed. An
//! Armed row without a stored Result is ambiguous on restart, never executable.
//! This module does not verify source replies, invent wire, or log identifiers.
//!
//! Last Modified: v1.1.0 — Nonmutating preflight and observed aggregate fence.
//! [REVERSE-ONION-RECIPIENT-DB-BOUNDARY 2026-10-04 by Codex] Phase/schema
//! semantics unchanged; a sampled physical bound is not an OS hard quota.
//! [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Typed relay admission
//! stores R's execution expiry, not an independently authenticated source route.

use std::path::Path;
use std::sync::{Arc, Mutex};

use aeronyx_core::protocol::chat::BlindRelayEnvelope;
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, ReverseOnionNoWorkReceiptV1, VerifiedRecipientLease, MAX_REVERSE_ONION_FRAME_BYTES,
    MAX_REVERSE_ONION_CLAIM_BYTES,
};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior};

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const MAX_ENTRIES: usize = 1024;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
// [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex] Charge
// eventual Lease/Result custody before a Claim can leave the journal. Keep
// the charge through recovery; only authenticated retirement/cleanup frees it.
pub(crate) const RECIPIENT_RESERVED_JOB_BYTES: u64 =
    (MAX_REVERSE_ONION_CLAIM_BYTES + 2 * MAX_REVERSE_ONION_FRAME_BYTES + 32 + 16) as u64;
// [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Independent fixture
// arithmetic checks compatibility of the existing on-disk retention bound.
#[cfg(test)]
const UNLEASED_EVIDENCE_SECS: u64 =
    aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_ENVELOPE_LIFETIME_SECS
    + aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_RESULT_RETENTION_SECS;
const APPLICATION_ID: i64 = 0x41585250;
const META_SQL: &str = "CREATE TABLE recipient_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1), relay BLOB NOT NULL, recipient BLOB NOT NULL, clock INTEGER NOT NULL)";
const ROW_SQL_V1: &str = "CREATE TABLE recipient_jobs (claim_id BLOB PRIMARY KEY NOT NULL, route_id BLOB UNIQUE, claim BLOB NOT NULL, lease BLOB, result BLOB, route_deadline INTEGER NOT NULL, retain_until INTEGER NOT NULL, phase INTEGER NOT NULL)";
const ROW_SQL: &str = "CREATE TABLE recipient_jobs (claim_id BLOB PRIMARY KEY NOT NULL, route_id BLOB UNIQUE, claim BLOB NOT NULL, lease BLOB, result BLOB, route_deadline INTEGER NOT NULL, retain_until INTEGER NOT NULL, phase INTEGER NOT NULL, route_origin_commitment BLOB)";

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
    // [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] Local intake
    // closure is distinct from validation, clock and persistence failures.
    #[error("recipient journal intake closed")]
    IntakeClosed,
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
    Poll { claim_id: [u8; 16], exact_bytes: Vec<u8>, route_origin_commitment: [u8; 32] },
    LeaseReady { claim_id: [u8; 16] },
    Result { claim_id: [u8; 16], exact_bytes: Vec<u8>, route_origin_commitment: Option<[u8; 32]> },
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
    pub(crate) route_origin_commitment: Option<[u8; 32]>,
    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] The
    // post-lock Armed clock already committed in recipient_meta. This local
    // one-shot projection is never reconstructed as executable after restart.
    pub(crate) armed_at: u64,
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
    route_origin_commitment: Option<[u8; 32]>,
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
    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Worker scans
    // and late terminal-result writes share one blocking-operation lane.
    // Hold its owned permit inside blocking work through the durability fence.
    blocking_lane: Arc<tokio::sync::Semaphore>,
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
    // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Test-only
    // commit-to-publication fault, shared with terminal lifecycle regressions.
    #[cfg(test)]
    fail_commit_fence: std::sync::atomic::AtomicBool,
}

impl ReverseOnionRecipientJournal {
    #[cfg(test)]
    pub(crate) fn fail_next_commit_fence(&self) {
        self.fail_commit_fence.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Runtime owners
    /// acquire before spawn_blocking, not while occupying a blocking thread.
    /// This lane has one worker caller plus at most four tracked terminal
    /// callers. Stop closes execution intake, not this completion lane: late
    /// results must still persist while their owner drains. Direct synchronous
    /// methods retain their fail-fast mutex and all existing database audits.
    pub(crate) fn blocking_operation_lane(&self) -> Arc<tokio::sync::Semaphore> {
        Arc::clone(&self.blocking_lane)
    }

    /// [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Own the same
    /// journal and permit in the blocking closure. Cancelling the async waiter
    /// after spawn cannot release either while DB work/fencing is unfinished.
    /// The worker and tracked terminal owners bound callers and drain them;
    /// this method neither detaches new worker tasks nor retries DB failures.
    pub(crate) async fn run_blocking<T: Send + 'static>(
        self: &Arc<Self>,
        action: impl FnOnce(&Self) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let permit = self.blocking_operation_lane().acquire_owned().await
            .map_err(|_| RecipientJournalError::Unavailable)?;
        let journal = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            action(&journal)
        }).await.map_err(|_| RecipientJournalError::Ambiguous)?
    }

    // [REVERSE-ROLE-RECOVERY 2026-10-05 by Codex] Missing history is not an
    // empty successful recovery. The standard opener retains all DB audits.
    pub(crate) fn open_existing(path: &Path, relay: [u8; 32], recipient: [u8; 32],
        limits: RecipientJournalLimits, now: u64) -> Result<Self> {
        // [PHALA-EXISTING-CUSTODY-OPEN 2026-10-07 by Codex]
        #[cfg(unix)]
        { Self::open_inner(path, relay, recipient, limits, now, true) }
        #[cfg(not(unix))]
        {
            let _ = (path, relay, recipient, limits, now);
            Err(RecipientJournalError::Rejected)
        }
    }

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
        Self::open_inner(path, relay, recipient, limits, now, false)
    }

    // [PHALA-EXISTING-CUSTODY-OPEN 2026-10-07 by Codex] Only live bootstrap
    // may create a primary; recovery keeps the same migration/crash barrier.
    #[cfg(unix)]
    fn open_inner(path: &Path, relay: [u8; 32], recipient: [u8; 32],
        limits: RecipientJournalLimits, now: u64, existing_only: bool) -> Result<Self> {
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
        let (target, existing_inode) = if existing_only {
            let (target, inode) = super::chat_relay_mailbox::open_existing_private_sqlite_target(path)
                .map_err(|_| RecipientJournalError::Rejected)?;
            (target, Some(inode))
        } else {
            (prepare_private_sqlite_target(path).map_err(|_| RecipientJournalError::Unavailable)?, None)
        };
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| RecipientJournalError::Rejected)?;
        let inode = if let Some(inode) = existing_inode { inode } else {
            std::fs::OpenOptions::new().read(true).write(true)
                .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC | nix::libc::O_NONBLOCK)
                .open(&target.resolved_path).map_err(|_| RecipientJournalError::Unavailable)?
        };
        let metadata = inode.metadata().map_err(|_| RecipientJournalError::Unavailable)?;
        if existing_only && metadata.len() == 0 { return Err(RecipientJournalError::Rejected); }
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
        // [PHALA-SQLITE-INODE-LOCK 2026-10-08 by Codex]
        super::chat_relay_mailbox::lock_private_sqlite_inode(&inode).map_err(|error| match error {
            super::chat_relay_mailbox::AnonymousMailboxStoreError::Busy => RecipientJournalError::Busy,
            _ => RecipientJournalError::Unavailable,
        })?;
        // [PHALA-OWNED-RECOVERY-SCHEMA 2026-10-07 by Codex] Reject a
        // nonempty foreign/blank SQLite primary before writable recovery.
        // Both owned versions remain eligible for the audited v1->v2 path.
        if metadata.len() != 0 {
            super::chat_relay_mailbox::verify_private_sqlite_header(&inode, APPLICATION_ID as u32, &[1, 2])
                .map_err(|_| RecipientJournalError::Corrupt)?;
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
        initialize_schema(&mut connection, relay, recipient, &limits, now)?;
        let journal = Self {
            inner: Mutex::new(Inner { connection, poisoned: false }),
            blocking_lane: Arc::new(tokio::sync::Semaphore::new(1)),
            relay, recipient, limits, _inode_lock: inode, _parent: target.parent,
            db_path: target.resolved_path, physical_limit,
            inode_identity: (metadata.dev(), metadata.ino()),
            parent_identity: (parent_metadata.dev(), parent_metadata.ino()),
            #[cfg(test)]
            post_commit_sidecar_bytes: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            fail_commit_fence: std::sync::atomic::AtomicBool::new(false),
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
    /// and a retained exact retry may outlive Claim freshness. A new Claim
    /// still must be fresh; the relay can only resolve a stale retry from an
    /// already committed row and cannot issue a new Lease for it.
    pub(crate) fn prepare_poll(
        &self,
        claim: &ReverseOnionFrameV1,
        route_origin_commitment: [u8; 32],
        now: u64,
    ) -> Result<Vec<u8>> {
        self.prepare_poll_at(claim, route_origin_commitment, recipient_clock(now))
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Fresh Claims
    // and historical exact retries use the post-lock time, with distinct bounds.
    pub(crate) fn prepare_poll_at(&self, claim: &ReverseOnionFrameV1,
        route_origin_commitment: [u8; 32], refresh_now: impl FnOnce() -> Result<u64>) -> Result<Vec<u8>> {
        self.prepare_poll_with_admission_at(claim, route_origin_commitment, refresh_now, || Ok(()))
    }

    // [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] The worker's
    // stop gate runs after audited SQL/clock/frame validation, immediately
    // before fresh allocation. Exact retained polls remain recovery reads.
    pub(crate) fn prepare_poll_with_admission_at(&self, claim: &ReverseOnionFrameV1,
        route_origin_commitment: [u8; 32], refresh_now: impl FnOnce() -> Result<u64>,
        admit: impl FnOnce() -> Result<()>) -> Result<Vec<u8>> {
        let bytes = claim.encode();
        self.transaction_at(refresh_now, |tx, now| {
            if route_origin_commitment == [0; 32] {
                return Err(RecipientJournalError::Rejected);
            }
            if let Some(row) = self.load(tx, claim.claim_id())? {
                row.claim.require_exact_retry(claim).map_err(|_| RecipientJournalError::Conflict)?;
                if row.phase != Phase::Poll { return Err(RecipientJournalError::Conflict); }
                if row.route_origin_commitment != Some(route_origin_commitment) {
                    return Err(RecipientJournalError::Conflict);
                }
                // [REVERSE-ONION-CLAIM-REPLAY 2026-10-06 by Codex] Only the
                // byte-identical durable poll bypasses Claim freshness.
                if now >= row.retain_until { return Err(RecipientJournalError::Expired); }
                return Ok(row.claim.encode());
            }
            claim.verify_claim(self.relay, self.recipient, claim.issued_at())
                .map_err(|_| RecipientJournalError::Rejected)?;
            if now < claim.issued_at() { return Err(RecipientJournalError::Rejected); }
            if now >= claim.expires_at() { return Err(RecipientJournalError::Expired); }
            // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Persist
            // the same immutable retry bound used before outbound transport.
            let retention = claim.recipient_retry_deadline().map_err(|_| RecipientJournalError::Rejected)?;
            admit()?;
            tx.execute("INSERT INTO recipient_jobs (claim_id,route_id,claim,lease,result,route_deadline,retain_until,phase,route_origin_commitment) VALUES(?1,NULL,?2,NULL,NULL,0,?3,1,?4)",
                params![claim.claim_id().as_slice(), bytes, to_sql(retention)?, route_origin_commitment.as_slice()]).map_err(unavailable)?;
            Ok(bytes.clone())
        })
    }

    /// Retires a poll only after a fresh relay signature binds the exact Claim
    /// and this journal's fixed adjacent identities. TLS/status alone is not proof.
    // [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex]
    pub(crate) fn complete_no_work_poll(
        &self,
        id: [u8; 16],
        exact_claim: &[u8],
        receipt_bytes: &[u8],
        now: u64,
    ) -> Result<()> {
        self.complete_no_work_poll_at(id, exact_claim, receipt_bytes, now, recipient_clock(now))
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Authenticate
    // at receipt time, then enforce its signed bound after SQL wait. Only a
    // genuinely expired authenticated receipt is routine, never invalid bytes.
    pub(crate) fn complete_no_work_poll_at(&self, id: [u8; 16], exact_claim: &[u8],
        receipt_bytes: &[u8], received_at: u64,
        refresh_now: impl FnOnce() -> Result<u64>) -> Result<()> {
        self.transaction_at(refresh_now, |tx, now| {
            let Some(row) = self.load(tx, id)? else {
                return Ok(());
            };
            if row.phase != Phase::Poll || row.claim.encode().as_slice() != exact_claim {
                return Err(RecipientJournalError::Conflict);
            }
            let claim = ReverseOnionFrameV1::decode_for_recovery(exact_claim)
                .map_err(|_| RecipientJournalError::Rejected)?;
            if claim.claim_id() != id {
                return Err(RecipientJournalError::Conflict);
            }
            let receipt = ReverseOnionNoWorkReceiptV1::decode_for_claim(
                receipt_bytes, &claim, self.relay, self.recipient, received_at,
            ).map_err(|_| RecipientJournalError::Rejected)?;
            if now < received_at { return Err(RecipientJournalError::Rejected); }
            if now >= receipt.expires_at() { return Err(RecipientJournalError::Expired); }
            changed(tx.execute(
                "DELETE FROM recipient_jobs WHERE claim_id=?1 AND phase=1",
                params![id.as_slice()],
            ).map_err(unavailable)?)
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
        self.record_relay_lease_at(proof, recipient_clock(now))
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] The worker
    // supplies its live clock here; proof ownership/immutable expiry stay exact.
    pub(crate) fn record_relay_lease_at(&self, proof: VerifiedRecipientLease<'_>,
        refresh_now: impl FnOnce() -> Result<u64>) -> Result<()> {
        let lease = proof.lease();
        if lease.relay() != self.relay || lease.immediate_recipient() != self.recipient {
            return Err(RecipientJournalError::Rejected);
        }
        self.record_bound_lease_at(lease.claim_id(), lease, proof.relay_execution_expiry(), refresh_now)
    }

    // Shared persistence only; the two entry points retain distinct authority
    // contracts. Existing schema, exact retries, retention and phase CAS remain.
    fn record_bound_lease(&self, id: [u8; 16], lease: &ReverseOnionFrameV1,
        authenticated_route_deadline: u64, now: u64) -> Result<()> {
        self.record_bound_lease_at(id, lease, authenticated_route_deadline, recipient_clock(now))
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] A borrowed
    // relay proof is not a timeless lease; recheck inside the audited transaction.
    fn record_bound_lease_at(&self, id: [u8; 16], lease: &ReverseOnionFrameV1,
        authenticated_route_deadline: u64, refresh_now: impl FnOnce() -> Result<u64>) -> Result<()> {
        self.transaction_at(refresh_now, |tx, now| {
            let row = self.load(tx, id)?.ok_or(RecipientJournalError::Rejected)?;
            if let Some(existing) = &row.lease {
                existing.require_exact_retry(lease).map_err(|_| RecipientJournalError::Conflict)?;
                return if row.deadline == authenticated_route_deadline { Ok(()) }
                    else { Err(RecipientJournalError::Conflict) };
            }
            if row.phase != Phase::Poll { return Err(RecipientJournalError::Conflict); }
            lease.verify_lease(&row.claim, authenticated_route_deadline, lease.issued_at())
                .map_err(|_| RecipientJournalError::Rejected)?;
            if now < lease.issued_at() { return Err(RecipientJournalError::Rejected); }
            if now >= lease.expires_at() { return Err(RecipientJournalError::Expired); }
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
        self.arm_at(id, recipient_clock(now))
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] LeaseReady
    // is only a projection; the post-lock callback governs the Armed mutation.
    pub(crate) fn arm_at(&self, id: [u8; 16], refresh_now: impl FnOnce() -> Result<u64>) -> Result<RecipientDispatch> {
        self.arm_with_admission_at(id, refresh_now, || Ok(()))
    }

    // [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] Cancellation
    // during SQL/lease verification must leave LeaseReady recoverable. Once
    // this gate admits, the Armed barrier remains one-shot even on later stop.
    pub(crate) fn arm_with_admission_at(&self, id: [u8; 16],
        refresh_now: impl FnOnce() -> Result<u64>, admit: impl FnOnce() -> Result<()>) -> Result<RecipientDispatch> {
        self.transaction_at(refresh_now, |tx, now| {
            let row = self.load(tx, id)?.ok_or(RecipientJournalError::Rejected)?;
            if row.phase == Phase::Armed || row.phase == Phase::Ambiguous {
                return Err(RecipientJournalError::Ambiguous);
            }
            if row.phase != Phase::Lease { return Err(RecipientJournalError::Conflict); }
            let lease = row.lease.ok_or(RecipientJournalError::Corrupt)?;
            // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] A valid
            // recovery page can age while queued for this transaction. Only
            // forward expiry is routine; rollback/corruption stays fail-closed.
            if now < lease.issued_at() { return Err(RecipientJournalError::Rejected); }
            if now >= lease.expires_at() { return Err(RecipientJournalError::Expired); }
            let envelope = lease.verify_lease(&row.claim, row.deadline, now)
                .map_err(|_| RecipientJournalError::Corrupt)?;
            admit()?;
            changed(tx.execute("UPDATE recipient_jobs SET phase=3 WHERE claim_id=?1 AND phase=2",
                params![id.as_slice()]).map_err(unavailable)?)?;
            Ok(RecipientDispatch {
                envelope,
                claim: row.claim,
                lease,
                route_deadline: row.deadline,
                route_origin_commitment: row.route_origin_commitment,
                armed_at: now,
            })
        })
    }

    /// Reopen an Armed lease only after the adapter proves its capacity gate
    /// rejected before entering the local router. The exact signed Claim and
    /// Lease must still match; restart/recovery never uses this transition.
    // [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex]
    pub(crate) fn restore_lease_after_zero_dispatch(
        &self,
        id: [u8; 16],
        expected_claim: &ReverseOnionFrameV1,
        expected_lease: &ReverseOnionFrameV1,
        expected_deadline: u64,
        now: u64,
    ) -> Result<()> {
        self.restore_lease_after_zero_dispatch_at(id, expected_claim, expected_lease,
            expected_deadline, recipient_clock(now))
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Late
    // zero-send handling must sample under SQL lock, not at adapter return.
    pub(crate) fn restore_lease_after_zero_dispatch_at(&self, id: [u8; 16],
        expected_claim: &ReverseOnionFrameV1, expected_lease: &ReverseOnionFrameV1,
        expected_deadline: u64, refresh_now: impl FnOnce() -> Result<u64>) -> Result<()> {
        // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Zero
        // dispatch proves no effect, not permission to revive an expired lease.
        self.transaction_at(refresh_now, |tx, now| {
            let row = self.load(tx, id)?.ok_or(RecipientJournalError::Rejected)?;
            if row.phase != Phase::Armed || row.result.is_some()
                || row.deadline != expected_deadline
            {
                return Err(RecipientJournalError::Conflict);
            }
            row.claim.require_exact_retry(expected_claim)
                .map_err(|_| RecipientJournalError::Conflict)?;
            row.lease.as_ref().ok_or(RecipientJournalError::Corrupt)?
                .require_exact_retry(expected_lease)
                .map_err(|_| RecipientJournalError::Conflict)?;
            if now < expected_lease.issued_at() { return Err(RecipientJournalError::Rejected); }
            if now >= expected_lease.expires_at() { return Err(RecipientJournalError::Expired); }
            changed(tx.execute(
                "UPDATE recipient_jobs SET phase=2 WHERE claim_id=?1 AND phase=3 AND result IS NULL",
                params![id.as_slice()],
            ).map_err(unavailable)?)
        })
    }

    /// The first exact Result is committed before submission. If capacity or
    /// storage fails, leave Armed/non-reexecutable; never rerun the terminal.
    pub(crate) fn record_result(&self, id: [u8; 16], result: &ReverseOnionFrameV1, now: u64) -> Result<()> {
        // [PHALA-JOURNAL-RESULT-CLOCK 2026-10-07 by Codex] Keep the old
        // scalar entry point without freezing time across SQL wait.
        self.record_result_at(id, result, recipient_clock(now))
    }

    // [PHALA-JOURNAL-RESULT-CLOCK 2026-10-07 by Codex] A completed router
    // operation cannot use a pre-transaction timestamp to persist a new Result
    // after its signed grace. An existing exact retry never rearms the task.
    pub(crate) fn record_result_at(&self, id: [u8; 16], result: &ReverseOnionFrameV1,
        refresh_now: impl FnOnce() -> Result<u64>) -> Result<()> {
        self.transaction_at(refresh_now, |tx, now| {
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
        // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] The same
        // post-lock clock filters the whole bounded recovery page.
        self.transaction_at(recipient_clock(now), |tx, now| {
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
                    Phase::Poll if now < row.retain_until => match row.route_origin_commitment {
                        Some(route_origin_commitment) => Some(RecipientRecovery::Poll {
                            claim_id: id,
                            exact_bytes: row.claim.encode(),
                            route_origin_commitment,
                        }),
                        // [REVERSE-ONION-ORIGIN-MIGRATION 2026-10-06 by Codex]
                        // Legacy Poll rows have no provable original host; never
                        // replay them to a possibly different queue backend.
                        None => Some(RecipientRecovery::Ambiguous { claim_id: id }),
                    },
                    Phase::Lease if now < row.lease.as_ref().ok_or(RecipientJournalError::Corrupt)?.expires_at() =>
                        Some(RecipientRecovery::LeaseReady { claim_id: id }),
                    Phase::Result if now < row.result.as_ref().ok_or(RecipientJournalError::Corrupt)?.expires_at() =>
                        match row.route_origin_commitment {
                            Some(route_origin_commitment) => Some(RecipientRecovery::Result {
                                claim_id: id,
                                exact_bytes: row.result.as_ref().ok_or(RecipientJournalError::Corrupt)?.encode(),
                                route_origin_commitment: Some(route_origin_commitment),
                            }),
                            // A migrated v1 Result remains durable but cannot
                            // be sent to an unproven backend after restart.
                            None => Some(RecipientRecovery::Ambiguous { claim_id: id }),
                        },
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
        // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Cleanup
        // observes the original replay bound at the same transaction clock.
        self.transaction_at(recipient_clock(now), |tx, now| {
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
    // time, reserved byte/count bounds and mutation. Any ambiguous DB error poisons
    // this handle; a reopen audits and retires Armed before returning work.
    fn transaction<T>(&self, now: u64, action: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<T> {
        self.transaction_at(|| Ok(now), |tx, _| action(tx))
    }

    // [PHALA-JOURNAL-RESULT-CLOCK 2026-10-07 by Codex] Preserve the audited
    // transaction, rollback, quota and post-commit fences for refreshed callers.
    // Other existing operations retain their original time contract.
    fn transaction_at<T>(&self, refresh_now: impl FnOnce() -> Result<u64>,
        action: impl FnOnce(&Transaction<'_>, u64) -> Result<T>) -> Result<T> {
        let mut inner = self.inner.try_lock().map_err(|_| RecipientJournalError::Busy)?;
        if inner.poisoned { return Err(RecipientJournalError::Unavailable); }
        let outcome = (|| {
            let tx = inner.connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(unavailable)?;
            let operation = (|| {
                // [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] Keep
                // canonical metadata types after initialization as well.
                let (relay, recipient, clock): (Vec<u8>, Vec<u8>, i64) = tx.query_row(
                    "SELECT relay,recipient,clock FROM recipient_meta WHERE singleton=1 AND typeof(relay)='blob' AND length(relay)=32 AND typeof(recipient)='blob' AND length(recipient)=32 AND typeof(clock)='integer'",
                    [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).map_err(|_| RecipientJournalError::Corrupt)?;
                if relay.as_slice() != self.relay.as_slice()
                    || recipient.as_slice() != self.recipient.as_slice() || clock < 0
                {
                    return Err(RecipientJournalError::Corrupt);
                }
                self.audit_bounds(&tx)?;
                let now = refresh_now()?;
                if to_sql(now)? < clock { return Err(RecipientJournalError::Rejected); }
                let value = action(&tx, now)?;
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
            if self.fail_commit_fence.swap(false, std::sync::atomic::Ordering::SeqCst) {
                return Err(RecipientJournalError::Unavailable);
            }
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
        // [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] Initialization
        // uses the same quota audit before committing a legacy migration.
        Self::audit_bounds_for(tx, &self.limits)
    }

    fn audit_bounds_for(tx: &Transaction<'_>, limits: &RecipientJournalLimits) -> Result<()> {
        // [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex]
        // Existing rows consume the same full budget as a new Poll, including
        // migrated/ambiguous rows. Preserve actual-byte accounting as a second
        // bound so malformed oversized custody cannot hide behind reservation.
        let (count, bytes): (i64, i64) = tx.query_row(
            "SELECT count(*),coalesce(sum(max(?2,c+l+r+o+t)),0) FROM (SELECT length(claim) AS c,coalesce(length(lease),0) AS l,coalesce(length(result),0) AS r,coalesce(length(route_origin_commitment),0) AS o,coalesce(length(route_id),0) AS t FROM recipient_jobs LIMIT ?1)",
            params![(MAX_ENTRIES + 1) as i64, RECIPIENT_RESERVED_JOB_BYTES as i64],
            |r| Ok((r.get(0)?, r.get(1)?))).map_err(|_| RecipientJournalError::Corrupt)?;
        if count < 0 || bytes < 0 { return Err(RecipientJournalError::Corrupt); }
        if count as usize > limits.max_entries || bytes as u64 > limits.max_bytes {
            return Err(RecipientJournalError::Capacity);
        }
        Ok(())
    }

    fn load(&self, tx: &Transaction<'_>, id: [u8; 16]) -> Result<Option<Record>> {
        // [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] Share the full
        // signed-row decoder with the pre-commit migration audit.
        Self::load_for(tx, id, self.relay, self.recipient)
    }

    fn load_for(tx: &Transaction<'_>, id: [u8; 16], relay: [u8; 32],
        recipient: [u8; 32]) -> Result<Option<Record>> {
        let lengths: Option<(i64, Option<i64>, Option<i64>, Option<i64>)> = tx.query_row(
            "SELECT length(claim),length(lease),length(result),length(route_id) FROM recipient_jobs WHERE claim_id=?1",
            params![id.as_slice()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))
            .optional().map_err(|_| RecipientJournalError::Corrupt)?;
        let Some((claim_len, lease_len, result_len, route_len)) = lengths else { return Ok(None); };
        if claim_len != 234 || [lease_len, result_len].into_iter().flatten()
            .any(|n| n < 234 || n > MAX_REVERSE_ONION_FRAME_BYTES as i64)
            || route_len.is_some_and(|n| n != 16)
        { return Err(RecipientJournalError::Corrupt); }
        let origin_shape: (String, Option<i64>) = tx.query_row(
            "SELECT typeof(route_origin_commitment),length(route_origin_commitment) FROM recipient_jobs WHERE claim_id=?1",
            params![id.as_slice()], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|_| RecipientJournalError::Corrupt)?;
        let route_origin_commitment = match origin_shape {
            (kind, Some(32)) if kind == "blob" => {
                let bytes: Vec<u8> = tx.query_row(
                    "SELECT route_origin_commitment FROM recipient_jobs WHERE claim_id=?1",
                    params![id.as_slice()], |r| r.get(0),
                ).map_err(|_| RecipientJournalError::Corrupt)?;
                let commitment: [u8; 32] = bytes.try_into().map_err(|_| RecipientJournalError::Corrupt)?;
                if commitment == [0; 32] { return Err(RecipientJournalError::Corrupt); }
                Some(commitment)
            }
            (kind, None) if kind == "null" => None,
            _ => return Err(RecipientJournalError::Corrupt),
        };
        let (claim, lease, result, route, deadline, retain, phase):
            (Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>, i64, i64, i64) = tx.query_row(
                "SELECT claim,lease,result,route_id,route_deadline,retain_until,phase FROM recipient_jobs WHERE claim_id=?1",
                params![id.as_slice()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?)))
                .map_err(|_| RecipientJournalError::Corrupt)?;
        let decode = |bytes: &[u8]| ReverseOnionFrameV1::decode_for_recovery(bytes)
            .map_err(|_| RecipientJournalError::Corrupt);
        let claim = decode(&claim)?;
        claim.verify_claim(relay, recipient, claim.issued_at())
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
            || retain as u64 != claim.recipient_retry_deadline().map_err(|_| RecipientJournalError::Corrupt)?
        { return Err(RecipientJournalError::Corrupt); }
        if result.is_some() != (phase == Phase::Result) { return Err(RecipientJournalError::Corrupt); }
        Ok(Some(Record {
            claim, lease, result, deadline: deadline as u64,
            retain_until: retain as u64, phase, route_origin_commitment,
        }))
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
fn initialize_schema(connection: &mut Connection, relay: [u8; 32], recipient: [u8; 32],
    limits: &RecipientJournalLimits, now: u64) -> Result<()> {
    let tx = connection.transaction_with_behavior(TransactionBehavior::Exclusive).map_err(unavailable)?;
    let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(unavailable)?;
    let app: i64 = tx.query_row("PRAGMA application_id", [], |r| r.get(0)).map_err(unavailable)?;
    let count: i64 = tx.query_row("SELECT count(*) FROM (SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 3)", [], |r| r.get(0)).map_err(unavailable)?;
    if version == 0 && app == 0 && count == 0 {
        tx.execute_batch(META_SQL).map_err(unavailable)?;
        tx.execute_batch(ROW_SQL).map_err(unavailable)?;
        tx.execute("INSERT INTO recipient_meta VALUES(1,?1,?2,?3)",
            params![relay.as_slice(), recipient.as_slice(), to_sql(now)?]).map_err(unavailable)?;
        tx.pragma_update(None, "user_version", 2).map_err(unavailable)?;
        tx.pragma_update(None, "application_id", APPLICATION_ID).map_err(unavailable)?;
    } else if version == 1 && app == APPLICATION_ID && count == 2 {
        let old_sql: String = tx.query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='recipient_jobs'",
            [], |row| row.get(0),
        ).map_err(|_| RecipientJournalError::Corrupt)?;
        if old_sql != ROW_SQL_V1 { return Err(RecipientJournalError::Corrupt); }
        tx.execute_batch("ALTER TABLE recipient_jobs ADD COLUMN route_origin_commitment BLOB")
            .map_err(unavailable)?;
        tx.pragma_update(None, "user_version", 2).map_err(unavailable)?;
    } else if version != 2 || app != APPLICATION_ID || count != 2 {
        return Err(RecipientJournalError::Corrupt);
    }
    for (name, expected) in [("recipient_meta", META_SQL), ("recipient_jobs", ROW_SQL)] {
        let sql: String = tx.query_row("SELECT CASE WHEN length(sql)=?2 THEN sql ELSE NULL END FROM sqlite_master WHERE type='table' AND name=?1", params![name, expected.len() as i64], |r| r.get(0)).map_err(|_| RecipientJournalError::Corrupt)?;
        if sql != expected { return Err(RecipientJournalError::Corrupt); }
    }
    // [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] Schema migration
    // is not independent of custody admission. Reject owner/clock/row/quota
    // failures in THIS transaction, so v1 remains v1 on failed recovery.
    let (stored_relay, stored_recipient, clock): (Vec<u8>, Vec<u8>, i64) = tx.query_row(
        "SELECT relay,recipient,clock FROM recipient_meta WHERE singleton=1 AND typeof(relay)='blob' AND length(relay)=32 AND typeof(recipient)='blob' AND length(recipient)=32 AND typeof(clock)='integer'",
        [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).map_err(|_| RecipientJournalError::Corrupt)?;
    if stored_relay.as_slice() != relay.as_slice()
        || stored_recipient.as_slice() != recipient.as_slice() || clock < 0
    {
        return Err(RecipientJournalError::Corrupt);
    }
    if to_sql(now)? < clock { return Err(RecipientJournalError::Rejected); }
    ReverseOnionRecipientJournal::audit_bounds_for(&tx, limits)?;
    for id in all_ids(&tx)? {
        ReverseOnionRecipientJournal::load_for(&tx, id, relay, recipient)?
            .ok_or(RecipientJournalError::Corrupt)?;
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

// [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] One monotonic
// anchor covers each synchronous call's SQL wait without restarting its clock.
fn recipient_clock(anchor: u64) -> impl FnOnce() -> Result<u64> {
    let started = std::time::Instant::now();
    move || anchor.checked_add(started.elapsed().as_secs()).ok_or(RecipientJournalError::Unavailable)
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
    // [REVERSE-ONION-ORIGIN-BINDING 2026-10-06 by Codex] Stable synthetic
    // origin commitment for journal state-machine fixtures.
    const TEST_ROUTE_ORIGIN: [u8; 32] = [9; 32];

    // [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] Authored only:
    // the full existing-only opener must roll back schema changes when any
    // owner, clock, signed row or quota audit rejects the legacy primary.
    #[test]
    fn rejected_legacy_open_preserves_version_and_exact_custody() {
        use std::os::unix::fs::PermissionsExt;
        for scenario in 0..8 {
            let f = Fixture::new(NOW + 600);
            let path = f.path();
            let mut claim = f.claim.encode();
            if scenario == 3 { *claim.last_mut().unwrap() ^= 1; }
            {
                let connection = Connection::open(&path).unwrap();
                connection.execute_batch(META_SQL).unwrap();
                connection.execute_batch(ROW_SQL_V1).unwrap();
                connection.execute("INSERT INTO recipient_meta VALUES(1,?1,?2,?3)",
                    params![f.relay.public_key_bytes().as_slice(),
                        f.recipient.public_key_bytes().as_slice(), to_sql(NOW).unwrap()]).unwrap();
                connection.execute("INSERT INTO recipient_jobs VALUES(?1,NULL,?2,NULL,NULL,0,?3,1)",
                    params![f.claim.claim_id().as_slice(), claim.as_slice(),
                        to_sql(f.claim.recipient_retry_deadline().unwrap()).unwrap()]).unwrap();
                match scenario {
                    5 => { connection.execute("UPDATE recipient_meta SET relay=?1",
                        params![hex::encode(f.relay.public_key_bytes())]).unwrap(); }
                    6 => { connection.execute("UPDATE recipient_meta SET clock=?1",
                        params![NOW as f64 + 0.5]).unwrap(); }
                    7 => { connection.execute("UPDATE recipient_meta SET clock=-1", []).unwrap(); }
                    _ => {}
                }
                connection.pragma_update(None, "application_id", APPLICATION_ID).unwrap();
                connection.pragma_update(None, "user_version", 1).unwrap();
            }
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            let mut relay = f.relay.public_key_bytes();
            let mut recipient = f.recipient.public_key_bytes();
            if scenario == 0 { relay = IdentityKeyPair::from_bytes(&[33; 32]).unwrap().public_key_bytes(); }
            if scenario == 1 { recipient = IdentityKeyPair::from_bytes(&[34; 32]).unwrap().public_key_bytes(); }
            let attempted_at = if scenario == 2 { NOW - 1 } else { NOW + 1 };
            // [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex]
            // Actual legacy bytes fit; only the missing future budget rejects.
            let limits = RecipientJournalLimits { max_entries: 8,
                max_bytes: if scenario == 4 { RECIPIENT_RESERVED_JOB_BYTES - 1 }
                    else { 16 * 1024 * 1024 } };
            let error = ReverseOnionRecipientJournal::open_existing(
                &path, relay, recipient, limits, attempted_at,
            ).err().expect("invalid legacy open must fail");
            let expected = match scenario {
                2 => RecipientJournalError::Rejected,
                4 => RecipientJournalError::Capacity,
                _ => RecipientJournalError::Corrupt,
            };
            assert_eq!(error, expected, "scenario {scenario}");
            {
                let connection = Connection::open(&path).unwrap();
                let version: i64 = connection.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
                let schema: String = connection.query_row(
                    "SELECT sql FROM sqlite_master WHERE name='recipient_jobs'", [], |r| r.get(0)).unwrap();
                let stored: Vec<u8> = connection.query_row("SELECT claim FROM recipient_jobs", [], |r| r.get(0)).unwrap();
                let phase: i64 = connection.query_row("SELECT phase FROM recipient_jobs", [], |r| r.get(0)).unwrap();
                assert_eq!(version, 1);
                assert_eq!(schema, ROW_SQL_V1);
                assert_eq!(stored, claim);
                assert_eq!(phase, 1);
                // Repair only this deliberately invalid fixture, then require
                // the same primary's valid migration as a positive control.
                connection.execute("UPDATE recipient_meta SET relay=?1,recipient=?2,clock=?3",
                    params![f.relay.public_key_bytes().as_slice(),
                        f.recipient.public_key_bytes().as_slice(), to_sql(NOW).unwrap()]).unwrap();
                connection.execute("UPDATE recipient_jobs SET claim=?1", params![f.claim.encode()]).unwrap();
            }
            let recovered = f.open_existing(NOW + 1).unwrap();
            assert!(matches!(recovered.resume(None, 1, NOW + 1).unwrap().items.as_slice(),
                [RecipientRecovery::Ambiguous { claim_id }] if *claim_id == f.claim.claim_id()));
        }
    }

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
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
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

        // [PHALA-OWNED-RECOVERY-SCHEMA 2026-10-07 by Codex] Exercise the
        // production recovery opener for all retained v1 phase migrations.
        fn open_existing(&self, now: u64) -> Result<ReverseOnionRecipientJournal> {
            ReverseOnionRecipientJournal::open_existing(&self.path(), self.relay.public_key_bytes(),
                self.recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes: 16 * 1024 * 1024 }, now)
        }

        fn ready(&self, journal: &ReverseOnionRecipientJournal) {
            assert_eq!(journal.prepare_poll(&self.claim, TEST_ROUTE_ORIGIN, NOW).unwrap(), self.claim.encode());
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

    // [PHALA-EXISTING-CUSTODY-OPEN 2026-10-07 by Codex] Authored, not run.
    #[test]
    fn recovery_open_never_bootstraps_missing_or_empty_recipient_history() {
        let f = Fixture::new(NOW + 600);
        let reopen = |path: &Path, now| ReverseOnionRecipientJournal::open_existing(
            path, f.relay.public_key_bytes(), f.recipient.public_key_bytes(),
            RecipientJournalLimits { max_entries: 8, max_bytes: 16 * 1024 * 1024 }, now);
        let nested = f.path().parent().unwrap().join("missing/recipient.sqlite");
        assert!(reopen(&nested, NOW).is_err());
        assert!(!nested.parent().unwrap().exists());
        assert!(reopen(&f.path(), NOW).is_err());
        assert!(!f.path().exists());
        let empty = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600)
            .open(f.path()).unwrap();
        drop(empty);
        assert!(reopen(&f.path(), NOW).is_err());
        assert_eq!(std::fs::metadata(f.path()).unwrap().len(), 0);
        std::fs::remove_file(f.path()).unwrap();
        let journal = f.open(NOW).unwrap();
        let exact = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
        drop(journal);
        let journal = reopen(&f.path(), NOW + 1).unwrap();
        let page = journal.resume(None, 1, NOW + 1).unwrap();
        assert!(matches!(page.items.as_slice(), [RecipientRecovery::Poll { exact_bytes, .. }]
            if *exact_bytes == exact));
    }

    // [PHALA-OWNED-RECOVERY-SCHEMA 2026-10-07 by Codex] Authored, not run.
    #[test]
    fn recipient_header_gate_preserves_foreign_or_unowned_sqlite_bytes() {
        use std::os::unix::fs::PermissionsExt;
        for (version, app) in [(0, 0), (1, 0x41585350), (3, APPLICATION_ID)] {
            let f = Fixture::new(NOW + 600);
            let connection = Connection::open(f.path()).unwrap();
            connection.execute_batch("CREATE TABLE foreign_state(value BLOB); DROP TABLE foreign_state;").unwrap();
            connection.pragma_update(None, "application_id", app).unwrap();
            connection.pragma_update(None, "user_version", version).unwrap();
            drop(connection);
            std::fs::set_permissions(f.path(), std::fs::Permissions::from_mode(0o600)).unwrap();
            let before = std::fs::read(f.path()).unwrap();
            assert!(!before.is_empty());
            assert_eq!(f.open(NOW).err(), Some(RecipientJournalError::Corrupt));
            assert_eq!(std::fs::read(f.path()).unwrap(), before);
            assert_eq!(ReverseOnionRecipientJournal::open_existing(&f.path(),
                f.relay.public_key_bytes(), f.recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 8, max_bytes: 16 * 1024 * 1024 }, NOW).err(),
                Some(RecipientJournalError::Corrupt));
            assert_eq!(std::fs::read(f.path()).unwrap(), before);
        }
    }

    #[test]
    fn exact_poll_replay_survives_restart_and_full_quota_conflicts_do_not_overwrite() {
        let f = Fixture::new(NOW + 600);
        // [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex]
        // Fund one complete job, not just its initial Claim frame.
        let journal = f.open_with_limits(NOW, 1, RECIPIENT_RESERVED_JOB_BYTES).unwrap();
        let exact = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
        assert_eq!(journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW + 1).unwrap(), exact);
        assert_eq!(
            journal.prepare_poll(&f.claim, [10; 32], NOW + 1).err(),
            Some(RecipientJournalError::Conflict),
        );
        let changed_claim = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
            f.claim.claim_id(), NOW, NOW + 29, &f.recipient).unwrap();
        assert_eq!(journal.prepare_poll(&changed_claim, TEST_ROUTE_ORIGIN, NOW + 1).err(), Some(RecipientJournalError::Conflict));
        let other = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
            [9; 16], NOW, NOW + 30, &f.recipient).unwrap();
        assert_eq!(journal.prepare_poll(&other, TEST_ROUTE_ORIGIN, NOW + 1).err(), Some(RecipientJournalError::Capacity));
        drop(journal);
        let journal = f.open_with_limits(NOW + 2, 1, RECIPIENT_RESERVED_JOB_BYTES).unwrap();
        let page = journal.resume(None, 64, NOW + 2).unwrap();
        assert_eq!(page.items.len(), 1);
        match &page.items[0] {
            RecipientRecovery::Poll { claim_id, exact_bytes, route_origin_commitment } => {
                assert_eq!(*claim_id, f.claim.claim_id());
                assert_eq!(*exact_bytes, exact);
                assert_eq!(*route_origin_commitment, TEST_ROUTE_ORIGIN);
            }
            _ => panic!("expected exact durable poll"),
        }
    }

    // [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex]
    // Authored, not run: byte capacity, independently of the item ceiling,
    // is charged before the first outbound Poll and survives exact recovery.
    #[test]
    fn poll_reservation_precedes_transport_and_survives_reopen() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open_with_limits(NOW, 8, RECIPIENT_RESERVED_JOB_BYTES - 1).unwrap();
        assert_eq!(journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).err(),
            Some(RecipientJournalError::Capacity));
        assert!(journal.resume(None, 64, NOW).unwrap().items.is_empty());
        drop(journal);
        let journal = f.open_with_limits(NOW, 8, RECIPIENT_RESERVED_JOB_BYTES).unwrap();
        let exact = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
        let other = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
            [9; 16], NOW, NOW + 30, &f.recipient).unwrap();
        assert_eq!(journal.prepare_poll(&other, TEST_ROUTE_ORIGIN, NOW + 1).err(),
            Some(RecipientJournalError::Capacity));
        assert_eq!(journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW + 1).unwrap(), exact);
        drop(journal);
        assert_eq!(f.open_with_limits(NOW + 2, 8, RECIPIENT_RESERVED_JOB_BYTES - 1).err(),
            Some(RecipientJournalError::Capacity));
        let journal = f.open_with_limits(NOW + 2, 8, RECIPIENT_RESERVED_JOB_BYTES).unwrap();
        assert!(matches!(journal.resume(None, 64, NOW + 2).unwrap().items.as_slice(),
            [RecipientRecovery::Poll { exact_bytes, .. }] if exact_bytes == &exact));
        assert_eq!(journal.prepare_poll(&other, TEST_ROUTE_ORIGIN, NOW + 2).err(),
            Some(RecipientJournalError::Capacity));
    }

    // [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex]
    // Authored, not run: refusing competing Polls leaves enough room for the
    // largest Result after execution; its evidence keeps the original charge.
    #[test]
    fn reserved_result_growth_keeps_exact_recovery_until_cleanup() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open_with_limits(NOW, 8, RECIPIENT_RESERVED_JOB_BYTES).unwrap();
        f.ready(&journal);
        let other = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
            [9; 16], NOW, NOW + 30, &f.recipient).unwrap();
        assert_eq!(journal.prepare_poll(&other, TEST_ROUTE_ORIGIN, NOW + 1).err(),
            Some(RecipientJournalError::Capacity));
        journal.arm(f.claim.claim_id(), NOW + 2).unwrap();
        let (request, _source) = OnionReplySession::prepare_source_sealed(
            f.lease.route_id(), f.recipient.public_key_bytes(),
            *ONION_REPLY_RESPONSE_SIZE_CLASSES.last().unwrap(), b"operation".to_vec(),
        ).unwrap();
        let sealed = seal_onion_reply(f.lease.route_id(), &request, b"reply", &f.recipient).unwrap();
        let result = ReverseOnionFrameV1::result(&f.claim, &f.lease,
            &encode_onion_sealed_response(&sealed).unwrap(), f.deadline, NOW + 3, &f.recipient).unwrap();
        assert_eq!(result.encode().len(), MAX_REVERSE_ONION_FRAME_BYTES);
        journal.record_result(f.claim.claim_id(), &result, NOW + 3).unwrap();
        assert_eq!(journal.prepare_poll(&other, TEST_ROUTE_ORIGIN, NOW + 3).err(),
            Some(RecipientJournalError::Capacity));
        drop(journal);
        let journal = f.open_with_limits(NOW + 4, 8, RECIPIENT_RESERVED_JOB_BYTES).unwrap();
        assert!(matches!(journal.resume(None, 64, NOW + 4).unwrap().items.as_slice(),
            [RecipientRecovery::Result { exact_bytes, .. }] if exact_bytes == &result.encode()));
        journal.record_result(f.claim.claim_id(), &result, NOW + 4).unwrap();
        // [PHALA-REVERSE-FIXTURE-REPAIR 2026-10-08 by Codex] A stored
        // Result retains its signed result horizon, not a Poll retry horizon.
        let retained_until = result.expires_at();
        assert_eq!(journal.cleanup(64, retained_until - 1).unwrap(), 0);
        assert_eq!(journal.cleanup(64, retained_until).unwrap(), 1);
        let next = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(), [9; 16],
            retained_until, retained_until + 30, &f.recipient).unwrap();
        assert_eq!(journal.prepare_poll(&next, TEST_ROUTE_ORIGIN, retained_until).unwrap(), next.encode());
    }

    // [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex] Authored, not run:
    // only a relay-signed receipt for the exact Claim may retire it.
    #[test]
    fn signed_no_work_retires_exact_poll_and_preserves_other_phases() {
        let f = Fixture::new(NOW + 600);
        // [PHALA-RECIPIENT-CAPACITY-RESERVATION 2026-10-08 by Codex]
        let journal = f.open_with_limits(NOW, 1, RECIPIENT_RESERVED_JOB_BYTES).unwrap();
        let exact = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
        let receipt = ReverseOnionNoWorkReceiptV1::issue_no_work(
            &f.claim, NOW + 1, &f.relay,
        ).unwrap().encode();
        let mut forged_receipt = receipt.clone();
        *forged_receipt.last_mut().unwrap() ^= 1;
        assert_eq!(
            journal.complete_no_work_poll(
                f.claim.claim_id(), &exact, &forged_receipt, NOW + 2,
            ).err(),
            Some(RecipientJournalError::Rejected),
        );
        assert!(matches!(
            journal.resume(None, 1, NOW + 2).unwrap().items.as_slice(),
            [RecipientRecovery::Poll { .. }],
        ));
        let mut altered = exact.clone();
        *altered.last_mut().unwrap() ^= 1;
        assert_eq!(
            journal.complete_no_work_poll(f.claim.claim_id(), &altered, &receipt, NOW + 2).err(),
            Some(RecipientJournalError::Conflict),
        );
        assert!(matches!(
            journal.resume(None, 1, NOW + 2).unwrap().items.as_slice(),
            [RecipientRecovery::Poll { .. }],
        ));
        journal.complete_no_work_poll(f.claim.claim_id(), &exact, &receipt, NOW + 2).unwrap();
        journal.complete_no_work_poll(f.claim.claim_id(), &exact, &receipt, NOW + 3).unwrap();
        assert!(journal.resume(None, 1, NOW + 3).unwrap().items.is_empty());

        let next = ReverseOnionFrameV1::claim(
            f.relay.public_key_bytes(), [9; 16], NOW + 4, NOW + 34, &f.recipient,
        ).unwrap();
        assert_eq!(journal.prepare_poll(&next, TEST_ROUTE_ORIGIN, NOW + 4).unwrap(), next.encode());

        let other = Fixture::new(NOW + 600);
        let journal = other.open(NOW).unwrap();
        other.ready(&journal);
        assert_eq!(
            journal.complete_no_work_poll(
                other.claim.claim_id(), &other.claim.encode(), &receipt, NOW + 2,
            ).err(),
            Some(RecipientJournalError::Conflict),
        );
    }

    // [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored,
    // not run: a post-commit fence error publishes no success and poisons this
    // handle. Reopening audits the actual committed Result, never re-arms it.
    #[test]
    fn failed_result_fence_requires_audited_exact_recovery_without_reexecution() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        journal.arm(f.claim.claim_id(), NOW + 2).unwrap();
        let result = f.result(NOW + 3);
        let exact = result.encode();
        journal.fail_next_commit_fence();
        assert_eq!(journal.record_result(f.claim.claim_id(), &result, NOW + 3).err(),
            Some(RecipientJournalError::Unavailable));
        assert_eq!(journal.resume(None, 64, NOW + 3).err(), Some(RecipientJournalError::Unavailable));
        drop(journal);
        let reopened = f.open(NOW + 4).unwrap();
        let page = reopened.resume(None, 64, NOW + 4).unwrap();
        let [RecipientRecovery::Result { exact_bytes, .. }] = page.items.as_slice() else {
            panic!("audited committed result must remain exact retry evidence");
        };
        assert_eq!(exact_bytes, &exact);
        assert!(reopened.arm(f.claim.claim_id(), NOW + 4).is_err());
        reopened.record_result(f.claim.claim_id(), &result, NOW + 4).unwrap();
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

    // [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex] Authored, not run:
    // only an exact in-process zero-dispatch proof may reopen the same Lease.
    #[test]
    fn known_zero_dispatch_restores_only_the_exact_live_lease() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        let armed = journal.arm(f.claim.claim_id(), NOW + 2).unwrap();
        journal.restore_lease_after_zero_dispatch(
            f.claim.claim_id(), &armed.claim, &armed.lease, armed.route_deadline, NOW + 3,
        ).unwrap();
        // [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex] A result cannot
        // be accepted while the exact lease is back in the unarmed phase.
        assert_eq!(
            journal.record_result(f.claim.claim_id(), &f.result(NOW + 3), NOW + 3).err(),
            Some(RecipientJournalError::Ambiguous),
        );
        assert!(matches!(journal.resume(None, 64, NOW + 3).unwrap().items.as_slice(),
            [RecipientRecovery::LeaseReady { .. }]));

        let retried = journal.arm(f.claim.claim_id(), NOW + 4).unwrap();
        assert_eq!(retried.claim.encode(), armed.claim.encode());
        assert_eq!(retried.lease.encode(), armed.lease.encode());
        assert_eq!(retried.route_deadline, armed.route_deadline);

        let other = Fixture::new(NOW + 600);
        let other_journal = other.open(NOW).unwrap();
        other.ready(&other_journal);
        let other_armed = other_journal.arm(other.claim.claim_id(), NOW + 2).unwrap();
        // [PHALA-REVERSE-FIXTURE-REPAIR 2026-10-08 by Codex] Fixture
        // identities and timestamps are deterministic; use a distinct Claim.
        let wrong_claim = ReverseOnionFrameV1::claim(other.relay.public_key_bytes(),
            [99; 16], NOW, NOW + 30, &other.recipient).unwrap();
        assert_eq!(
            other_journal.restore_lease_after_zero_dispatch(
                other.claim.claim_id(), &wrong_claim, &other_armed.lease,
                other_armed.route_deadline, NOW + 3,
            ).err(),
            Some(RecipientJournalError::Conflict),
        );
        assert!(matches!(other_journal.resume(None, 64, NOW + 3).unwrap().items.as_slice(),
            [RecipientRecovery::Ambiguous { .. }]));
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

    // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Authored, not run:
    // aging a projected lease cannot arm it or shorten its evidence retention.
    #[test]
    fn lease_expiry_between_page_and_arm_is_nonmutating_and_does_not_poison() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        assert!(matches!(journal.resume(None, 64, NOW + 9).unwrap().items.as_slice(),
            [RecipientRecovery::LeaseReady { .. }]));
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 10).err(), Some(RecipientJournalError::Expired));
        assert_eq!(journal.cleanup(64, NOW + 10).unwrap(), 0);
        {
            let inner = journal.inner.lock().unwrap();
            let (phase, lease): (i64, Vec<u8>) = inner.connection.query_row(
                "SELECT phase,lease FROM recipient_jobs WHERE claim_id=?1",
                params![f.claim.claim_id().as_slice()], |row| Ok((row.get(0)?, row.get(1)?)),
            ).unwrap();
            assert_eq!(phase, Phase::Lease as i64);
            assert_eq!(lease, f.lease.encode());
        }
        assert_eq!(journal.cleanup(64, NOW + 599).unwrap(), 0);
        assert_eq!(journal.cleanup(64, NOW + 600).unwrap(), 1);
    }

    // [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Calibrate the
    // normal-expiry classification against a known invalid signed row.
    #[test]
    fn invalid_expired_lease_is_corruption_not_routine_expiry() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        let mut invalid = f.lease.encode();
        *invalid.last_mut().unwrap() ^= 1;
        journal.inner.lock().unwrap().connection.execute(
            "UPDATE recipient_jobs SET lease=?1 WHERE claim_id=?2",
            params![invalid, f.claim.claim_id().as_slice()],
        ).unwrap();
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 10).err(), Some(RecipientJournalError::Corrupt));
        assert_eq!(journal.cleanup(64, NOW + 600).err(), Some(RecipientJournalError::Unavailable));
    }

    // [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Serialization
    // belongs to runtime owners; synchronous calls still fail fast on actual
    // mutex contention instead of silently bypassing audits or retrying writes.
    #[test]
    fn direct_journal_contention_remains_busy_and_lane_is_shared() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        let lane = journal.blocking_operation_lane();
        assert!(Arc::ptr_eq(&lane, &journal.blocking_operation_lane()));
        assert_eq!(lane.available_permits(), 1);
        let held = journal.inner.lock().unwrap();
        assert_eq!(journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).err(),
            Some(RecipientJournalError::Busy));
        drop(held);
        assert!(journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).is_ok());
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
        // [REVERSE-ONION-RESULT-CUSTODY-ECHO 2026-10-05 by Codex] The local
        // journal has no remote-ACK bit: after a crash it replays exact bytes.
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

    // [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] Authored, not
    // run: closed intake rolls back allocation/arming and the durable clock,
    // while an exact existing poll and an admitted Armed barrier stay exact.
    #[test]
    fn post_validation_intake_closure_preserves_restartable_recipient_state() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        let mut admission_calls = 0;
        assert_eq!(journal.prepare_poll_with_admission_at(&f.claim, TEST_ROUTE_ORIGIN,
            || Ok(NOW + 1), || {
                assert!(journal.inner.try_lock().is_err());
                admission_calls += 1;
                Err(RecipientJournalError::IntakeClosed)
            }).err(), Some(RecipientJournalError::IntakeClosed));
        assert_eq!(admission_calls, 1);
        {
            let inner = journal.inner.lock().unwrap();
            let state: (i64, i64) = inner.connection.query_row(
                "SELECT (SELECT count(*) FROM recipient_jobs),clock FROM recipient_meta WHERE singleton=1",
                [], |row| Ok((row.get(0)?, row.get(1)?)),
            ).unwrap();
            assert_eq!(state, (0, NOW as i64));
            assert!(!inner.poisoned);
        }
        drop(journal);
        let journal = f.open(NOW + 1).unwrap();
        let exact = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW + 1).unwrap();
        assert_eq!(journal.prepare_poll_with_admission_at(&f.claim, TEST_ROUTE_ORIGIN,
            || Ok(NOW + 1), || panic!("exact recovery is not fresh allocation")).unwrap(), exact);
        journal.record_lease(f.claim.claim_id(), &f.lease, f.deadline, NOW + 1).unwrap();
        assert_eq!(journal.arm_with_admission_at(f.claim.claim_id(), || Ok(NOW + 2), || {
            assert!(journal.inner.try_lock().is_err());
            Err(RecipientJournalError::IntakeClosed)
        }).err(), Some(RecipientJournalError::IntakeClosed));
        {
            let inner = journal.inner.lock().unwrap();
            let state: (i64, Vec<u8>, Vec<u8>, bool, i64) = inner.connection.query_row(
                "SELECT phase,claim,lease,result IS NULL,clock FROM recipient_jobs CROSS JOIN recipient_meta WHERE claim_id=?1 AND singleton=1",
                params![f.claim.claim_id().as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            ).unwrap();
            assert_eq!(state, (2, exact, f.lease.encode(), true, (NOW + 1) as i64));
            assert!(!inner.poisoned);
        }
        drop(journal);
        let journal = f.open(NOW + 2).unwrap();
        assert!(matches!(journal.resume(None, 64, NOW + 2).unwrap().items.as_slice(),
            [RecipientRecovery::LeaseReady { .. }]));
        // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] The
        // terminal receives the actual transaction floor, not the old page time.
        let armed = journal.arm_with_admission_at(f.claim.claim_id(), || Ok(NOW + 3), || Ok(())).unwrap();
        assert_eq!(armed.armed_at, NOW + 3);
        assert_eq!(journal.arm_with_admission_at(f.claim.claim_id(), || Ok(NOW + 3),
            || panic!("closed intake cannot reopen Armed")).err(), Some(RecipientJournalError::Ambiguous));
    }

    // [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] Authored, not
    // run: the gate is later than clock/record validation, so stop cannot hide
    // rollback, unavailable storage, or a missing lease as ordinary closure.
    #[test]
    fn recipient_intake_gate_cannot_mask_clock_or_record_failures() {
        for fault in [false, true] {
            let f = Fixture::new(NOW + 600);
            let journal = f.open(NOW).unwrap();
            f.ready(&journal);
            assert_eq!(journal.arm_with_admission_at(f.claim.claim_id(),
                || if fault { Err(RecipientJournalError::Unavailable) } else { Ok(NOW) },
                || panic!("clock faults precede intake closure")).err(),
                Some(if fault { RecipientJournalError::Unavailable } else { RecipientJournalError::Rejected }));
            assert_eq!(journal.inner.lock().unwrap().poisoned, fault);
        }
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        assert_eq!(journal.arm_with_admission_at([99; 16], || Ok(NOW),
            || panic!("missing leases precede intake closure")).err(), Some(RecipientJournalError::Rejected));
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Authored, not
    // run: late new Claims cannot allocate, exact historical Polls still replay.
    #[test]
    fn post_lock_claim_time_distinguishes_new_admission_from_exact_recovery() {
        let f = Fixture::new(NOW + 600);
        let journal = f.open(NOW).unwrap();
        assert_eq!(journal.prepare_poll_at(&f.claim, TEST_ROUTE_ORIGIN, || {
            assert!(journal.inner.try_lock().is_err());
            Ok(f.claim.expires_at())
        }).err(), Some(RecipientJournalError::Expired));
        assert!(journal.resume(None, 64, NOW).unwrap().items.is_empty());
        let exact = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
        assert_eq!(journal.prepare_poll_at(&f.claim, TEST_ROUTE_ORIGIN,
            || Ok(NOW + 31)).unwrap(), exact);
        assert_eq!(journal.prepare_poll_at(&f.claim, TEST_ROUTE_ORIGIN,
            || Ok(NOW + 30)).err(), Some(RecipientJournalError::Rejected));
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Authored,
    // not run: only authenticated forward expiry is skippable after SQL wait.
    #[test]
    fn post_lock_no_work_and_lease_expiry_preserve_the_exact_poll() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        let exact = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
        let receipt = ReverseOnionNoWorkReceiptV1::issue_no_work(&f.claim, NOW + 1, &f.relay).unwrap();
        let encoded = receipt.encode();
        assert_eq!(journal.complete_no_work_poll_at(
            f.claim.claim_id(), &exact, &encoded, NOW + 1, || {
                assert!(journal.inner.try_lock().is_err());
                Ok(receipt.expires_at())
            },
        ).err(), Some(RecipientJournalError::Expired));
        let mut altered = encoded.clone();
        *altered.last_mut().unwrap() ^= 1;
        assert_eq!(journal.complete_no_work_poll_at(
            f.claim.claim_id(), &exact, &altered, NOW + 1, || Ok(receipt.expires_at()),
        ).err(), Some(RecipientJournalError::Rejected));
        assert_eq!(journal.complete_no_work_poll_at(
            f.claim.claim_id(), &exact, &encoded, NOW + 1, || Ok(NOW),
        ).err(), Some(RecipientJournalError::Rejected));
        assert_eq!(journal.record_bound_lease_at(
            f.claim.claim_id(), &f.lease, f.deadline, || {
                assert!(journal.inner.try_lock().is_err());
                Ok(f.deadline)
            },
        ).err(), Some(RecipientJournalError::Expired));
        let inner = journal.inner.lock().unwrap();
        let stored: (i64, Vec<u8>, bool, i64) = inner.connection.query_row(
            "SELECT phase,claim,lease IS NULL,clock FROM recipient_jobs CROSS JOIN recipient_meta WHERE claim_id=?1 AND singleton=1",
            params![f.claim.claim_id().as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        ).unwrap();
        assert_eq!(stored, (1, exact, true, NOW as i64));
        assert!(!inner.poisoned);
    }

    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] Authored,
    // not run: expired zero-dispatch restoration cannot reopen execution.
    #[test]
    fn zero_dispatch_does_not_restore_an_expired_exact_lease() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        f.ready(&journal);
        journal.arm(f.claim.claim_id(), NOW + 2).unwrap();
        assert_eq!(journal.restore_lease_after_zero_dispatch(
            f.claim.claim_id(), &f.claim, &f.lease, f.deadline, NOW + 10,
        ).err(), Some(RecipientJournalError::Expired));
        assert!(matches!(journal.resume(None, 64, NOW + 10).unwrap().items.as_slice(),
            [RecipientRecovery::Ambiguous { .. }]));
        assert_eq!(journal.arm(f.claim.claim_id(), NOW + 10).err(),
            Some(RecipientJournalError::Ambiguous));
    }

    // [PHALA-JOURNAL-RESULT-CLOCK 2026-10-07 by Codex] Authored, not run:
    // delay is injected at the post-lock clock point, not via wall-clock sleeps.
    #[test]
    fn result_post_lock_expiry_or_clock_failure_never_changes_armed_custody() {
        for fault in [false, true] {
            let f = Fixture::new(NOW + 600);
            let journal = f.open(NOW).unwrap();
            f.ready(&journal);
            journal.arm(f.claim.claim_id(), NOW + 2).unwrap();
            let result = f.result(NOW + 3);
            let outcome = journal.record_result_at(f.claim.claim_id(), &result, || {
                assert!(journal.inner.try_lock().is_err(), "clock must run under journal lock");
                if fault { Err(RecipientJournalError::Unavailable) }
                else { Ok(result.expires_at()) }
            });
            assert_eq!(outcome.err(), Some(if fault { RecipientJournalError::Unavailable }
                else { RecipientJournalError::Rejected }));
            let inner = journal.inner.lock().unwrap();
            let stored: (i64, bool, i64) = inner.connection.query_row(
                "SELECT phase,result IS NULL,clock FROM recipient_jobs CROSS JOIN recipient_meta WHERE claim_id=?1 AND singleton=1",
                params![f.claim.claim_id().as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            ).unwrap();
            assert_eq!(stored, (3, true, (NOW + 2) as i64));
            assert_eq!(inner.poisoned, fault);
            drop(inner);
            drop(journal);
            let reopened = f.open(NOW + 4).unwrap();
            assert!(matches!(reopened.resume(None, 64, NOW + 4).unwrap().items.as_slice(),
                [RecipientRecovery::Ambiguous { .. }]));
            assert_eq!(reopened.arm(f.claim.claim_id(), NOW + 4).err(),
                Some(RecipientJournalError::Ambiguous));
        }
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
            journal.prepare_poll(&claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
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
        journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
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
        journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
        let proof = f.lease.verify_recipient_lease(&f.claim,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW).unwrap();
        // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] A valid
        // proof that ages in transit is expiry, not malformed authority.
        assert_eq!(journal.record_relay_lease(proof, NOW + 10).err(), Some(RecipientJournalError::Expired));
        let page = journal.resume(None, 1, NOW + 10).unwrap();
        assert!(matches!(page.items.first(), Some(RecipientRecovery::Poll { .. })));
    }

    // [RECIPIENT-HISTORICAL-POLL 2026-10-04 by Codex] Authored, unexecuted.
    #[test]
    fn historical_poll_remains_exact_until_evidence_horizon_not_claim_expiry() {
        let f = Fixture::new(NOW + 10);
        let journal = f.open(NOW).unwrap();
        let bytes = journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW).unwrap();
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
            // [REVERSE-ONION-CLAIM-REPLAY 2026-10-06 by Codex] Retained
            // retries must return the committed bytes beyond 30-second freshness.
            assert_eq!(journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, now).unwrap(), bytes);
        }
        assert!(f.lease.verify_recipient_lease(&f.claim, f.relay.public_key_bytes(),
            f.recipient.public_key_bytes(), horizon - 1).is_err());
        assert!(journal.arm(f.claim.claim_id(), horizon - 1).is_err());
        assert_eq!(journal.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, horizon).err(), Some(RecipientJournalError::Expired));
        assert!(journal.resume(None, 1, horizon).unwrap().items.is_empty());
        assert_eq!(journal.cleanup(1, horizon).unwrap(), 1);

        let fresh = Fixture::new(NOW + 60);
        let journal = fresh.open(NOW).unwrap();
        assert_eq!(
            journal.prepare_poll(&fresh.claim, TEST_ROUTE_ORIGIN, NOW + 31).err(),
            // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex]
            Some(RecipientJournalError::Expired),
        );
        assert!(journal.resume(None, 1, NOW + 31).unwrap().items.is_empty());
    }

    // [REVERSE-ONION-ORIGIN-MIGRATION 2026-10-06 by Codex] Source-authored,
    // not executed: a v1 Poll without endpoint evidence migrates without data
    // loss but is never sent to a newly advertised relay origin.
    #[test]
    fn legacy_poll_migration_preserves_claim_as_non_network_ambiguity() {
        use std::os::unix::fs::PermissionsExt;

        let f = Fixture::new(NOW + 600);
        let path = f.path();
        let connection = Connection::open(&path).unwrap();
        connection.execute_batch(META_SQL).unwrap();
        connection.execute_batch(ROW_SQL_V1).unwrap();
        connection.execute(
            "INSERT INTO recipient_meta VALUES(1,?1,?2,?3)",
            params![
                f.relay.public_key_bytes().as_slice(),
                f.recipient.public_key_bytes().as_slice(),
                to_sql(NOW).unwrap(),
            ],
        ).unwrap();
        connection.execute(
            "INSERT INTO recipient_jobs VALUES(?1,NULL,?2,NULL,NULL,0,?3,1)",
            params![
                f.claim.claim_id().as_slice(),
                f.claim.encode(),
                to_sql(f.claim.expires_at() + UNLEASED_EVIDENCE_SECS).unwrap(),
            ],
        ).unwrap();
        connection.pragma_update(None, "user_version", 1).unwrap();
        connection.pragma_update(None, "application_id", APPLICATION_ID).unwrap();
        drop(connection);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let migrated = f.open_existing(NOW + 1).unwrap();
        let page = migrated.resume(None, 1, NOW + 1).unwrap();
        assert!(matches!(page.items.first(), Some(RecipientRecovery::Ambiguous { claim_id })
            if *claim_id == f.claim.claim_id()));
        assert!(migrated.prepare_poll(&f.claim, TEST_ROUTE_ORIGIN, NOW + 1).is_err());
    }

    // [REVERSE-ONION-ORIGIN-MIGRATION 2026-10-06 by Codex] Source-authored,
    // not executed: an already signed v1 Lease remains locally executable;
    // migration adds no endpoint evidence and does not strand accepted work.
    #[test]
    fn legacy_lease_migration_keeps_local_execution_without_origin_evidence() {
        use std::os::unix::fs::PermissionsExt;

        let f = Fixture::new(NOW + 600);
        let path = f.path();
        let connection = Connection::open(&path).unwrap();
        connection.execute_batch(META_SQL).unwrap();
        connection.execute_batch(ROW_SQL_V1).unwrap();
        connection.execute(
            "INSERT INTO recipient_meta VALUES(1,?1,?2,?3)",
            params![
                f.relay.public_key_bytes().as_slice(),
                f.recipient.public_key_bytes().as_slice(),
                to_sql(NOW).unwrap(),
            ],
        ).unwrap();
        connection.execute(
            "INSERT INTO recipient_jobs VALUES(?1,?2,?3,?4,NULL,?5,?6,2)",
            params![
                f.claim.claim_id().as_slice(),
                f.lease.route_id().as_slice(),
                f.claim.encode(),
                f.lease.encode(),
                to_sql(f.deadline).unwrap(),
                to_sql(f.lease.replay_evidence_deadline().unwrap()).unwrap(),
            ],
        ).unwrap();
        connection.pragma_update(None, "user_version", 1).unwrap();
        connection.pragma_update(None, "application_id", APPLICATION_ID).unwrap();
        drop(connection);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let migrated = f.open_existing(NOW + 1).unwrap();
        let page = migrated.resume(None, 1, NOW + 1).unwrap();
        assert!(matches!(page.items.first(), Some(RecipientRecovery::LeaseReady { claim_id })
            if *claim_id == f.claim.claim_id()));
        let dispatch = migrated.arm(f.claim.claim_id(), NOW + 1).unwrap();
        assert_eq!(dispatch.route_origin_commitment, None);
        assert_eq!(dispatch.claim.encode(), f.claim.encode());
        assert_eq!(dispatch.lease.encode(), f.lease.encode());
    }

    // [REVERSE-ONION-ORIGIN-MIGRATION 2026-10-06 by Codex] Source-authored,
    // not executed: a legacy terminal Result remains durable but cannot be
    // acknowledged against a newly resolved or rotated relay origin.
    #[test]
    fn legacy_result_migration_preserves_result_as_non_network_ambiguity() {
        use std::os::unix::fs::PermissionsExt;

        let f = Fixture::new(NOW + 600);
        let result = f.result(NOW + 2);
        let path = f.path();
        let connection = Connection::open(&path).unwrap();
        connection.execute_batch(META_SQL).unwrap();
        connection.execute_batch(ROW_SQL_V1).unwrap();
        connection.execute(
            "INSERT INTO recipient_meta VALUES(1,?1,?2,?3)",
            params![
                f.relay.public_key_bytes().as_slice(),
                f.recipient.public_key_bytes().as_slice(),
                to_sql(NOW).unwrap(),
            ],
        ).unwrap();
        connection.execute(
            "INSERT INTO recipient_jobs VALUES(?1,?2,?3,?4,?5,?6,?7,4)",
            params![
                f.claim.claim_id().as_slice(),
                f.lease.route_id().as_slice(),
                f.claim.encode(),
                f.lease.encode(),
                result.encode(),
                to_sql(f.deadline).unwrap(),
                to_sql(f.lease.replay_evidence_deadline().unwrap()).unwrap(),
            ],
        ).unwrap();
        connection.pragma_update(None, "user_version", 1).unwrap();
        connection.pragma_update(None, "application_id", APPLICATION_ID).unwrap();
        drop(connection);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let migrated = f.open_existing(NOW + 3).unwrap();
        let page = migrated.resume(None, 1, NOW + 3).unwrap();
        assert!(matches!(page.items.first(), Some(RecipientRecovery::Ambiguous { claim_id })
            if *claim_id == f.claim.claim_id()));
        let preserved: (Vec<u8>, Option<Vec<u8>>) = migrated.inner.lock().unwrap().connection
            .query_row(
                "SELECT result,route_origin_commitment FROM recipient_jobs WHERE claim_id=?1",
                params![f.claim.claim_id().as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).unwrap();
        assert_eq!(preserved.0, result.encode());
        assert_eq!(preserved.1, None);
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
