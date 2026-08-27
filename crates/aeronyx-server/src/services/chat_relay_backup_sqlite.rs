// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_sqlite.rs
// ============================================
// Version: 1.0.0-SqliteBackupAdapter
//
// Creation Reason:
//   [CHAT-BACKUP-SQLITE-DOMAIN 2026-08-28 by Codex] Extract live SQLite
//   backup copying, durability activation, and private-file enforcement from
//   the relay orchestration service.
//
// Main Functionality:
//   - Implements the verified-backup database certification capability.
//   - Runs bounded SQLite online-backup steps with fail-closed busy retries.
//   - Normalizes and certifies the isolated recovery image before publication.
//   - Enforces owner-only SQLite file permissions on Unix hosts.
//   - Activates and verifies FULL-or-stronger live custody durability.
//
// Dependencies:
//   - `chat_relay_backup_certification.rs` owns logical recovery verification.
//   - `chat_relay_backup_copy.rs` owns the side-effect-free retry state machine.
//   - `chat_relay_backup_create.rs` defines the creation capability contract.
//   - `chat_relay_backup_io.rs` supplies path-free host I/O failures.
//   - `rusqlite` supplies online backup and durable connection primitives.
//
// Main Logical Flow:
//   1. Lock the live source only for the SQLite online-copy lifetime.
//   2. Copy bounded pages and apply the retry policy to each step result.
//   3. Normalize the isolated journal and certify complete recovery semantics.
//   4. Restrict the final artifact before its creation command publishes it.
//
// Important Note for Next Developer:
//   - Never expose source or destination paths through returned failures.
//   - Keep busy retries bounded; indefinite backup lock waits are unacceptable.
//   - Do not weaken FULL durability for acknowledged custody writes.
//   - Certification must remain on the isolated image, never mutable live state.
//   - The source mutex must not cover validation or filesystem publication.
//
// Last Modified:
//   v1.0.0-SqliteBackupAdapter - Initial SQLite capability extraction
// ============================================

use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use rusqlite::{
    backup::{Backup, StepResult},
    Connection,
};

use super::chat_relay_backup_certification::{
    BackupRecoveryImageCertification, SqliteBackupRecoveryImageCertifier,
};
use super::chat_relay_backup_copy::{
    BackupCopyAction, BackupCopyPolicyError, BackupCopyProgress, BackupCopyRetryPolicy,
    BackupCopyRetryState, BoundedBackupCopyRetryPolicy,
};
use super::chat_relay_backup_create::BackupDatabaseCertification;
use super::chat_relay_backup_io::backup_io_error;
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Live SQLite source composed with isolated recovery-image certification.
pub(crate) struct SqliteRelayBackupDatabase<'a> {
    source: &'a Mutex<Connection>,
    certifier: SqliteBackupRecoveryImageCertifier,
    pages_per_step: i32,
    retry_policy: BoundedBackupCopyRetryPolicy,
}

impl<'a> SqliteRelayBackupDatabase<'a> {
    /// Composes one live source with bounded copy and certification policies.
    pub(crate) fn new(
        source: &'a Mutex<Connection>,
        certifier: SqliteBackupRecoveryImageCertifier,
        pages_per_step: i32,
        busy_timeout: Duration,
        retry_delay: Duration,
    ) -> Self {
        Self {
            source,
            certifier,
            pages_per_step,
            retry_policy: BoundedBackupCopyRetryPolicy::new(busy_timeout, retry_delay),
        }
    }

    fn copy_into(&self, destination: &mut Connection) -> ChatRelayResult<()> {
        let source = self.source.lock();
        let backup = Backup::new(&source, destination)?;
        let mut retry_state = BackupCopyRetryState::default();
        loop {
            let progress = map_step_result(backup.step(self.pages_per_step)?);
            let action = self
                .retry_policy
                .transition(&mut retry_state, progress, Instant::now())
                .map_err(map_copy_policy_error)?;
            match action {
                BackupCopyAction::Complete => return Ok(()),
                BackupCopyAction::Continue => {}
                BackupCopyAction::RetryAfter(delay) => std::thread::sleep(delay),
            }
        }
    }
}

impl BackupDatabaseCertification for SqliteRelayBackupDatabase<'_> {
    fn copy_source_into(&self, destination: &mut Connection) -> ChatRelayResult<()> {
        self.copy_into(destination)
    }

    fn normalize_isolated_journal(&self, connection: &Connection) -> ChatRelayResult<()> {
        self.certifier.normalize_journal(connection)
    }

    fn verify_recovery_image(&self, connection: &Connection) -> ChatRelayResult<()> {
        self.certifier.verify(connection, now_secs())
    }

    fn restrict_file_permissions(&self, path: &Path) -> ChatRelayResult<()> {
        restrict_private_sqlite_permissions(path)
    }
}

/// Activates WAL plus FULL durability and verifies the effective SQLite level.
pub(crate) fn configure_full_durability(
    connection: &Connection,
    minimum_synchronous_level: i64,
) -> ChatRelayResult<u8> {
    // [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] NORMAL protects SQLite
    // consistency across process failure but may lose a recently acknowledged
    // transaction after host power loss. Signed custody activation therefore
    // requires FULL-or-stronger durability and verifies the effective level.
    connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
    let synchronous_level =
        connection.query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))?;
    if synchronous_level < minimum_synchronous_level {
        return Err(ChatRelayError::CorruptStoredData {
            field: "sqlite_synchronous_level",
        });
    }
    u8::try_from(synchronous_level).map_err(|_| ChatRelayError::CorruptStoredData {
        field: "sqlite_synchronous_level",
    })
}

/// Restricts a relay SQLite database or recovery artifact to its owner.
#[cfg(unix)]
pub(crate) fn restrict_private_sqlite_permissions(path: &Path) -> ChatRelayResult<()> {
    use std::os::unix::fs::PermissionsExt;

    // [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] SQLite creates WAL
    // sidecars using the primary database mode. Tighten the primary file
    // before enabling WAL, and keep the configured path out of the error.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|_| {
        ChatRelayError::Sqlite(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_PERM),
            Some("unable to restrict relay database permissions".to_string()),
        ))
    })
}

/// Non-Unix hosts enforce file privacy through their platform ACL boundary.
#[cfg(not(unix))]
pub(crate) fn restrict_private_sqlite_permissions(_path: &Path) -> ChatRelayResult<()> {
    Ok(())
}

fn map_step_result(step: StepResult) -> BackupCopyProgress {
    match step {
        StepResult::Done => BackupCopyProgress::Complete,
        StepResult::More => BackupCopyProgress::More,
        StepResult::Busy => BackupCopyProgress::Busy,
        StepResult::Locked => BackupCopyProgress::Locked,
        _ => BackupCopyProgress::Unsupported,
    }
}

fn map_copy_policy_error(error: BackupCopyPolicyError) -> ChatRelayError {
    match error {
        BackupCopyPolicyError::BusyTimeout => {
            backup_io_error(rusqlite::ffi::SQLITE_BUSY, "relay backup remained busy")
        }
        BackupCopyPolicyError::ObservationTimeRegressed => backup_io_error(
            rusqlite::ffi::SQLITE_ABORT,
            "relay backup retry observation time regressed",
        ),
        BackupCopyPolicyError::UnsupportedProgress => backup_io_error(
            rusqlite::ffi::SQLITE_ERROR,
            "unsupported relay backup step result",
        ),
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
