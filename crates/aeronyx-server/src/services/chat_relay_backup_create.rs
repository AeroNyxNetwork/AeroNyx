// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_create.rs
// ============================================
// Version: 1.0.0-VerifiedBackupCreationDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-CREATE-DOMAIN 2026-08-27 by Codex] Extract verified
//   recovery-image certification, no-replace publication, replay verification,
//   and failure cleanup from the oversized relay service.
//
// Main Functionality:
//   - Creates one owner-private SQLite recovery image through a typed command.
//   - Re-verifies an existing idempotent artifact without replacing it.
//   - Publishes by hard link so a destination is never overwritten.
//   - Tracks publication ownership and removes only this command's artifacts.
//   - Rejects mutable sidecars, symbolic links, empty files, and metadata drift.
//
// Dependencies:
//   - `chat_relay_backup_io` supplies private-file reservation and parent fsync.
//   - `chat_relay_backup_contract` supplies the aggregate-only receipt.
//   - The composed `BackupDatabaseCertification` capability owns SQLite policy.
//
// Main Logical Flow:
//   1. Reuse and fully verify an existing destination when replay is allowed.
//   2. Reserve an unpredictable owner-private temporary artifact.
//   3. Ask the composed database capability to copy, normalize, and certify it.
//   4. Publish with a no-replace hard link and durably synchronize the parent.
//   5. Return only size/creation state; RAII removes owned partial artifacts.
//
// Important Note for Next Developer:
//   - Never replace an existing destination or follow its final symlink.
//   - `destination_owned` becomes true only after this command creates the link.
//   - Existing replay artifacts must remain untouched even when corrupt.
//   - Database verification must stay path-free and inspect no E2E plaintext.
//
// Last Modified:
//   v1.0.0-VerifiedBackupCreationDomain - Initial composed creation command
// ============================================

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags};

use crate::services::chat_relay_backup_contract::ChatRelayBackupReceipt;
use crate::services::chat_relay_backup_io::{backup_io_error, BackupFilesystem};
use crate::services::chat_relay_error::ChatRelayResult;

/// SQLite-specific capability required by backup artifact orchestration.
pub(crate) trait BackupDatabaseCertification {
    /// Copies the live source database into one already-open isolated image.
    fn copy_source_into(&self, destination: &mut Connection) -> ChatRelayResult<()>;

    /// Normalizes mutable journal state on an isolated recovery image.
    fn normalize_isolated_journal(&self, connection: &Connection) -> ChatRelayResult<()>;

    /// Proves physical and logical recovery-image integrity.
    fn verify_recovery_image(&self, connection: &Connection) -> ChatRelayResult<()>;

    /// Restricts one SQLite artifact to the node service account.
    fn restrict_file_permissions(&self, path: &Path) -> ChatRelayResult<()>;
}

/// Immutable input for one verified backup creation attempt.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VerifiedBackupCreationRequest<'a> {
    pub backup_directory: &'a Path,
    pub destination: &'a Path,
    pub temporary: &'a Path,
    pub reuse_existing: bool,
}

/// Composed verified backup creation command.
pub(crate) struct VerifiedBackupCreationCommand<F, C> {
    filesystem: F,
    certification: C,
}

impl<F, C> VerifiedBackupCreationCommand<F, C> {
    pub(crate) const fn new(filesystem: F, certification: C) -> Self {
        Self {
            filesystem,
            certification,
        }
    }
}

/// RAII ownership record for one no-replace publication attempt.
struct BackupPublicationGuard<'a> {
    temporary: &'a Path,
    destination: &'a Path,
    destination_owned: bool,
}

impl<'a> BackupPublicationGuard<'a> {
    const fn new(temporary: &'a Path, destination: &'a Path) -> Self {
        Self {
            temporary,
            destination,
            destination_owned: false,
        }
    }

    fn record_destination_publication(&mut self) {
        self.destination_owned = true;
    }

    fn disarm_destination_cleanup(&mut self) {
        self.destination_owned = false;
    }
}

impl Drop for BackupPublicationGuard<'_> {
    fn drop(&mut self) {
        remove_sqlite_artifact(self.temporary);
        if self.destination_owned {
            // [CHAT-RELAY-BACKUP-CREATE-DOMAIN 2026-08-27 by Codex] A failed
            // command may remove only the destination link it published itself.
            // A replay destination observed before publication is never owned.
            remove_sqlite_artifact(self.destination);
        }
    }
}

impl<F, C> VerifiedBackupCreationCommand<F, C>
where
    F: BackupFilesystem,
    C: BackupDatabaseCertification,
{
    /// Creates or reuses one fully certified private recovery image.
    pub(crate) fn execute(
        &self,
        request: VerifiedBackupCreationRequest<'_>,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        match std::fs::symlink_metadata(request.destination) {
            Ok(_) if request.reuse_existing => {
                return Ok(ChatRelayBackupReceipt {
                    size_bytes: self.verify_existing_artifact(request.destination)?,
                    created: false,
                });
            }
            Ok(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CONSTRAINT,
                    "relay backup destination already exists",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect relay backup destination",
                ));
            }
        }

        self.filesystem.reserve_private_file(request.temporary)?;
        let mut guard = BackupPublicationGuard::new(request.temporary, request.destination);
        self.prepare_temporary_artifact(request.temporary)?;

        match std::fs::hard_link(request.temporary, request.destination) {
            Ok(()) => guard.record_destination_publication(),
            Err(error)
                if request.reuse_existing && error.kind() == std::io::ErrorKind::AlreadyExists =>
            {
                // [CHAT-RELAY-BACKUP-CREATE-DOMAIN 2026-08-27 by Codex]
                // Remove our unpublished temporary before the existing-image
                // verifier fsyncs the shared directory, preserving the v1
                // crash-recovery ordering for concurrent command replay.
                remove_sqlite_artifact(request.temporary);
                return Ok(ChatRelayBackupReceipt {
                    size_bytes: self.verify_existing_artifact(request.destination)?,
                    created: false,
                });
            }
            Err(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CONSTRAINT,
                    "unable to publish relay backup without replacement",
                ));
            }
        }

        self.certification
            .restrict_file_permissions(request.destination)?;
        std::fs::remove_file(request.temporary).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to finalize relay backup publication",
            )
        })?;
        self.filesystem
            .sync_backup_parent(request.backup_directory)?;

        let metadata = std::fs::symlink_metadata(request.destination).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to inspect published relay backup artifact",
            )
        })?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "published relay backup artifact is invalid",
            ));
        }

        guard.disarm_destination_cleanup();
        Ok(ChatRelayBackupReceipt {
            size_bytes: metadata.len(),
            created: true,
        })
    }

    fn prepare_temporary_artifact(&self, path: &Path) -> ChatRelayResult<()> {
        let mut backup_connection = Connection::open(path).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to open private relay backup file",
            )
        })?;
        self.certification.restrict_file_permissions(path)?;
        self.certification
            .copy_source_into(&mut backup_connection)?;
        self.certification
            .normalize_isolated_journal(&backup_connection)?;
        self.certification
            .verify_recovery_image(&backup_connection)?;
        drop(backup_connection);

        self.certification.restrict_file_permissions(path)?;
        std::fs::File::open(path)
            .and_then(|file| file.sync_all())
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to synchronize relay backup file",
                )
            })
    }

    fn verify_existing_artifact(&self, path: &Path) -> ChatRelayResult<u64> {
        verify_existing_backup_artifact(&self.filesystem, path, |connection| {
            self.certification.verify_recovery_image(connection)
        })
    }
}

/// Fully verifies one existing immutable recovery image without replacing it.
///
/// This free capability also serves retention inventory reads, which do not
/// own a live `ChatRelayService` instance but share the exact certification
/// and no-follow filesystem semantics used by command replay.
pub(crate) fn verify_existing_backup_artifact<F, V>(
    filesystem: &F,
    path: &Path,
    verify_recovery_image: V,
) -> ChatRelayResult<u64>
where
    F: BackupFilesystem,
    V: FnOnce(&Connection) -> ChatRelayResult<()>,
{
    let metadata = std::fs::symlink_metadata(path).map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "unable to inspect existing relay backup artifact",
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "existing relay backup artifact is not a private regular file",
        ));
    }

    verify_owner_private_permissions(&metadata)?;
    reject_mutable_sidecars(path)?;

    // Resolve only the already-verified private directory. macOS temp paths
    // commonly contain a system-managed ancestor symlink (`/var`), while
    // SQLite NOFOLLOW rejects any symlink component. The final filename stays
    // separate so the final-component race defense remains active.
    let parent = path.parent().ok_or_else(|| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "existing relay backup artifact has no private parent",
        )
    })?;
    let file_name = path.file_name().ok_or_else(|| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "existing relay backup artifact has no private name",
        )
    })?;
    let canonical_parent = std::fs::canonicalize(parent).map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "unable to resolve private relay backup directory",
        )
    })?;
    let nofollow_path = canonical_parent.join(file_name);

    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NOFOLLOW;
    let backup_connection = Connection::open_with_flags(&nofollow_path, flags).map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "unable to open existing relay backup artifact",
        )
    })?;
    verify_recovery_image(&backup_connection)?;
    drop(backup_connection);

    let verified_metadata = std::fs::symlink_metadata(&nofollow_path).map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_IOERR,
            "verified relay backup artifact became unavailable",
        )
    })?;
    if verified_metadata.file_type().is_symlink()
        || !verified_metadata.is_file()
        || verified_metadata.len() != metadata.len()
    {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_CORRUPT,
            "verified relay backup artifact changed during inspection",
        ));
    }

    // A concurrent hard-link creator may not yet have reached its parent fsync.
    // Either successful replay receipt must make the shared name durable.
    filesystem.sync_backup_parent(&canonical_parent)?;
    Ok(verified_metadata.len())
}

#[cfg(unix)]
fn verify_owner_private_permissions(metadata: &std::fs::Metadata) -> ChatRelayResult<()> {
    use std::os::unix::fs::PermissionsExt;

    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_PERM,
            "existing relay backup artifact is not owner-private",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn verify_owner_private_permissions(_metadata: &std::fs::Metadata) -> ChatRelayResult<()> {
    Ok(())
}

fn reject_mutable_sidecars(path: &Path) -> ChatRelayResult<()> {
    for sidecar in sqlite_sidecars(path) {
        match std::fs::symlink_metadata(sidecar) {
            Ok(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "existing relay backup artifact has mutable sidecar state",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect existing relay backup sidecar state",
                ));
            }
        }
    }
    Ok(())
}

fn remove_sqlite_artifact(path: &Path) {
    let _ = std::fs::remove_file(path);
    for sidecar in sqlite_sidecars(path) {
        let _ = std::fs::remove_file(sidecar);
    }
}

fn sqlite_sidecars(path: &Path) -> [PathBuf; 3] {
    ["-journal", "-wal", "-shm"].map(|suffix| {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(suffix);
        PathBuf::from(sidecar)
    })
}
