// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_io.rs
// ============================================
// Version: 1.0.0-BackupFilesystemDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-FILESYSTEM-DOMAIN 2026-08-27 by Codex] Extract the
//   host-local backup filesystem boundary from the oversized relay service.
//
// Main Functionality:
//   - Creates and verifies owner-private backup directories and files.
//   - Rejects symbolic links and non-regular control-file boundaries.
//   - Rejects foreign-owned or multiply-linked private control files.
//   - Acquires an exclusive, RAII-released cross-process maintenance lock.
//   - Durably synchronizes backup publication boundaries.
//
// Dependencies:
//   - `chat_relay_error` supplies the stable relay error contract.
//   - `rusqlite` supplies portable locking and stable SQLite error codes.
//   - The host filesystem supplies permissions, canonical paths, and fsync.
//
// Main Logical Flow:
//   1. Resolve a database-relative private backup directory.
//   2. Create or inspect the directory without accepting a symbolic link.
//   3. Open control files with owner-private, no-follow semantics.
//   4. Serialize maintenance through an exclusive SQLite transaction lock.
//   5. Synchronize the affected directory before reporting durable success.
//
// Important Note for Next Developer:
//   - Do not weaken Unix `0700` directory or `0600` file requirements.
//   - Never follow a symbolic link at a private backup boundary.
//   - Validate file owner and link count before returning a writable handle.
//   - Lock acquisition and publication durability must remain fail-closed.
//   - Keep backup policy, artifact names, payloads, and service state elsewhere.
//
// Last Modified:
//   v1.1.0-ControlFileIdentity - Rejected foreign-owned and multiply-linked
//     control files before callers can mutate their shared inode
//   v1.0.0-BackupFilesystemDomain - Initial filesystem capability extraction
// ============================================

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};

use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Access mode for a private relay backup control file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrivateBackupControlFileMode {
    /// Open for bounded read/write replacement operations.
    ReadWrite,
    /// Open for append-only maintenance audit publication.
    Append,
}

impl PrivateBackupControlFileMode {
    const fn appends(self) -> bool {
        matches!(self, Self::Append)
    }
}

/// Host filesystem capability required by relay backup orchestration.
pub(super) trait BackupFilesystem {
    /// Atomically reserves a new owner-private backup artifact.
    fn reserve_private_file(&self, path: &Path) -> ChatRelayResult<()>;

    /// Creates or validates an owner-private backup directory.
    fn ensure_private_directory(&self, path: &Path) -> ChatRelayResult<()>;

    /// Resolves and prepares the backup directory adjacent to a database.
    fn private_directory_for_database(&self, db_path: &str) -> ChatRelayResult<PathBuf>;

    /// Opens or creates a private regular control file.
    fn open_control_file(
        &self,
        path: &Path,
        mode: PrivateBackupControlFileMode,
    ) -> ChatRelayResult<File>;

    /// Opens an existing private regular control file without creating it.
    fn open_existing_control_file(&self, path: &Path) -> ChatRelayResult<Option<File>>;

    /// Acquires the host-local exclusive backup maintenance lock.
    fn acquire_maintenance_lock(
        &self,
        backup_directory: &Path,
        lock_file_name: &str,
    ) -> ChatRelayResult<Connection>;

    /// Synchronizes backup-directory entries after mutation.
    fn sync_backup_directory(&self, backup_directory: &Path) -> ChatRelayResult<()>;

    /// Synchronizes a parent directory after atomic publication.
    fn sync_backup_parent(&self, parent: &Path) -> ChatRelayResult<()>;
}

/// Production host-filesystem implementation of [`BackupFilesystem`].
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LocalBackupFilesystem;

/// Converts a host backup I/O failure into the relay's stable `SQLite` boundary.
pub(super) fn backup_io_error(code: i32, message: &'static str) -> ChatRelayError {
    ChatRelayError::Sqlite(rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(code),
        Some(message.to_string()),
    ))
}

#[cfg(unix)]
fn reserve_private_file(path: &Path) -> ChatRelayResult<()> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map(drop)
        .map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to reserve private relay backup file",
            )
        })
}

#[cfg(not(unix))]
fn reserve_private_file(path: &Path) -> ChatRelayResult<()> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map(drop)
        .map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to reserve private relay backup file",
            )
        })
}

#[cfg(unix)]
fn create_private_directory(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_directory(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir(path)
}

#[cfg(unix)]
fn restrict_directory_permissions(path: &Path) -> ChatRelayResult<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_PERM,
            "unable to restrict relay backup directory permissions",
        )
    })
}

#[cfg(not(unix))]
fn restrict_directory_permissions(_path: &Path) -> ChatRelayResult<()> {
    Ok(())
}

fn validate_private_control_file(file: &File) -> ChatRelayResult<()> {
    let metadata = file.metadata().map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_IOERR,
            "unable to inspect private relay backup control file",
        )
    })?;
    if !metadata.is_file() {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_PERM,
            "relay backup control boundary is not a private regular file",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        // [CHAT-RELAY-BACKUP-FILE-IDENTITY 2026-08-31 by Codex] A private
        // mode does not prove exclusive ownership of an inode. Reject foreign
        // ownership and hard links before a caller receives a writable handle.
        // SAFETY: `geteuid` has no preconditions and does not access memory.
        let effective_user_id = unsafe { nix::libc::geteuid() };
        if metadata.uid() != effective_user_id
            || metadata.nlink() != 1
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "relay backup control file is not owner-private",
            ));
        }
    }
    Ok(())
}

impl BackupFilesystem for LocalBackupFilesystem {
    fn reserve_private_file(&self, path: &Path) -> ChatRelayResult<()> {
        reserve_private_file(path)
    }

    fn ensure_private_directory(&self, path: &Path) -> ChatRelayResult<()> {
        match std::fs::symlink_metadata(path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(backup_io_error(
                        rusqlite::ffi::SQLITE_CANTOPEN,
                        "relay backup boundary is not a private directory",
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // [CHAT-RELAY-BACKUP-FILESYSTEM-DOMAIN 2026-08-27 by Codex]
                // A single-level owner-private create avoids a permissive umask
                // window. Re-inspection below resolves an AlreadyExists race.
                match create_private_directory(path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(_) => {
                        return Err(backup_io_error(
                            rusqlite::ffi::SQLITE_CANTOPEN,
                            "unable to create private relay backup directory",
                        ));
                    }
                }
            }
            Err(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect private relay backup directory",
                ));
            }
        }

        let metadata = std::fs::symlink_metadata(path).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect private relay backup directory",
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup boundary is not a private directory",
            ));
        }
        restrict_directory_permissions(path)
    }

    fn private_directory_for_database(&self, db_path: &str) -> ChatRelayResult<PathBuf> {
        if db_path == ":memory:" {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "in-memory relay storage has no private backup boundary",
            ));
        }
        let source_path = Path::new(db_path);
        let source_parent = source_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let backup_directory = source_parent.join(".aeronyx-relay-backups");
        self.ensure_private_directory(&backup_directory)?;
        Ok(backup_directory)
    }

    fn open_control_file(
        &self,
        path: &Path,
        mode: PrivateBackupControlFileMode,
    ) -> ChatRelayResult<File> {
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt;

        #[cfg(not(unix))]
        if let Ok(metadata) = std::fs::symlink_metadata(path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup control boundary is not a private regular file",
                ));
            }
        }

        let append = mode.appends();
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(!append)
            .append(append)
            .create(true);
        #[cfg(unix)]
        options
            .mode(0o600)
            .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
        let file = options.open(path).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to open private relay backup control file",
            )
        })?;
        validate_private_control_file(&file)?;
        Ok(file)
    }

    fn open_existing_control_file(&self, path: &Path) -> ChatRelayResult<Option<File>> {
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt;

        #[cfg(not(unix))]
        match std::fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup control boundary is not a private regular file",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect private relay backup control file",
                ));
            }
        }

        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
        let file = match options.open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to open private relay backup control file",
                ));
            }
        };
        validate_private_control_file(&file)?;
        Ok(Some(file))
    }

    fn acquire_maintenance_lock(
        &self,
        backup_directory: &Path,
        lock_file_name: &str,
    ) -> ChatRelayResult<Connection> {
        let parent = backup_directory.parent().ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private control parent",
            )
        })?;
        let lock_path = parent.join(lock_file_name);
        // [CHAT-RELAY-BACKUP-FILESYSTEM-DOMAIN 2026-08-27 by Codex]
        // `File::try_lock` requires Rust 1.89 while the workspace promises
        // MSRV 1.75. SQLite provides a portable RAII transaction lock.
        drop(self.open_control_file(&lock_path, PrivateBackupControlFileMode::ReadWrite)?);
        let canonical_parent = std::fs::canonicalize(parent).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to resolve relay backup maintenance lock parent",
            )
        })?;
        let lock_name = lock_path.file_name().ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup maintenance lock has no private name",
            )
        })?;
        let nofollow_lock_path = canonical_parent.join(lock_name);
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let lock = Connection::open_with_flags(&nofollow_lock_path, flags).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to open relay backup maintenance lock",
            )
        })?;
        lock.busy_timeout(Duration::ZERO).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to configure relay backup maintenance lock",
            )
        })?;
        lock.execute_batch("BEGIN EXCLUSIVE;").map_err(|error| {
            let code = match error.sqlite_error_code() {
                Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked) => {
                    rusqlite::ffi::SQLITE_BUSY
                }
                _ => rusqlite::ffi::SQLITE_IOERR,
            };
            backup_io_error(code, "relay backup maintenance lock is unavailable")
        })?;
        Ok(lock)
    }

    fn sync_backup_directory(&self, backup_directory: &Path) -> ChatRelayResult<()> {
        File::open(backup_directory)
            .and_then(|directory| directory.sync_all())
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR_FSYNC,
                    "unable to durably sync relay backup directory",
                )
            })
    }

    #[cfg(unix)]
    fn sync_backup_parent(&self, parent: &Path) -> ChatRelayResult<()> {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to synchronize relay backup directory",
                )
            })
    }

    #[cfg(not(unix))]
    fn sync_backup_parent(&self, _parent: &Path) -> ChatRelayResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sqlite_extended_code(error: ChatRelayError) -> i32 {
        match error {
            ChatRelayError::Sqlite(rusqlite::Error::SqliteFailure(error, _)) => error.extended_code,
            other => panic!("unexpected backup filesystem error: {other}"),
        }
    }

    #[test]
    fn private_directory_is_database_relative_and_owner_private() {
        let root = tempfile::tempdir().expect("temporary directory");
        let database = root.path().join("relay.sqlite3");
        let filesystem = LocalBackupFilesystem;

        let backup = filesystem
            .private_directory_for_database(database.to_str().expect("UTF-8 path"))
            .expect("private backup directory");

        assert_eq!(backup, root.path().join(".aeronyx-relay-backups"));
        assert!(backup.is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&backup)
                .expect("backup metadata")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o700);
        }
    }

    #[test]
    fn in_memory_database_has_no_private_backup_boundary() {
        let error = LocalBackupFilesystem
            .private_directory_for_database(":memory:")
            .expect_err("in-memory backup must fail closed");

        assert_eq!(sqlite_extended_code(error), rusqlite::ffi::SQLITE_CANTOPEN);
    }

    #[test]
    fn private_control_file_supports_append_and_existing_read() {
        use std::io::Write;

        let root = tempfile::tempdir().expect("temporary directory");
        let path = root.path().join("control.log");
        let filesystem = LocalBackupFilesystem;
        filesystem
            .open_control_file(&path, PrivateBackupControlFileMode::ReadWrite)
            .expect("create control file");
        let mut append = filesystem
            .open_control_file(&path, PrivateBackupControlFileMode::Append)
            .expect("append control file");
        append.write_all(b"record\n").expect("append record");
        append.sync_all().expect("sync record");

        assert!(filesystem
            .open_existing_control_file(&path)
            .expect("inspect existing control file")
            .is_some());
        assert!(filesystem
            .open_existing_control_file(&root.path().join("absent"))
            .expect("inspect absent control file")
            .is_none());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path)
                .expect("control metadata")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn maintenance_lock_is_exclusive_and_raii_released() {
        let root = tempfile::tempdir().expect("temporary directory");
        let backup = root.path().join(".aeronyx-relay-backups");
        let filesystem = LocalBackupFilesystem;
        filesystem
            .ensure_private_directory(&backup)
            .expect("private backup directory");

        let first = filesystem
            .acquire_maintenance_lock(&backup, ".maintenance.lock")
            .expect("first maintenance lock");
        let error = filesystem
            .acquire_maintenance_lock(&backup, ".maintenance.lock")
            .expect_err("concurrent maintenance lock must fail closed");
        assert_eq!(sqlite_extended_code(error), rusqlite::ffi::SQLITE_BUSY);

        drop(first);
        filesystem
            .acquire_maintenance_lock(&backup, ".maintenance.lock")
            .expect("released maintenance lock can be reacquired");
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_link_directory_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temporary directory");
        let actual = root.path().join("actual");
        let boundary = root.path().join("boundary");
        std::fs::create_dir(&actual).expect("actual directory");
        symlink(&actual, &boundary).expect("directory symlink");

        let error = LocalBackupFilesystem
            .ensure_private_directory(&boundary)
            .expect_err("symbolic link boundary must fail closed");
        assert_eq!(sqlite_extended_code(error), rusqlite::ffi::SQLITE_CANTOPEN);
    }

    #[cfg(unix)]
    #[test]
    fn unsafe_control_file_boundaries_are_rejected() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let root = tempfile::tempdir().expect("temporary directory");
        let filesystem = LocalBackupFilesystem;
        let actual = root.path().join("actual-control");
        let symbolic = root.path().join("symbolic-control");
        std::fs::write(&actual, b"control").expect("actual control file");
        std::fs::set_permissions(&actual, std::fs::Permissions::from_mode(0o600))
            .expect("private control permissions");
        symlink(&actual, &symbolic).expect("control symlink");

        let symbolic_error = filesystem
            .open_control_file(&symbolic, PrivateBackupControlFileMode::ReadWrite)
            .expect_err("symbolic control file must fail closed");
        assert_eq!(
            sqlite_extended_code(symbolic_error),
            rusqlite::ffi::SQLITE_CANTOPEN
        );

        std::fs::set_permissions(&actual, std::fs::Permissions::from_mode(0o644))
            .expect("permissive control permissions");
        let permission_error = filesystem
            .open_existing_control_file(&actual)
            .expect_err("permissive control file must fail closed");
        assert_eq!(
            sqlite_extended_code(permission_error),
            rusqlite::ffi::SQLITE_PERM
        );
    }

    #[cfg(unix)]
    #[test]
    fn hardlinked_control_files_are_rejected_before_writable_access() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().expect("temporary directory");
        let filesystem = LocalBackupFilesystem;
        let external = root.path().join("external-control");
        let linked = root.path().join("linked-control");
        std::fs::write(&external, b"external").expect("external control file");
        std::fs::set_permissions(&external, std::fs::Permissions::from_mode(0o600))
            .expect("private external permissions");
        std::fs::hard_link(&external, &linked).expect("hard-linked control boundary");

        let writable_error = filesystem
            .open_control_file(&linked, PrivateBackupControlFileMode::ReadWrite)
            .expect_err("hard-linked writable control file must fail closed");
        assert_eq!(
            sqlite_extended_code(writable_error),
            rusqlite::ffi::SQLITE_PERM
        );

        let existing_error = filesystem
            .open_existing_control_file(&linked)
            .expect_err("hard-linked existing control file must fail closed");
        assert_eq!(
            sqlite_extended_code(existing_error),
            rusqlite::ffi::SQLITE_PERM
        );
        assert_eq!(
            std::fs::read(&external).expect("external file remains readable"),
            b"external"
        );
    }
}
