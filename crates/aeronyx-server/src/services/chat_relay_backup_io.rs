// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_io.rs
// ============================================
// Version: 1.4.0-DirectorySyncIdentity
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
//   - Proves an exact read-only pair of canonical crash-publication links.
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
//   - Never relax the single-file `nlink == 1` rule; only the paired read
//     capability may admit two canonical names for one exact `nlink == 2` inode.
//
// Last Modified:
//   v1.4.0-DirectorySyncIdentity - Opens directory fsync targets without
//     following final symlinks or waiting on special-file peers
//   v1.3.0-PairedLinkRecovery - Added identity-bound read-only recovery for
//     exact two-name hard-link publication states without weakening writers
//   v1.2.0-DirectoryIdentity - Pinned and validated the effective-user-owned
//     directory inode before normalizing its private mode
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

/// Stable storage identity for one fully validated private control inode.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrivateBackupControlFileIdentity {
    device_id: u64,
    inode: u64,
}

/// Portable placeholder; paired-link recovery remains Unix-only.
#[cfg(not(unix))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrivateBackupControlFileIdentity;

/// Two read-only descriptors proven to be the only names of one private inode.
#[derive(Debug)]
pub(super) struct PrivateBackupControlFilePair {
    pub(super) first: File,
    pub(super) second: File,
    pub(super) identity: PrivateBackupControlFileIdentity,
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

    /// Opens two canonical read-only names only when they are the exact two
    /// links to one owner-private regular inode.
    fn open_existing_control_file_pair(
        &self,
        first: &Path,
        second: &Path,
    ) -> ChatRelayResult<Option<PrivateBackupControlFilePair>>;

    /// Removes one exact paired link after revalidating its storage identity.
    fn remove_verified_control_link(
        &self,
        parent: &Path,
        path: &Path,
        expected: PrivateBackupControlFileIdentity,
    ) -> ChatRelayResult<()>;

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
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_DIRECTORY)
        .open(path)
        .map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup boundary is not a private directory",
            )
        })?;
    let metadata = directory.metadata().map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_IOERR,
            "unable to inspect private relay backup directory",
        )
    })?;
    // [CHAT-RELAY-BACKUP-DIRECTORY-IDENTITY 2026-08-31 by Codex] Pin and
    // validate the directory inode before changing permissions. Path-based
    // chmod could otherwise mutate a foreign or raced replacement directory.
    if !metadata.is_dir() || metadata.uid() != effective_user_id() {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_PERM,
            "relay backup boundary is not an owner-private directory",
        ));
    }
    directory
        .set_permissions(std::fs::Permissions::from_mode(0o700))
        .map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "unable to restrict relay backup directory permissions",
            )
        })?;
    let normalized = directory.metadata().map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_IOERR,
            "unable to re-inspect private relay backup directory",
        )
    })?;
    if !normalized.is_dir()
        || normalized.uid() != effective_user_id()
        || normalized.permissions().mode() & 0o777 != 0o700
    {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_PERM,
            "relay backup directory permission normalization was not retained",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn restrict_directory_permissions(_path: &Path) -> ChatRelayResult<()> {
    Ok(())
}

#[cfg(unix)]
fn effective_user_id() -> u32 {
    // SAFETY: `geteuid` has no preconditions and does not access memory.
    unsafe { nix::libc::geteuid() }
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrivateControlFileFacts {
    identity: PrivateBackupControlFileIdentity,
    owner_id: u32,
    link_count: u64,
    mode: u32,
}

#[cfg(unix)]
fn private_control_file_facts(file: &File) -> ChatRelayResult<PrivateControlFileFacts> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

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
    Ok(PrivateControlFileFacts {
        identity: PrivateBackupControlFileIdentity {
            device_id: metadata.dev(),
            inode: metadata.ino(),
        },
        owner_id: metadata.uid(),
        link_count: metadata.nlink(),
        mode: metadata.permissions().mode(),
    })
}

#[cfg(unix)]
fn validate_owner_private_facts(facts: PrivateControlFileFacts) -> ChatRelayResult<()> {
    if facts.owner_id != effective_user_id() || facts.mode & 0o077 != 0 {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_PERM,
            "relay backup control file is not owner-private",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn paired_private_control_identity(
    first: PrivateControlFileFacts,
    second: PrivateControlFileFacts,
) -> ChatRelayResult<Option<PrivateBackupControlFileIdentity>> {
    validate_owner_private_facts(first)?;
    validate_owner_private_facts(second)?;
    if first.identity != second.identity {
        return Ok(None);
    }
    // [CHAT-RELAY-BACKUP-PAIRED-LINK-RECOVERY 2026-08-31 by Codex] Both
    // expected canonical names must account for the inode's entire link count.
    // A third or external alias is unknown state and remains fail-closed.
    if first.link_count != 2 || second.link_count != 2 {
        return Err(backup_io_error(
            rusqlite::ffi::SQLITE_PERM,
            "relay backup control link pair is not exclusive",
        ));
    }
    Ok(Some(first.identity))
}

fn open_existing_control_file_unvalidated(path: &Path) -> ChatRelayResult<Option<File>> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    match options.open(path) {
        Ok(file) => Ok(Some(file)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "unable to open private relay backup control file",
        )),
    }
}

fn canonical_control_path(path: &Path) -> ChatRelayResult<(PathBuf, PathBuf)> {
    let parent = path.parent().ok_or_else(|| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "relay backup control file has no private parent",
        )
    })?;
    let file_name = path.file_name().ok_or_else(|| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "relay backup control file has no private name",
        )
    })?;
    let canonical_parent = std::fs::canonicalize(parent).map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_CANTOPEN,
            "unable to resolve private relay backup control parent",
        )
    })?;
    let canonical_path = canonical_parent.join(file_name);
    Ok((canonical_parent, canonical_path))
}

#[cfg(unix)]
fn open_directory_for_sync(
    path: &Path,
    sqlite_code: i32,
    message: &'static str,
) -> ChatRelayResult<File> {
    use std::os::unix::fs::OpenOptionsExt;

    // [CHAT-RELAY-BACKUP-DIRECTORY-SYNC 2026-09-01 by Codex] Durability must
    // apply to the directory inode the caller named, not a final symlink or a
    // special file that can wait for an external peer before type rejection.
    // O_DIRECTORY performs the type check during open; O_NONBLOCK keeps the
    // failure bounded if a raced replacement is a FIFO or device boundary.
    OpenOptions::new()
        .read(true)
        .custom_flags(
            nix::libc::O_CLOEXEC
                | nix::libc::O_NOFOLLOW
                | nix::libc::O_DIRECTORY
                | nix::libc::O_NONBLOCK,
        )
        .open(path)
        .map_err(|_| backup_io_error(sqlite_code, message))
}

#[cfg(not(unix))]
fn open_directory_for_sync(
    path: &Path,
    sqlite_code: i32,
    message: &'static str,
) -> ChatRelayResult<File> {
    let file = File::open(path).map_err(|_| backup_io_error(sqlite_code, message))?;
    let metadata = file
        .metadata()
        .map_err(|_| backup_io_error(sqlite_code, message))?;
    if !metadata.is_dir() {
        return Err(backup_io_error(sqlite_code, message));
    }
    Ok(file)
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
        // [CHAT-RELAY-BACKUP-FILE-IDENTITY 2026-08-31 by Codex] A private
        // mode does not prove exclusive ownership of an inode. Reject foreign
        // ownership and hard links before a caller receives a writable handle.
        let facts = private_control_file_facts(file)?;
        validate_owner_private_facts(facts)?;
        if facts.link_count != 1 {
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

        let Some(file) = open_existing_control_file_unvalidated(path)? else {
            return Ok(None);
        };
        validate_private_control_file(&file)?;
        Ok(Some(file))
    }

    fn open_existing_control_file_pair(
        &self,
        first: &Path,
        second: &Path,
    ) -> ChatRelayResult<Option<PrivateBackupControlFilePair>> {
        #[cfg(not(unix))]
        {
            let _ = (first, second);
            return Ok(None);
        }

        #[cfg(unix)]
        {
            let (first_parent, first_path) = canonical_control_path(first)?;
            let (second_parent, second_path) = canonical_control_path(second)?;
            if first_parent != second_parent || first_path == second_path {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup control link pair is not canonical",
                ));
            }
            let Some(first_file) = open_existing_control_file_unvalidated(&first_path)? else {
                return Ok(None);
            };
            let Some(second_file) = open_existing_control_file_unvalidated(&second_path)? else {
                return Ok(None);
            };
            let Some(identity) = paired_private_control_identity(
                private_control_file_facts(&first_file)?,
                private_control_file_facts(&second_file)?,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(PrivateBackupControlFilePair {
                first: first_file,
                second: second_file,
                identity,
            }))
        }
    }

    fn remove_verified_control_link(
        &self,
        parent: &Path,
        path: &Path,
        expected: PrivateBackupControlFileIdentity,
    ) -> ChatRelayResult<()> {
        #[cfg(not(unix))]
        {
            let _ = (parent, path, expected);
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "paired relay backup control recovery is unavailable",
            ));
        }

        #[cfg(unix)]
        {
            let canonical_parent = std::fs::canonicalize(parent).map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to resolve relay backup recovery parent",
                )
            })?;
            let (path_parent, canonical_path) = canonical_control_path(path)?;
            if path_parent != canonical_parent {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup recovery link escaped its private parent",
                ));
            }
            let file =
                open_existing_control_file_unvalidated(&canonical_path)?.ok_or_else(|| {
                    backup_io_error(
                        rusqlite::ffi::SQLITE_CORRUPT,
                        "relay backup recovery link became unavailable",
                    )
                })?;
            let facts = private_control_file_facts(&file)?;
            validate_owner_private_facts(facts)?;
            if facts.identity != expected || facts.link_count != 2 {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup recovery link identity changed",
                ));
            }
            std::fs::remove_file(&canonical_path).map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR_DELETE,
                    "unable to remove verified relay backup recovery link",
                )
            })
        }
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
        let code = rusqlite::ffi::SQLITE_IOERR_FSYNC;
        let message = "unable to durably sync relay backup directory";
        open_directory_for_sync(backup_directory, code, message)?
            .sync_all()
            .map_err(|_| backup_io_error(code, message))
    }

    #[cfg(unix)]
    fn sync_backup_parent(&self, parent: &Path) -> ChatRelayResult<()> {
        let code = rusqlite::ffi::SQLITE_IOERR;
        let message = "unable to synchronize relay backup directory";
        open_directory_for_sync(parent, code, message)?
            .sync_all()
            .map_err(|_| backup_io_error(code, message))
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

    #[cfg(unix)]
    #[test]
    fn existing_owned_directory_is_normalized_through_its_opened_inode() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().expect("temporary directory");
        let backup = root.path().join(".aeronyx-relay-backups");
        std::fs::create_dir(&backup).expect("backup directory");
        std::fs::set_permissions(&backup, std::fs::Permissions::from_mode(0o755))
            .expect("permissive backup mode");

        LocalBackupFilesystem
            .ensure_private_directory(&backup)
            .expect("normalize owned backup directory");

        let mode = std::fs::metadata(&backup)
            .expect("normalized backup metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700);
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

    #[cfg(unix)]
    #[test]
    fn canonical_paired_links_are_identity_bound_and_recoverable() {
        use std::os::unix::fs::MetadataExt;

        let root = tempfile::tempdir().expect("temporary paired-link directory");
        let canonical_root = std::fs::canonicalize(root.path()).expect("canonical paired parent");
        let active = root.path().join("active-control");
        let segment = canonical_root.join("segment-control");
        let filesystem = LocalBackupFilesystem;
        filesystem
            .reserve_private_file(&active)
            .expect("reserve private active control");
        std::fs::hard_link(&active, &segment).expect("publish exact paired link");

        let pair = filesystem
            .open_existing_control_file_pair(&active, &segment)
            .expect("inspect exact paired link")
            .expect("paired link must be recognized");
        assert_eq!(
            pair.first.metadata().expect("first metadata").ino(),
            pair.second.metadata().expect("second metadata").ino()
        );
        assert!(filesystem.open_existing_control_file(&active).is_err());
        let identity = pair.identity;
        drop(pair);

        filesystem
            .remove_verified_control_link(root.path(), &active, identity)
            .expect("remove identity-bound active link");
        assert!(!active.exists());
        assert_eq!(
            std::fs::metadata(&segment)
                .expect("retained segment metadata")
                .nlink(),
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn paired_link_facts_reject_uid_mode_third_link_and_distinct_inode() {
        let identity = PrivateBackupControlFileIdentity {
            device_id: 7,
            inode: 11,
        };
        let valid = PrivateControlFileFacts {
            identity,
            owner_id: effective_user_id(),
            link_count: 2,
            mode: 0o100600,
        };
        assert_eq!(
            paired_private_control_identity(valid, valid).expect("valid pair facts"),
            Some(identity)
        );

        for invalid in [
            PrivateControlFileFacts {
                owner_id: effective_user_id().wrapping_add(1),
                ..valid
            },
            PrivateControlFileFacts {
                mode: 0o100644,
                ..valid
            },
            PrivateControlFileFacts {
                link_count: 3,
                ..valid
            },
        ] {
            assert!(paired_private_control_identity(valid, invalid).is_err());
        }
        assert_eq!(
            paired_private_control_identity(
                valid,
                PrivateControlFileFacts {
                    identity: PrivateBackupControlFileIdentity {
                        device_id: 7,
                        inode: 12,
                    },
                    ..valid
                }
            )
            .expect("distinct private files are not a pair"),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn paired_link_identity_or_mode_drift_is_rejected_before_unlink() {
        use std::os::unix::fs::PermissionsExt;

        for replace_identity in [false, true] {
            let root = tempfile::tempdir().expect("temporary paired-link drift directory");
            let filesystem = LocalBackupFilesystem;
            let active = root.path().join("active-control");
            let segment = root.path().join("segment-control");
            filesystem
                .reserve_private_file(&active)
                .expect("reserve drift active control");
            std::fs::hard_link(&active, &segment).expect("publish drift segment link");
            let pair = filesystem
                .open_existing_control_file_pair(&active, &segment)
                .expect("inspect drift pair")
                .expect("drift pair starts valid");
            let identity = pair.identity;
            drop(pair);

            if replace_identity {
                std::fs::remove_file(&active).expect("remove original active name");
                filesystem
                    .reserve_private_file(&active)
                    .expect("replace active with distinct inode");
            } else {
                std::fs::set_permissions(&active, std::fs::Permissions::from_mode(0o644))
                    .expect("drift paired inode mode");
            }

            assert!(filesystem
                .remove_verified_control_link(root.path(), &active, identity)
                .is_err());
            assert!(active.exists() && segment.exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn fifo_and_external_or_third_links_fail_without_removal() {
        use nix::sys::stat::Mode;

        let root = tempfile::tempdir().expect("temporary unsafe-link directory");
        let outside = tempfile::tempdir().expect("external alias directory");
        let filesystem = LocalBackupFilesystem;
        let active = root.path().join("active-control");
        let segment = root.path().join("segment-control");
        let third = root.path().join("third-control");
        filesystem
            .reserve_private_file(&active)
            .expect("reserve active control");
        std::fs::hard_link(&active, &segment).expect("publish segment link");
        std::fs::hard_link(&active, &third).expect("publish forbidden third link");
        assert!(filesystem
            .open_existing_control_file_pair(&active, &segment)
            .is_err());
        assert!(active.exists() && segment.exists() && third.exists());

        let external_source = root.path().join("external-source");
        let expected_alias = root.path().join("expected-alias");
        filesystem
            .reserve_private_file(&external_source)
            .expect("reserve externally-linked control");
        let external_alias = outside.path().join("external-alias");
        std::fs::hard_link(&external_source, &external_alias)
            .expect("create external hard-link alias");
        filesystem
            .reserve_private_file(&expected_alias)
            .expect("reserve distinct expected alias");
        assert!(filesystem
            .open_existing_control_file_pair(&external_source, &external_alias)
            .is_err());
        assert!(filesystem
            .open_existing_control_file_pair(&external_source, &expected_alias)
            .expect("distinct in-parent names are inspectable")
            .is_none());
        assert!(external_source.exists() && external_alias.exists() && expected_alias.exists());

        let fifo = root.path().join("fifo-control");
        nix::unistd::mkfifo(&fifo, Mode::S_IRUSR | Mode::S_IWUSR).expect("create private FIFO");
        assert!(filesystem.open_existing_control_file(&fifo).is_err());
        assert!(fifo.exists());
    }

    #[cfg(unix)]
    #[test]
    fn directory_sync_rejects_symlink_and_fifo_boundaries() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temporary sync directory");
        let filesystem = LocalBackupFilesystem;
        let actual = root.path().join("actual-directory");
        let symbolic = root.path().join("symbolic-directory");
        let fifo = root.path().join("directory-fifo");
        std::fs::create_dir(&actual).expect("actual sync directory");
        symlink(&actual, &symbolic).expect("directory sync symlink");
        nix::unistd::mkfifo(
            &fifo,
            nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
        )
        .expect("directory sync FIFO");

        filesystem
            .sync_backup_parent(&actual)
            .expect("real directory remains synchronizable");
        for unsafe_path in [&symbolic, &fifo] {
            let error = filesystem
                .sync_backup_parent(unsafe_path)
                .expect_err("unsafe directory sync target must fail closed");
            assert_eq!(sqlite_extended_code(error), rusqlite::ffi::SQLITE_IOERR);
        }
        assert!(symbolic.is_symlink());
        assert!(fifo.exists());
    }
}
