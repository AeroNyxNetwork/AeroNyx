// ============================================
// File: crates/aeronyx-server/src/services/blind_vault_replica_recovery_io.rs
// ============================================
//! Restrictive crash-safe host I/O for private replica recovery generations.
//!
//! ## Creation Reason
//! The core recovery contract intentionally has no filesystem dependency. The
//! Linux node adapter needs one audited place for permissions, process fencing,
//! symlink rejection, bounded reads, and durable atomic replacement.
//!
//! ## Main Functionality
//! - Creates/verifies a private `0700` recovery directory.
//! - Pins the opened directory inode for every subsequent filesystem action.
//! - Holds one non-blocking exclusive process fence for the store lifetime.
//! - Requires directory and private files to belong to the effective user.
//! - Reads only bounded regular `0600` files without following symlinks.
//! - Publishes a temp file through file fsync, rename, and directory fsync.
//! - Removes only adapter-owned unfinished temp files after acquiring the lock.
//!
//! ## Dependencies
//! - `nix::fcntl::Flock`: process ownership fence.
//! - `rand::rngs::OsRng`: collision-resistant private temp names.
//! - Unix filesystem extensions: mode, link-count, and `O_NOFOLLOW` checks.
//!
//! ## Main Logical Flow
//! 1. Validate/create the root and normalize its mode to `0700`.
//! 2. Open the private lock file with `O_NOFOLLOW`, then acquire ownership.
//! 3. Remove only stale files matching the exact private temp prefix.
//! 4. Bound and validate every read before allocating the complete file.
//! 5. Publish replacement bytes durably and fsync the containing directory.
//!
//! ## Important Note For The Next Developer
//! - This adapter is Unix-only because AeroNyx nodes run on Linux hosts.
//! - Never weaken `O_NOFOLLOW`, regular-file, link-count, or mode checks.
//! - Never include private paths or file bytes in errors, Debug, or telemetry.
//! - Do not clean unknown files; only the exact temp prefix belongs to us.
//! - The process fence prevents concurrent writers, not privileged rollback.
//! - Never return to path-based state I/O after the directory FD is pinned.
//!
//! Last Modified: v1.2.0-OwnerFence - Rejected recovery directories and files
//! not owned by the process effective user.
//! v1.1.0-DirectoryFdFence - Bound lock, reads, temporary files,
//! cleanup, replacement, and fsync to one opened private directory inode.
//! v1.0.0-PrivateAtomicRecoveryIo - Initial restrictive atomic
//! generation file and process-fence implementation.
//! ============================================

#![cfg(unix)]

use std::fmt;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;

use nix::dir::{Dir, Type as DirectoryEntryType};
use nix::fcntl::{openat, renameat, AtFlags, Flock, FlockArg, OFlag};
use nix::sys::stat::{fstatat, Mode, SFlag};
use nix::unistd::{dup, unlinkat, UnlinkatFlags};
use rand::{rngs::OsRng, RngCore};
use thiserror::Error;

const PRIVATE_DIRECTORY_MODE: u32 = 0o700;
const PRIVATE_FILE_MODE: u32 = 0o600;
const STATE_FILE_NAME: &str = "recovery-state-v1.bin";
const LOCK_FILE_NAME: &str = ".recovery-state-v1.lock";
const TEMP_FILE_PREFIX: &str = ".recovery-state-v1.tmp.";

/// Host I/O failures without private path disclosure.
#[derive(Debug, Error)]
pub(crate) enum PrivateRecoveryIoError {
    #[error("blind vault replica recovery path is unsafe")]
    UnsafePath,
    #[error("blind vault replica recovery store is already owned")]
    AlreadyOwned,
    #[error("blind vault replica recovery file exceeds its bound")]
    TooLarge,
    #[error("blind vault replica recovery filesystem operation failed")]
    Filesystem(#[source] std::io::Error),
}

impl From<std::io::Error> for PrivateRecoveryIoError {
    fn from(error: std::io::Error) -> Self {
        Self::Filesystem(error)
    }
}

/// One privately fenced atomic state file.
///
/// [BLIND-VAULT-RECOVERY-DIRECTORY-FD 2026-08-30 by Codex] The directory
/// descriptor, lock, state file, and temporary files share one inode-rooted
/// namespace. Renaming or replacing the configured path cannot create a second
/// writer namespace behind the already-held process fence.
pub(crate) struct PrivateAtomicRecoveryFile {
    directory: File,
    _fence: Flock<File>,
}

impl PrivateAtomicRecoveryFile {
    /// Opens one single-writer private recovery namespace.
    pub(crate) fn open(directory: &Path) -> Result<Self, PrivateRecoveryIoError> {
        prepare_private_directory(directory)?;
        let directory = open_private_directory(directory)?;
        let lock_file =
            open_private_regular_file_at(&directory, LOCK_FILE_NAME, true, false, true)?;
        let fence =
            Flock::lock(lock_file, FlockArg::LockExclusiveNonblock).map_err(|(_file, error)| {
                if matches!(error, nix::errno::Errno::EWOULDBLOCK) {
                    PrivateRecoveryIoError::AlreadyOwned
                } else {
                    PrivateRecoveryIoError::Filesystem(std::io::Error::from_raw_os_error(
                        error as i32,
                    ))
                }
            })?;
        cleanup_owned_temp_files(&directory)?;
        Ok(Self {
            directory,
            _fence: fence,
        })
    }

    /// Reads the current bounded state without following links.
    pub(crate) fn read(
        &self,
        maximum_bytes: usize,
    ) -> Result<Option<Vec<u8>>, PrivateRecoveryIoError> {
        let mut file = match open_private_regular_file_at(
            &self.directory,
            STATE_FILE_NAME,
            false,
            false,
            false,
        ) {
            Ok(file) => file,
            Err(PrivateRecoveryIoError::Filesystem(error))
                if error.kind() == ErrorKind::NotFound =>
            {
                return Ok(None)
            }
            Err(error) => return Err(error),
        };
        let length = usize::try_from(file.metadata()?.len())
            .map_err(|_| PrivateRecoveryIoError::TooLarge)?;
        if length == 0 || length > maximum_bytes {
            return Err(PrivateRecoveryIoError::TooLarge);
        }
        let mut bytes = Vec::with_capacity(length);
        file.take((maximum_bytes as u64).saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() != length || bytes.len() > maximum_bytes {
            return Err(PrivateRecoveryIoError::TooLarge);
        }
        Ok(Some(bytes))
    }

    /// Atomically and durably replaces the current complete generation.
    pub(crate) fn replace(
        &self,
        bytes: &[u8],
        maximum_bytes: usize,
    ) -> Result<(), PrivateRecoveryIoError> {
        if bytes.is_empty() || bytes.len() > maximum_bytes {
            return Err(PrivateRecoveryIoError::TooLarge);
        }
        validate_existing_state_at(&self.directory)?;

        let temp_name = self.unique_temp_name();
        let result = (|| {
            let mut temporary =
                open_private_regular_file_at(&self.directory, &temp_name, true, true, true)?;
            temporary.write_all(bytes)?;
            temporary.sync_all()?;
            drop(temporary);
            renameat(
                Some(self.directory.as_raw_fd()),
                temp_name.as_str(),
                Some(self.directory.as_raw_fd()),
                STATE_FILE_NAME,
            )
            .map_err(nix_filesystem_error)?;
            self.directory.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = unlinkat(
                Some(self.directory.as_raw_fd()),
                temp_name.as_str(),
                UnlinkatFlags::NoRemoveDir,
            );
        }
        result
    }

    fn unique_temp_name(&self) -> String {
        let mut random = [0u8; 16];
        OsRng.fill_bytes(&mut random);
        format!("{TEMP_FILE_PREFIX}{}", hex::encode(random))
    }
}

impl fmt::Debug for PrivateAtomicRecoveryFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrivateAtomicRecoveryFile")
            .field("directory_fd", &"<redacted>")
            .field("process_fence", &"held")
            .finish_non_exhaustive()
    }
}

fn prepare_private_directory(directory: &Path) -> Result<(), PrivateRecoveryIoError> {
    match fs::symlink_metadata(directory) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(PrivateRecoveryIoError::UnsafePath);
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            let mut builder = DirBuilder::new();
            builder.recursive(true).mode(PRIVATE_DIRECTORY_MODE);
            builder.create(directory)?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn open_private_directory(path: &Path) -> Result<File, PrivateRecoveryIoError> {
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_DIRECTORY)
        .open(path)?;
    directory.set_permissions(fs::Permissions::from_mode(PRIVATE_DIRECTORY_MODE))?;
    let metadata = directory.metadata()?;
    // [BLIND-VAULT-RECOVERY-OWNER-FENCE 2026-08-30 by Codex] Private mode bits
    // are insufficient when a privileged process opens an attacker-owned
    // directory. Ownership is checked on the pinned inode and every child.
    if !metadata.is_dir()
        || metadata.uid() != effective_user_id()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    Ok(directory)
}

fn open_private_regular_file_at(
    directory: &File,
    name: &str,
    create: bool,
    create_new: bool,
    writable: bool,
) -> Result<File, PrivateRecoveryIoError> {
    let mut flags = OFlag::O_CLOEXEC | OFlag::O_NOFOLLOW;
    flags |= if writable {
        OFlag::O_RDWR
    } else {
        OFlag::O_RDONLY
    };
    if create {
        flags |= OFlag::O_CREAT;
    }
    if create_new {
        flags |= OFlag::O_EXCL;
    }
    let raw_fd = openat(
        Some(directory.as_raw_fd()),
        name,
        flags,
        Mode::from_bits_truncate(PRIVATE_FILE_MODE),
    )
    .map_err(|error| {
        if error == nix::errno::Errno::ELOOP {
            PrivateRecoveryIoError::UnsafePath
        } else {
            PrivateRecoveryIoError::Filesystem(nix_filesystem_error(error))
        }
    })?;
    // SAFETY: `openat` returned a new owned descriptor and this is its sole
    // owner. `File` closes it exactly once when dropped.
    let file = unsafe { File::from_raw_fd(raw_fd) };
    if writable {
        file.set_permissions(fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
    }
    validate_private_regular_metadata(&file.metadata()?)?;
    Ok(file)
}

fn validate_private_regular_metadata(
    metadata: &fs::Metadata,
) -> Result<(), PrivateRecoveryIoError> {
    if !metadata.is_file()
        || metadata.uid() != effective_user_id()
        || metadata.nlink() != 1
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    Ok(())
}

fn validate_existing_state_at(directory: &File) -> Result<(), PrivateRecoveryIoError> {
    match open_private_regular_file_at(directory, STATE_FILE_NAME, false, false, false) {
        Ok(_) => Ok(()),
        Err(PrivateRecoveryIoError::Filesystem(error)) if error.kind() == ErrorKind::NotFound => {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn cleanup_owned_temp_files(directory: &File) -> Result<(), PrivateRecoveryIoError> {
    // [BLIND-VAULT-RECOVERY-TEMP-CLEANUP 2026-08-29 by Codex] Cleanup runs
    // only while holding the exclusive fence and only for our exact prefix.
    // Unknown files and directories are never touched.
    let duplicated = dup(directory.as_raw_fd()).map_err(nix_filesystem_error)?;
    let mut entries = Dir::from_fd(duplicated).map_err(nix_filesystem_error)?;
    for entry in entries.iter() {
        let entry = entry.map_err(nix_filesystem_error)?;
        let Some(name) = entry.file_name().to_str().ok() else {
            continue;
        };
        if !name.starts_with(TEMP_FILE_PREFIX) {
            continue;
        }
        let removable = match entry.file_type() {
            Some(DirectoryEntryType::File | DirectoryEntryType::Symlink) => true,
            Some(_) => false,
            None => {
                let metadata = fstatat(
                    Some(directory.as_raw_fd()),
                    entry.file_name(),
                    AtFlags::AT_SYMLINK_NOFOLLOW,
                )
                .map_err(nix_filesystem_error)?;
                let kind = SFlag::from_bits_truncate(metadata.st_mode);
                kind == SFlag::S_IFREG || kind == SFlag::S_IFLNK
            }
        };
        if removable {
            unlinkat(
                Some(directory.as_raw_fd()),
                entry.file_name(),
                UnlinkatFlags::NoRemoveDir,
            )
            .map_err(nix_filesystem_error)?;
        }
    }
    directory.sync_all()?;
    Ok(())
}

fn nix_filesystem_error(error: nix::errno::Errno) -> std::io::Error {
    std::io::Error::from_raw_os_error(error as i32)
}

fn effective_user_id() -> u32 {
    // SAFETY: `geteuid` has no preconditions and does not dereference memory.
    unsafe { nix::libc::geteuid() }
}
