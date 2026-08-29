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
//! - Holds one non-blocking exclusive process fence for the store lifetime.
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
//!
//! Last Modified: v1.0.0-PrivateAtomicRecoveryIo - Initial restrictive atomic
//! generation file and process-fence implementation.
//! ============================================

#![cfg(unix)]

use std::fmt;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use nix::fcntl::{Flock, FlockArg};
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
pub(crate) struct PrivateAtomicRecoveryFile {
    directory: PathBuf,
    state_path: PathBuf,
    _fence: Flock<File>,
}

impl PrivateAtomicRecoveryFile {
    /// Opens one single-writer private recovery namespace.
    pub(crate) fn open(directory: &Path) -> Result<Self, PrivateRecoveryIoError> {
        prepare_private_directory(directory)?;
        let lock_path = directory.join(LOCK_FILE_NAME);
        let lock_file = open_private_regular_file(&lock_path, true, false, true)?;
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
        cleanup_owned_temp_files(directory)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            state_path: directory.join(STATE_FILE_NAME),
            _fence: fence,
        })
    }

    /// Reads the current bounded state without following links.
    pub(crate) fn read(
        &self,
        maximum_bytes: usize,
    ) -> Result<Option<Vec<u8>>, PrivateRecoveryIoError> {
        match fs::symlink_metadata(&self.state_path) {
            Ok(metadata) => validate_private_regular_metadata(&metadata)?,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        }
        let mut file = open_private_regular_file(&self.state_path, false, false, false)?;
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
        reject_unsafe_existing_state(&self.state_path)?;

        let temp_path = self.unique_temp_path();
        let result = (|| {
            let mut temporary = open_private_regular_file(&temp_path, true, true, true)?;
            temporary.write_all(bytes)?;
            temporary.sync_all()?;
            drop(temporary);
            fs::rename(&temp_path, &self.state_path)?;
            File::open(&self.directory)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        result
    }

    fn unique_temp_path(&self) -> PathBuf {
        let mut random = [0u8; 16];
        OsRng.fill_bytes(&mut random);
        self.directory
            .join(format!("{TEMP_FILE_PREFIX}{}", hex::encode(random)))
    }
}

impl fmt::Debug for PrivateAtomicRecoveryFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrivateAtomicRecoveryFile")
            .field("directory", &"<redacted>")
            .field("state_path", &"<redacted>")
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
    fs::set_permissions(
        directory,
        fs::Permissions::from_mode(PRIVATE_DIRECTORY_MODE),
    )?;
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    Ok(())
}

fn open_private_regular_file(
    path: &Path,
    create: bool,
    create_new: bool,
    writable: bool,
) -> Result<File, PrivateRecoveryIoError> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(writable)
        .create(create)
        .create_new(create_new)
        .mode(PRIVATE_FILE_MODE)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
    let file = options.open(path)?;
    if writable {
        file.set_permissions(fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
    }
    validate_private_regular_metadata(&file.metadata()?)?;
    Ok(file)
}

fn validate_private_regular_metadata(
    metadata: &fs::Metadata,
) -> Result<(), PrivateRecoveryIoError> {
    if !metadata.is_file() || metadata.nlink() != 1 || metadata.permissions().mode() & 0o077 != 0 {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    Ok(())
}

fn reject_unsafe_existing_state(path: &Path) -> Result<(), PrivateRecoveryIoError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => validate_private_regular_metadata(&metadata),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn cleanup_owned_temp_files(directory: &Path) -> Result<(), PrivateRecoveryIoError> {
    // [BLIND-VAULT-RECOVERY-TEMP-CLEANUP 2026-08-29 by Codex] Cleanup runs
    // only while holding the exclusive fence and only for our exact prefix.
    // Unknown files and directories are never touched.
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with(TEMP_FILE_PREFIX) {
            continue;
        }
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() || metadata.is_file() {
            fs::remove_file(entry.path())?;
        }
    }
    File::open(directory)?.sync_all()?;
    Ok(())
}
