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
//! - Re-confirms current file and directory durability after ambiguous errors.
//! - Removes only exact adapter-generated unfinished temp files after locking.
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
//! - Do not clean unknown files; only the exact generated temp grammar is ours.
//! - The process fence prevents concurrent writers, not privileged rollback.
//! - Never return to path-based state I/O after the directory FD is pinned.
//!
//! Last Modified: v1.10.0-TypedPublicationCleanup - Bound failed-publish
//! cleanup to the pinned inode and the pre-rename publication phase.
//! v1.9.0-ValidatedTempOwnership - Required exact unfinished
//! regular-file ownership proof before cleanup may unlink its directory entry.
//! v1.8.0-ValidatedFileIdentity - Required regular-file,
//! effective-owner, and single-link proof before writable mode normalization.
//! v1.7.0-OwnedTempGrammar - Restricted unfinished cleanup to
//! the exact generated lowercase-hex filename grammar.
//! v1.6.0-ParentDirectoryDurability - Required every traversed
//! recovery directory entry to be durable in its pinned parent before use.
//! v1.5.0-PortableNixMode - Enabled directory enumeration and made fixed
//! private modes portable across Unix `mode_t` widths.
//! v1.4.0-DurabilityConfirmation - Added descriptor-pinned file
//! and directory synchronization for exact idempotent transition retries.
//! v1.3.0-ComponentWalk - Rejected symlinks in every configured
//! path component through descriptor-relative creation and traversal.
//! v1.2.0-OwnerFence - Rejected recovery directories and files
//! not owned by the process effective user.
//! v1.1.0-DirectoryFdFence - Bound lock, reads, temporary files,
//! cleanup, replacement, and fsync to one opened private directory inode.
//! v1.0.0-PrivateAtomicRecoveryIo - Initial restrictive atomic
//! generation file and process-fence implementation.
//! ============================================

#![cfg(unix)]

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path};

use nix::dir::{Dir, Type as DirectoryEntryType};
use nix::fcntl::{openat, renameat, AtFlags, Flock, FlockArg, OFlag};
use nix::sys::stat::{fstatat, mkdirat, Mode, SFlag};
use nix::unistd::{dup, unlinkat, UnlinkatFlags};
use rand::{rngs::OsRng, RngCore};
use thiserror::Error;

const PRIVATE_DIRECTORY_MODE: u32 = 0o700;
const PRIVATE_FILE_MODE: u32 = 0o600;
const STATE_FILE_NAME: &str = "recovery-state-v1.bin";
const LOCK_FILE_NAME: &str = ".recovery-state-v1.lock";
const TEMP_FILE_PREFIX: &str = ".recovery-state-v1.tmp.";
const TEMP_FILE_RANDOM_BYTES: usize = 16;
const TEMP_FILE_HEX_LENGTH: usize = TEMP_FILE_RANDOM_BYTES * 2;

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
        let directory = open_or_create_private_directory(directory)?;
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
        let temp_name = self.unique_temp_name();
        self.replace_with_temp_name_and_durability(
            bytes,
            maximum_bytes,
            &temp_name,
            &HostPublicationDirectoryDurability,
        )
    }

    fn replace_with_temp_name_and_durability(
        &self,
        bytes: &[u8],
        maximum_bytes: usize,
        temp_name: &str,
        durability: &impl PublicationDirectoryDurability,
    ) -> Result<(), PrivateRecoveryIoError> {
        if bytes.is_empty() || bytes.len() > maximum_bytes {
            return Err(PrivateRecoveryIoError::TooLarge);
        }
        if !is_owned_temp_file_name(temp_name.as_bytes()) {
            return Err(PrivateRecoveryIoError::UnsafePath);
        }
        validate_existing_state_at(&self.directory)?;

        let mut phase = PublicationPhase::TempNotCreated;
        let result = (|| {
            let temporary =
                open_private_regular_file_at(&self.directory, temp_name, true, true, true)?;
            phase = PublicationPhase::TempOpened(PinnedPublicationTemp::new(temporary)?);
            {
                let temporary = phase.opened_mut()?;
                temporary.file.write_all(bytes)?;
                temporary.file.sync_all()?;
            }
            renameat(
                Some(self.directory.as_raw_fd()),
                temp_name,
                Some(self.directory.as_raw_fd()),
                STATE_FILE_NAME,
            )
            .map_err(nix_filesystem_error)?;
            phase = PublicationPhase::Renamed;
            durability.sync_after_rename(&self.directory)?;
            Ok(())
        })();
        if let Err(publication_error) = result {
            // [BLIND-VAULT-RECOVERY-PUBLICATION-CLEANUP 2026-08-31 by Codex]
            // The publication failure is authoritative. Cleanup ambiguity
            // leaves its exact entry for the fenced restart cleanup rather
            // than replacing the original error or unlinking unknown state.
            let _cleanup_result = phase.cleanup_failed_publish(&self.directory, temp_name);
            return Err(publication_error);
        }
        Ok(())
    }

    /// Re-confirms the visible generation across an ambiguous prior result.
    ///
    /// [BLIND-VAULT-RECOVERY-DURABILITY-CONFIRMATION 2026-08-30 by Codex]
    /// A matching file after `renameat` does not prove its directory entry
    /// survived power loss. Exact idempotent retries synchronize both the
    /// validated state file and the already pinned containing directory.
    pub(crate) fn confirm_current_durable(&self) -> Result<(), PrivateRecoveryIoError> {
        let state =
            open_private_regular_file_at(&self.directory, STATE_FILE_NAME, false, false, false)?;
        state.sync_all()?;
        self.directory.sync_all()?;
        Ok(())
    }

    fn unique_temp_name(&self) -> String {
        let mut random = [0u8; TEMP_FILE_RANDOM_BYTES];
        OsRng.fill_bytes(&mut random);
        format!("{TEMP_FILE_PREFIX}{}", hex::encode(random))
    }
}

trait PublicationDirectoryDurability {
    fn sync_after_rename(&self, directory: &File) -> std::io::Result<()>;
}

struct HostPublicationDirectoryDurability;

impl PublicationDirectoryDurability for HostPublicationDirectoryDurability {
    fn sync_after_rename(&self, directory: &File) -> std::io::Result<()> {
        directory.sync_all()
    }
}

/// The only three ownership states of one replacement attempt.
///
/// [BLIND-VAULT-RECOVERY-PUBLICATION-CLEANUP 2026-08-31 by Codex] A failed
/// attempt can delete a name only while it retains the descriptor and identity
/// acquired by its successful exclusive create. After rename, the old name is
/// unowned even when the parent-directory durability result is ambiguous.
enum PublicationPhase {
    TempNotCreated,
    TempOpened(PinnedPublicationTemp),
    Renamed,
}

impl PublicationPhase {
    fn opened_mut(&mut self) -> Result<&mut PinnedPublicationTemp, PrivateRecoveryIoError> {
        match self {
            Self::TempOpened(temporary) => Ok(temporary),
            Self::TempNotCreated | Self::Renamed => Err(PrivateRecoveryIoError::UnsafePath),
        }
    }

    fn cleanup_failed_publish(
        self,
        directory: &File,
        temp_name: &str,
    ) -> Result<(), PrivateRecoveryIoError> {
        let Self::TempOpened(temporary) = self else {
            return Ok(());
        };
        temporary.identity.validate_entry_at(directory, temp_name)?;
        unlinkat(
            Some(directory.as_raw_fd()),
            temp_name,
            UnlinkatFlags::NoRemoveDir,
        )
        .map_err(nix_filesystem_error)?;
        directory.sync_all()?;
        Ok(())
    }
}

struct PinnedPublicationTemp {
    file: File,
    identity: PinnedPrivateFileIdentity,
}

impl PinnedPublicationTemp {
    fn new(file: File) -> Result<Self, PrivateRecoveryIoError> {
        let identity = PinnedPrivateFileIdentity::from_file(&file)?;
        Ok(Self { file, identity })
    }
}

struct PinnedPrivateFileIdentity {
    device: u64,
    inode: u64,
}

impl PinnedPrivateFileIdentity {
    fn from_file(file: &File) -> Result<Self, PrivateRecoveryIoError> {
        let metadata = file.metadata()?;
        validate_exact_private_regular_metadata(&metadata)?;
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }

    fn validate_entry_at(
        &self,
        directory: &File,
        name: &str,
    ) -> Result<(), PrivateRecoveryIoError> {
        let metadata = fstatat(
            Some(directory.as_raw_fd()),
            name,
            AtFlags::AT_SYMLINK_NOFOLLOW,
        )
        .map_err(nix_filesystem_error)?;
        let kind = SFlag::from_bits_truncate(metadata.st_mode);
        if kind != SFlag::S_IFREG
            || u64::try_from(metadata.st_dev).ok() != Some(self.device)
            || u64::try_from(metadata.st_ino).ok() != Some(self.inode)
            || metadata.st_uid != effective_user_id()
            || metadata.st_nlink != 1
            || metadata.st_mode & (0o7777 as nix::libc::mode_t)
                != PRIVATE_FILE_MODE as nix::libc::mode_t
        {
            return Err(PrivateRecoveryIoError::UnsafePath);
        }
        Ok(())
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

fn open_or_create_private_directory(path: &Path) -> Result<File, PrivateRecoveryIoError> {
    let components = private_directory_components(path)?;
    let anchor = if path.is_absolute() { "/" } else { "." };
    let mut directory = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_DIRECTORY)
        .open(anchor)?;

    // [BLIND-VAULT-RECOVERY-COMPONENT-WALK 2026-08-30 by Codex] Resolve and
    // create every component relative to a previously opened directory FD.
    // This closes the parent-symlink and path-swap window left by recursive
    // path creation while retaining support for absolute and relative paths.
    for component in components {
        directory = open_or_create_directory_at(&directory, &component)?;
    }

    let metadata = directory.metadata()?;
    // [BLIND-VAULT-RECOVERY-OWNER-FENCE 2026-08-30 by Codex] Private mode bits
    // are insufficient when a privileged process opens an attacker-owned
    // directory. Ownership is checked on the pinned inode and every child.
    if !metadata.is_dir() || metadata.uid() != effective_user_id() {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    directory.set_permissions(fs::Permissions::from_mode(PRIVATE_DIRECTORY_MODE))?;
    if directory.metadata()?.permissions().mode() & 0o077 != 0 {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    Ok(directory)
}

trait ParentDirectoryDurability {
    fn sync_parent(&self, parent: &File) -> std::io::Result<()>;
}

struct HostParentDirectoryDurability;

impl ParentDirectoryDurability for HostParentDirectoryDurability {
    fn sync_parent(&self, parent: &File) -> std::io::Result<()> {
        parent.sync_all()
    }
}

fn private_directory_components(path: &Path) -> Result<Vec<OsString>, PrivateRecoveryIoError> {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(name) => components.push(name.to_os_string()),
            Component::ParentDir | Component::Prefix(_) => {
                return Err(PrivateRecoveryIoError::UnsafePath)
            }
        }
    }
    if components.is_empty() {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    Ok(components)
}

fn open_or_create_directory_at(
    parent: &File,
    name: &OsStr,
) -> Result<File, PrivateRecoveryIoError> {
    open_or_create_directory_at_with(parent, name, &HostParentDirectoryDurability)
}

fn open_or_create_directory_at_with(
    parent: &File,
    name: &OsStr,
    durability: &impl ParentDirectoryDurability,
) -> Result<File, PrivateRecoveryIoError> {
    let directory = match open_directory_at(parent, name) {
        Ok(directory) => Ok(directory),
        Err(PrivateRecoveryIoError::Filesystem(error)) if error.kind() == ErrorKind::NotFound => {
            match mkdirat(
                Some(parent.as_raw_fd()),
                name,
                // [BLIND-VAULT-RECOVERY-IO-BUILD 2026-08-31 by Codex] These
                // fixed literals fit every supported Unix `mode_t` width.
                Mode::from_bits_truncate(PRIVATE_DIRECTORY_MODE as nix::libc::mode_t),
            ) {
                Ok(()) | Err(nix::errno::Errno::EEXIST) => open_directory_at(parent, name),
                Err(error) => Err(PrivateRecoveryIoError::Filesystem(nix_filesystem_error(
                    error,
                ))),
            }
        }
        Err(error) => Err(error),
    }?;

    // [BLIND-VAULT-RECOVERY-IO 2026-08-31 by Codex] A synced child does not
    // make its directory entry durable. Synchronize the pinned parent even
    // for an existing/raced entry so a retry cannot bypass an ambiguous prior
    // mkdir fsync failure.
    durability.sync_parent(parent)?;
    Ok(directory)
}

fn open_directory_at(parent: &File, name: &OsStr) -> Result<File, PrivateRecoveryIoError> {
    let raw_fd = openat(
        Some(parent.as_raw_fd()),
        name,
        OFlag::O_RDONLY | OFlag::O_CLOEXEC | OFlag::O_NOFOLLOW | OFlag::O_DIRECTORY,
        Mode::empty(),
    )
    .map_err(|error| match error {
        nix::errno::Errno::ELOOP | nix::errno::Errno::ENOTDIR => PrivateRecoveryIoError::UnsafePath,
        _ => PrivateRecoveryIoError::Filesystem(nix_filesystem_error(error)),
    })?;
    // SAFETY: `openat` returned a new owned descriptor and this is its sole
    // owner. `File` closes it exactly once when dropped.
    Ok(unsafe { File::from_raw_fd(raw_fd) })
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
        Mode::from_bits_truncate(PRIVATE_FILE_MODE as nix::libc::mode_t),
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
    let identity = ValidatedPrivateFileIdentity::new(&file)?;
    if writable {
        identity.normalize_mode()?;
    }
    identity.revalidate()?;
    Ok(file)
}

/// Proof that one opened descriptor had private-file identity before mutation.
///
/// [BLIND-VAULT-RECOVERY-IO 2026-08-31 by Codex] Writable mode normalization
/// must be reachable only after regular-file, effective-owner, and single-link
/// validation. A final metadata read rechecks both identity and private mode.
struct ValidatedPrivateFileIdentity<'file> {
    file: &'file File,
}

impl<'file> ValidatedPrivateFileIdentity<'file> {
    fn new(file: &'file File) -> Result<Self, PrivateRecoveryIoError> {
        validate_private_regular_identity(&file.metadata()?)?;
        Ok(Self { file })
    }

    fn normalize_mode(&self) -> Result<(), PrivateRecoveryIoError> {
        self.file
            .set_permissions(fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
        Ok(())
    }

    fn revalidate(&self) -> Result<(), PrivateRecoveryIoError> {
        let metadata = self.file.metadata()?;
        validate_private_regular_identity(&metadata)?;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(PrivateRecoveryIoError::UnsafePath);
        }
        Ok(())
    }
}

fn validate_private_regular_identity(
    metadata: &fs::Metadata,
) -> Result<(), PrivateRecoveryIoError> {
    if !metadata.is_file() || metadata.uid() != effective_user_id() || metadata.nlink() != 1 {
        return Err(PrivateRecoveryIoError::UnsafePath);
    }
    Ok(())
}

fn validate_exact_private_regular_metadata(
    metadata: &fs::Metadata,
) -> Result<(), PrivateRecoveryIoError> {
    validate_private_regular_identity(metadata)?;
    if metadata.permissions().mode() & 0o7777 != PRIVATE_FILE_MODE {
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

// [BLIND-VAULT-RECOVERY-IO 2026-08-31 by Codex] A prefix alone does not prove
// adapter ownership. Generated names are exactly the ASCII prefix followed by
// one lowercase hexadecimal encoding of the fixed-size random nonce.
fn is_owned_temp_file_name(name: &[u8]) -> bool {
    let Some(suffix) = name.strip_prefix(TEMP_FILE_PREFIX.as_bytes()) else {
        return false;
    };
    suffix.len() == TEMP_FILE_HEX_LENGTH
        && suffix
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

/// Proof that one exact-name unfinished regular file is safe to unlink.
///
/// [BLIND-VAULT-RECOVERY-TEMP-OWNERSHIP 2026-08-31 by Codex] The open
/// descriptor pins the inode that passed regular-file, owner, single-link, and
/// exact private-mode validation. Cleanup keeps this guard alive until after
/// `unlinkat`, so an unvalidated hardlink can never reach the unlink effect.
struct ValidatedOwnedTempRegularFile {
    _file: File,
}

impl ValidatedOwnedTempRegularFile {
    fn open_at(directory: &File, name: &str) -> Result<Self, PrivateRecoveryIoError> {
        let file = open_private_regular_file_at(directory, name, false, false, false)?;
        let metadata = file.metadata()?;
        validate_exact_private_regular_metadata(&metadata)?;
        Ok(Self { _file: file })
    }
}

fn cleanup_owned_temp_files(directory: &File) -> Result<(), PrivateRecoveryIoError> {
    // [BLIND-VAULT-RECOVERY-TEMP-CLEANUP 2026-08-29 by Codex] Cleanup runs
    // only while holding the exclusive fence and only for our exact grammar.
    // Unknown files and directories are never touched.
    let duplicated = dup(directory.as_raw_fd()).map_err(nix_filesystem_error)?;
    let mut entries = Dir::from_fd(duplicated).map_err(nix_filesystem_error)?;
    for entry in entries.iter() {
        let entry = entry.map_err(nix_filesystem_error)?;
        if !is_owned_temp_file_name(entry.file_name().to_bytes()) {
            continue;
        }
        let entry_type = match entry.file_type() {
            Some(entry_type) => Some(entry_type),
            None => {
                let metadata = fstatat(
                    Some(directory.as_raw_fd()),
                    entry.file_name(),
                    AtFlags::AT_SYMLINK_NOFOLLOW,
                )
                .map_err(nix_filesystem_error)?;
                let kind = SFlag::from_bits_truncate(metadata.st_mode);
                if kind == SFlag::S_IFREG {
                    Some(DirectoryEntryType::File)
                } else if kind == SFlag::S_IFLNK {
                    Some(DirectoryEntryType::Symlink)
                } else {
                    None
                }
            }
        };
        match entry_type {
            Some(DirectoryEntryType::File) => {
                let name = std::str::from_utf8(entry.file_name().to_bytes())
                    .map_err(|_| PrivateRecoveryIoError::UnsafePath)?;
                let _validated = ValidatedOwnedTempRegularFile::open_at(directory, name)?;
                unlinkat(
                    Some(directory.as_raw_fd()),
                    entry.file_name(),
                    UnlinkatFlags::NoRemoveDir,
                )
                .map_err(nix_filesystem_error)?;
            }
            Some(DirectoryEntryType::Symlink) => {
                unlinkat(
                    Some(directory.as_raw_fd()),
                    entry.file_name(),
                    UnlinkatFlags::NoRemoveDir,
                )
                .map_err(nix_filesystem_error)?;
            }
            None => {
                continue;
            }
            Some(_) => continue,
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

#[cfg(test)]
mod tests {
    use std::{cell::Cell, path::PathBuf};

    use super::*;

    struct RecordingParentDirectoryDurability {
        calls: Cell<usize>,
        fail: bool,
    }

    impl RecordingParentDirectoryDurability {
        fn succeeding() -> Self {
            Self {
                calls: Cell::new(0),
                fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                calls: Cell::new(0),
                fail: true,
            }
        }
    }

    impl ParentDirectoryDurability for RecordingParentDirectoryDurability {
        fn sync_parent(&self, _parent: &File) -> std::io::Result<()> {
            self.calls.set(self.calls.get() + 1);
            if self.fail {
                Err(std::io::Error::new(
                    ErrorKind::Other,
                    "injected parent directory sync failure",
                ))
            } else {
                Ok(())
            }
        }
    }

    struct RecreateTempThenFailPublicationDurability {
        external_alias: PathBuf,
        recreated_temp: PathBuf,
        calls: Cell<usize>,
    }

    impl PublicationDirectoryDurability for RecreateTempThenFailPublicationDurability {
        fn sync_after_rename(&self, _directory: &File) -> std::io::Result<()> {
            self.calls.set(self.calls.get() + 1);
            fs::hard_link(&self.external_alias, &self.recreated_temp)?;
            Err(std::io::Error::new(
                ErrorKind::Other,
                "injected publication directory sync failure",
            ))
        }
    }

    fn entry_exists(path: &Path) -> bool {
        match fs::symlink_metadata(path) {
            Ok(_) => true,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(error) => panic!("inspect recovery fixture entry: {error}"),
        }
    }

    #[test]
    fn existing_component_retries_parent_sync_after_ambiguous_creation_failure() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let canonical_root =
            fs::canonicalize(root.path()).expect("canonical temporary recovery root");
        let parent = OpenOptions::new()
            .read(true)
            .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW | nix::libc::O_DIRECTORY)
            .open(canonical_root)
            .expect("open canonical temporary root");

        let first_attempt = RecordingParentDirectoryDurability::failing();
        assert!(matches!(
            open_or_create_directory_at_with(
                &parent,
                OsStr::new("recovery"),
                &first_attempt,
            ),
            Err(PrivateRecoveryIoError::Filesystem(error))
                if error.kind() == ErrorKind::Other
        ));
        assert_eq!(first_attempt.calls.get(), 1);

        let retry = RecordingParentDirectoryDurability::succeeding();
        let directory = open_or_create_directory_at_with(&parent, OsStr::new("recovery"), &retry)
            .expect("retry existing recovery directory");
        assert_eq!(retry.calls.get(), 1);
        assert!(directory.metadata().expect("recovery metadata").is_dir());
    }

    #[test]
    fn owned_temp_name_requires_exact_lowercase_hex_grammar() {
        let exact = format!("{TEMP_FILE_PREFIX}{}", "0a".repeat(TEMP_FILE_RANDOM_BYTES));
        let short = format!("{TEMP_FILE_PREFIX}{}", "1".repeat(TEMP_FILE_HEX_LENGTH - 1));
        let long = format!("{TEMP_FILE_PREFIX}{}", "2".repeat(TEMP_FILE_HEX_LENGTH + 1));
        let non_hex = format!(
            "{TEMP_FILE_PREFIX}{}g",
            "3".repeat(TEMP_FILE_HEX_LENGTH - 1)
        );
        let uppercase = format!("{TEMP_FILE_PREFIX}{}", "A".repeat(TEMP_FILE_HEX_LENGTH));

        assert!(is_owned_temp_file_name(exact.as_bytes()));
        for unknown in [short, long, non_hex, uppercase] {
            assert!(!is_owned_temp_file_name(unknown.as_bytes()));
        }
    }

    #[test]
    fn cleanup_removes_only_exact_regular_and_symlink_temp_entries() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let directory = fs::canonicalize(root.path())
            .expect("canonical temporary recovery root")
            .join("recovery");
        let store = PrivateAtomicRecoveryFile::open(&directory).expect("open recovery fixture");

        let exact_regular = format!("{TEMP_FILE_PREFIX}{}", "0".repeat(TEMP_FILE_HEX_LENGTH));
        let exact_symlink = format!("{TEMP_FILE_PREFIX}{}", "1".repeat(TEMP_FILE_HEX_LENGTH));
        let exact_directory = format!("{TEMP_FILE_PREFIX}{}", "2".repeat(TEMP_FILE_HEX_LENGTH));
        let short = format!("{TEMP_FILE_PREFIX}{}", "3".repeat(TEMP_FILE_HEX_LENGTH - 1));
        let long = format!("{TEMP_FILE_PREFIX}{}", "4".repeat(TEMP_FILE_HEX_LENGTH + 1));
        let non_hex = format!(
            "{TEMP_FILE_PREFIX}{}g",
            "5".repeat(TEMP_FILE_HEX_LENGTH - 1)
        );
        let uppercase = format!("{TEMP_FILE_PREFIX}{}", "A".repeat(TEMP_FILE_HEX_LENGTH));
        let unrelated = ".recovery-state-v1.tmp-unknown";

        for name in [&exact_regular, &short, &long, &non_hex, &uppercase] {
            fs::write(directory.join(name), [0xa5; 8]).expect("write opaque recovery fixture");
        }
        // [BLIND-VAULT-RECOVERY-TEMP-OWNERSHIP 2026-08-31 by Codex] Model the
        // exact mode produced by the real writable temp-file creation path.
        fs::set_permissions(
            directory.join(&exact_regular),
            fs::Permissions::from_mode(PRIVATE_FILE_MODE),
        )
        .expect("set exact recovery fixture mode");
        std::os::unix::fs::symlink("opaque-target", directory.join(&exact_symlink))
            .expect("create recovery fixture symlink");
        fs::create_dir(directory.join(&exact_directory))
            .expect("create recovery fixture directory");
        fs::write(directory.join(unrelated), [0x5a; 8]).expect("write unrelated recovery fixture");

        cleanup_owned_temp_files(&store.directory).expect("cleanup owned recovery fixtures");

        assert!(!entry_exists(&directory.join(exact_regular)));
        assert!(!entry_exists(&directory.join(exact_symlink)));
        for name in [short, long, non_hex, uppercase] {
            assert!(entry_exists(&directory.join(name)));
        }
        assert!(entry_exists(&directory.join(exact_directory)));
        assert!(entry_exists(&directory.join(unrelated)));
    }

    #[test]
    // [BLIND-VAULT-RECOVERY-TEMP-OWNERSHIP 2026-08-31 by Codex] Invalid
    // ownership must fail before either directory entry or link count changes.
    fn cleanup_rejects_hardlinked_exact_temp_without_unlinking_external_alias() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let canonical_root =
            fs::canonicalize(root.path()).expect("canonical temporary recovery root");
        let directory = canonical_root.join("recovery");
        let store = PrivateAtomicRecoveryFile::open(&directory).expect("open recovery fixture");

        let external_alias = canonical_root.join("opaque-alias.bin");
        fs::write(&external_alias, [0x3c; 8]).expect("write opaque alias fixture");
        fs::set_permissions(
            &external_alias,
            fs::Permissions::from_mode(PRIVATE_FILE_MODE),
        )
        .expect("set opaque alias fixture mode");
        let exact_temp = format!("{TEMP_FILE_PREFIX}{}", "6".repeat(TEMP_FILE_HEX_LENGTH));
        let recovery_alias = directory.join(exact_temp);
        fs::hard_link(&external_alias, &recovery_alias)
            .expect("hard-link exact recovery temp fixture");
        assert_eq!(
            fs::metadata(&external_alias)
                .expect("external alias metadata before cleanup")
                .nlink(),
            2
        );

        assert!(matches!(
            cleanup_owned_temp_files(&store.directory),
            Err(PrivateRecoveryIoError::UnsafePath)
        ));
        assert!(entry_exists(&recovery_alias));
        assert_eq!(
            fs::metadata(&external_alias)
                .expect("external alias metadata after cleanup")
                .nlink(),
            2
        );
    }

    #[test]
    // [BLIND-VAULT-RECOVERY-PUBLICATION-CLEANUP 2026-08-31 by Codex] An
    // exclusive-create collision never grants cleanup ownership of its name.
    fn failed_temp_create_does_not_unlink_preexisting_hardlink() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let canonical_root =
            fs::canonicalize(root.path()).expect("canonical temporary recovery root");
        let directory = canonical_root.join("recovery");
        let store = PrivateAtomicRecoveryFile::open(&directory).expect("open recovery fixture");

        let external_alias = canonical_root.join("opaque-publish-alias.bin");
        fs::write(&external_alias, [0x87; 16]).expect("write opaque publish alias fixture");
        fs::set_permissions(
            &external_alias,
            fs::Permissions::from_mode(PRIVATE_FILE_MODE),
        )
        .expect("set opaque publish alias fixture mode");
        let temp_name = format!("{TEMP_FILE_PREFIX}{}", "7".repeat(TEMP_FILE_HEX_LENGTH));
        let recovery_alias = directory.join(&temp_name);
        fs::hard_link(&external_alias, &recovery_alias)
            .expect("hard-link deterministic publish collision");

        assert!(matches!(
            store.replace_with_temp_name_and_durability(
                &[0xc3; 16],
                64,
                &temp_name,
                &HostPublicationDirectoryDurability,
            ),
            Err(PrivateRecoveryIoError::Filesystem(error))
                if error.kind() == ErrorKind::AlreadyExists
        ));
        assert!(entry_exists(&recovery_alias));
        assert_eq!(
            fs::metadata(&external_alias)
                .expect("external publish alias metadata after collision")
                .nlink(),
            2
        );
        assert!(!entry_exists(&directory.join(STATE_FILE_NAME)));
    }

    #[test]
    // [BLIND-VAULT-RECOVERY-PUBLICATION-CLEANUP 2026-08-31 by Codex] Once
    // rename commits, its former temp name is never cleanup-owned again.
    fn post_rename_sync_failure_does_not_unlink_recreated_temp_name() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let canonical_root =
            fs::canonicalize(root.path()).expect("canonical temporary recovery root");
        let directory = canonical_root.join("recovery");
        let store = PrivateAtomicRecoveryFile::open(&directory).expect("open recovery fixture");

        let external_alias = canonical_root.join("opaque-recreated-alias.bin");
        fs::write(&external_alias, [0x96; 16]).expect("write opaque recreated alias fixture");
        fs::set_permissions(
            &external_alias,
            fs::Permissions::from_mode(PRIVATE_FILE_MODE),
        )
        .expect("set opaque recreated alias fixture mode");
        let temp_name = format!("{TEMP_FILE_PREFIX}{}", "8".repeat(TEMP_FILE_HEX_LENGTH));
        let recreated_temp = directory.join(&temp_name);
        let durability = RecreateTempThenFailPublicationDurability {
            external_alias: external_alias.clone(),
            recreated_temp: recreated_temp.clone(),
            calls: Cell::new(0),
        };
        let sealed_generation = [0xd4; 16];

        assert!(matches!(
            store.replace_with_temp_name_and_durability(
                &sealed_generation,
                64,
                &temp_name,
                &durability,
            ),
            Err(PrivateRecoveryIoError::Filesystem(error))
                if error.kind() == ErrorKind::Other
        ));
        assert_eq!(durability.calls.get(), 1);
        assert!(entry_exists(&recreated_temp));
        assert_eq!(
            fs::metadata(&external_alias)
                .expect("external recreated alias metadata after sync failure")
                .nlink(),
            2
        );
        assert_eq!(
            store.read(64).expect("read renamed opaque generation"),
            Some(sealed_generation.to_vec())
        );
    }

    #[test]
    // [BLIND-VAULT-RECOVERY-PUBLICATION-CLEANUP 2026-08-31 by Codex] The
    // opened phase cleans only its pinned inode and retains unknown mismatches.
    fn opened_phase_cleanup_requires_matching_pinned_inode() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let canonical_root =
            fs::canonicalize(root.path()).expect("canonical temporary recovery root");
        let directory = canonical_root.join("recovery");
        let store = PrivateAtomicRecoveryFile::open(&directory).expect("open recovery fixture");

        let owned_name = format!("{TEMP_FILE_PREFIX}{}", "9".repeat(TEMP_FILE_HEX_LENGTH));
        let owned_file =
            open_private_regular_file_at(&store.directory, &owned_name, true, true, true)
                .expect("create pinned owned temp fixture");
        let owned_phase = PublicationPhase::TempOpened(
            PinnedPublicationTemp::new(owned_file).expect("pin owned temp fixture"),
        );
        owned_phase
            .cleanup_failed_publish(&store.directory, &owned_name)
            .expect("cleanup matching pinned temp fixture");
        assert!(!entry_exists(&directory.join(&owned_name)));

        let replaced_name = format!("{TEMP_FILE_PREFIX}{}", "a".repeat(TEMP_FILE_HEX_LENGTH));
        let replaced_path = directory.join(&replaced_name);
        let replaced_file =
            open_private_regular_file_at(&store.directory, &replaced_name, true, true, true)
                .expect("create pinned replaced temp fixture");
        let replaced_phase = PublicationPhase::TempOpened(
            PinnedPublicationTemp::new(replaced_file).expect("pin replaced temp fixture"),
        );
        fs::remove_file(&replaced_path).expect("remove pinned temp directory entry");

        let unknown = canonical_root.join("opaque-replacement.bin");
        fs::write(&unknown, [0xb7; 16]).expect("write opaque replacement fixture");
        fs::set_permissions(&unknown, fs::Permissions::from_mode(PRIVATE_FILE_MODE))
            .expect("set opaque replacement fixture mode");
        let unknown_inode = fs::metadata(&unknown)
            .expect("opaque replacement metadata")
            .ino();
        fs::rename(&unknown, &replaced_path).expect("replace pinned temp directory entry");

        assert!(matches!(
            replaced_phase.cleanup_failed_publish(&store.directory, &replaced_name),
            Err(PrivateRecoveryIoError::UnsafePath)
        ));
        assert!(entry_exists(&replaced_path));
        assert_eq!(
            fs::metadata(&replaced_path)
                .expect("retained replacement metadata")
                .ino(),
            unknown_inode
        );
    }

    #[test]
    fn hardlinked_lock_is_rejected_without_changing_external_alias_mode() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let canonical_root =
            fs::canonicalize(root.path()).expect("canonical temporary recovery root");
        let directory = canonical_root.join("recovery");
        fs::create_dir(&directory).expect("create recovery fixture directory");

        let external_alias = canonical_root.join("opaque-alias.bin");
        fs::write(&external_alias, [0x6d; 8]).expect("write opaque alias fixture");
        fs::set_permissions(&external_alias, fs::Permissions::from_mode(0o644))
            .expect("set alias fixture mode");
        fs::hard_link(&external_alias, directory.join(LOCK_FILE_NAME))
            .expect("hard-link recovery lock fixture");

        assert!(matches!(
            PrivateAtomicRecoveryFile::open(&directory),
            Err(PrivateRecoveryIoError::UnsafePath)
        ));
        assert_eq!(
            fs::metadata(&external_alias)
                .expect("external alias metadata")
                .permissions()
                .mode()
                & 0o777,
            0o644
        );
    }

    #[test]
    fn legitimate_single_link_lock_mode_is_normalized() {
        let root = tempfile::tempdir().expect("temporary recovery root");
        let directory = fs::canonicalize(root.path())
            .expect("canonical temporary recovery root")
            .join("recovery");
        fs::create_dir(&directory).expect("create recovery fixture directory");
        let lock = directory.join(LOCK_FILE_NAME);
        fs::write(&lock, [0x7e; 8]).expect("write opaque lock fixture");
        fs::set_permissions(&lock, fs::Permissions::from_mode(0o644))
            .expect("set lock fixture mode");

        let store =
            PrivateAtomicRecoveryFile::open(&directory).expect("open legitimate recovery fixture");
        assert_eq!(
            fs::metadata(&lock)
                .expect("lock fixture metadata")
                .permissions()
                .mode()
                & 0o777,
            PRIVATE_FILE_MODE
        );
        drop(store);
    }
}
