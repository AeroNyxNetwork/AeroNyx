// ============================================================================
// File: crates/aeronyx-server/src/services/chat_relay_runtime_fence.rs
// ============================================================================
// Creation Reason:
//   Isolate persistent chat-relay database ownership from the large custody
//   service so restart recovery has one auditable process-lifecycle boundary.
//
// Main Functionality:
//   - Derive an owner-private control path beside the custody database.
//   - Reject symlinks and non-regular control files.
//   - Acquire one non-blocking exclusive OS lock for the service lifetime.
//   - Release ownership automatically through RAII after the service drops.
//
// Dependencies:
//   - Composed by `services/chat_relay.rs` before SQLite is opened.
//   - Uses the workspace-pinned `nix` filesystem API on Unix nodes.
//
// Main Logical Flow:
//   1. Validate the database path and existing control-file type.
//   2. Open the control file with no-follow and close-on-exec protections.
//   3. Acquire an exclusive non-blocking kernel lock and enforce mode 0600.
//   4. Retain the lock handle until `ChatRelayRuntimeFence` is dropped.
//
// Important Note for Next Developer:
//   - [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] Never serialize, log, or
//     report the control path, process id, host identity, raw errno, or handle.
//   - Never replace the kernel-owned lock with a wall-clock lease. Verified
//     submit and blind-route recovery require proof that the predecessor exited.
//   - Keep acquisition non-blocking; a second node process must fail closed.
//
// Last Modified:
//   v1.0.0-ChatRelayRuntimeFence - Initial OS-owned custody database fence
// ============================================================================

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use nix::errno::Errno;
use nix::fcntl::{Flock, FlockArg};

const CHAT_RELAY_RUNTIME_FENCE_SUFFIX: &str = ".chat-relay-runtime-v1.lock";

/// Closed, path-free failure vocabulary for persistent relay ownership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChatRelayRuntimeFenceError {
    InvalidDatabasePath,
    UnsafeControlFile,
    ControlFileInspectionFailed,
    ControlFileOpenFailed,
    AlreadyOwned,
    LockFailed,
    ControlFilePermissionsFailed,
}

impl ChatRelayRuntimeFenceError {
    /// Stable operator-safe reason without path, identity, or OS error details.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidDatabasePath => "invalid_database_path",
            Self::UnsafeControlFile => "unsafe_control_file",
            Self::ControlFileInspectionFailed => "control_file_inspection_failed",
            Self::ControlFileOpenFailed => "control_file_open_failed",
            Self::AlreadyOwned => "already_owned",
            Self::LockFailed => "lock_failed",
            Self::ControlFilePermissionsFailed => "control_file_permissions_failed",
        }
    }

    /// Preserves the established storage-path bucket while distinguishing
    /// ownership and unsafe-control failures that require operator action.
    pub(crate) const fn public_reason_bucket(self) -> &'static str {
        match self {
            Self::InvalidDatabasePath
            | Self::ControlFileInspectionFailed
            | Self::ControlFileOpenFailed
            | Self::ControlFilePermissionsFailed => "sqlite_error",
            Self::UnsafeControlFile | Self::AlreadyOwned | Self::LockFailed => {
                "runtime_fence_unavailable"
            }
        }
    }
}

/// RAII ownership of one persistent chat-relay custody database.
pub(crate) struct ChatRelayRuntimeFence {
    /// The handle itself is the capability; dropping it releases the OS lock.
    _handle: Flock<File>,
}

impl ChatRelayRuntimeFence {
    /// Derives the private control path without exposing it outside the service.
    pub(crate) fn control_path(
        database_path: &Path,
    ) -> Result<PathBuf, ChatRelayRuntimeFenceError> {
        let file_name = database_path
            .file_name()
            .ok_or(ChatRelayRuntimeFenceError::InvalidDatabasePath)?;
        let mut lock_name = file_name.to_os_string();
        lock_name.push(CHAT_RELAY_RUNTIME_FENCE_SUFFIX);
        Ok(database_path.with_file_name(lock_name))
    }

    /// Acquires exclusive non-blocking ownership before SQLite can be opened.
    pub(crate) fn acquire(database_path: &Path) -> Result<Self, ChatRelayRuntimeFenceError> {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

        let path = Self::control_path(database_path)?;
        match std::fs::symlink_metadata(&path) {
            Ok(metadata)
                if metadata.file_type().is_symlink() || !metadata.file_type().is_file() =>
            {
                return Err(ChatRelayRuntimeFenceError::UnsafeControlFile);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err(ChatRelayRuntimeFenceError::ControlFileInspectionFailed),
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW)
            .open(&path)
            .map_err(|error| {
                if error.raw_os_error() == Some(nix::libc::ELOOP) {
                    ChatRelayRuntimeFenceError::UnsafeControlFile
                } else {
                    ChatRelayRuntimeFenceError::ControlFileOpenFailed
                }
            })?;
        let metadata = file
            .metadata()
            .map_err(|_| ChatRelayRuntimeFenceError::ControlFileInspectionFailed)?;
        // [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] Inspect the opened
        // inode, not only the pre-open path. A hard-linked control inode could
        // otherwise let permission tightening mutate an unrelated file.
        if !metadata.file_type().is_file() || metadata.nlink() != 1 {
            return Err(ChatRelayRuntimeFenceError::UnsafeControlFile);
        }

        let handle = Flock::lock(file, FlockArg::LockExclusiveNonblock).map_err(|(_, error)| {
            if error == Errno::EWOULDBLOCK {
                ChatRelayRuntimeFenceError::AlreadyOwned
            } else {
                ChatRelayRuntimeFenceError::LockFailed
            }
        })?;
        handle
            .set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|_| ChatRelayRuntimeFenceError::ControlFilePermissionsFailed)?;
        Ok(Self { _handle: handle })
    }
}
