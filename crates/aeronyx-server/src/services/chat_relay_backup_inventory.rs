// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_inventory.rs
// ============================================
// Version: 1.0.0-VerifiedBackupInventoryDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-INVENTORY-DOMAIN 2026-08-27 by Codex] Extract the
//   fail-closed private recovery inventory from the oversized relay service.
//
// Main Functionality:
//   - Inspects owner-private complete and interrupted backup artifacts.
//   - Composes namespace classification, SQLite verification, and retention.
//   - Revalidates exact storage identity before destructive maintenance.
//   - Inspects the active restore boundary without opening live SQLite state.
//
// Dependencies:
//   - `chat_relay_backup_artifact` models immutable storage observations.
//   - `chat_relay_backup_namespace` classifies the managed private namespace.
//   - `chat_relay_backup_retention` supplies side-effect-free policy planning.
//   - A caller-provided verifier performs full SQLite artifact verification.
//
// Main Logical Flow:
//   1. Bound and inspect every private backup-directory entry.
//   2. Reject unmanaged, unsafe, empty, or changing storage objects.
//   3. Fully verify every complete recovery image through the composed trait.
//   4. Produce a path-private inventory and aggregate public receipt.
//   5. Reinspect planned candidates immediately before caller-owned deletion.
//
// Important Note for Next Developer:
//   - Unknown entries and metadata races must continue to fail closed.
//   - Never expose paths, filenames, timestamps, device IDs, or inodes.
//   - The verifier must perform full SQLite validation, not a size-only check.
//   - Deletion, audit publication, and cross-process locking remain outside.
//
// Last Modified:
//   v1.0.0-VerifiedBackupInventoryDomain - Initial composed inventory extraction
// ============================================

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::services::chat_relay_backup_artifact::{
    BackupArtifactAccountingError, BackupArtifactIdentityState, BackupArtifactSnapshot,
    BackupStorageIdentity,
};
use crate::services::chat_relay_backup_contract::ChatRelayBackupRetentionReceipt;
use crate::services::chat_relay_backup_io::backup_io_error;
use crate::services::chat_relay_backup_namespace::{BackupArtifactKind, BackupArtifactNamespace};
use crate::services::chat_relay_backup_retention::{
    BackupRetentionLimits, BackupRetentionPlanner, BackupRetentionPolicyError,
};
use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Complete path-private result of one authenticated backup inventory scan.
#[derive(Debug)]
pub(super) struct ChatRelayBackupRetentionInspection {
    pub(super) receipt: ChatRelayBackupRetentionReceipt,
    pub(super) newest_backup: Option<BackupArtifactSnapshot>,
    pub(super) excess_backups: Vec<BackupArtifactSnapshot>,
    pub(super) stale_partials: Vec<BackupArtifactSnapshot>,
}

/// Private active-custody metadata included in restore-plan commitments.
#[derive(Debug, Default)]
pub(super) struct ChatRelayActiveRestoreBoundary {
    pub(super) present: bool,
    pub(super) size_bytes: u64,
    pub(super) sidecars_present: bool,
    pub(super) modified_at: Option<SystemTime>,
    pub(super) device_id: u64,
    pub(super) inode: u64,
}

/// Immutable resource and retention limits for one inventory scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BackupInventoryLimits {
    max_directory_entries: usize,
    retention: BackupRetentionLimits,
}

impl BackupInventoryLimits {
    pub(super) const fn new(
        max_directory_entries: usize,
        retention: BackupRetentionLimits,
    ) -> Self {
        Self {
            max_directory_entries,
            retention,
        }
    }
}

/// Replaceable full-verification boundary for one recovery image.
pub(super) trait BackupArtifactVerifier {
    fn verify_existing(&self, path: &Path) -> ChatRelayResult<u64>;
}

impl<F> BackupArtifactVerifier for F
where
    F: Fn(&Path) -> ChatRelayResult<u64>,
{
    fn verify_existing(&self, path: &Path) -> ChatRelayResult<u64> {
        self(path)
    }
}

/// Capability boundary for trusted backup inventory and candidate rechecks.
pub(super) trait BackupInventory {
    fn inspect(
        &self,
        backup_directory: &Path,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayBackupRetentionInspection>;

    fn reverify_candidate(
        &self,
        artifact: &BackupArtifactSnapshot,
        verify_sqlite: bool,
    ) -> ChatRelayResult<()>;
}

/// Composed production inventory over replaceable pure policies and verifier.
#[derive(Debug)]
pub(super) struct VerifiedBackupInventory<Namespace, Planner, Verifier> {
    namespace: Namespace,
    planner: Planner,
    verifier: Verifier,
}

impl<Namespace, Planner, Verifier> VerifiedBackupInventory<Namespace, Planner, Verifier> {
    pub(super) const fn new(namespace: Namespace, planner: Planner, verifier: Verifier) -> Self {
        Self {
            namespace,
            planner,
            verifier,
        }
    }

    fn inspect_private_entry(
        path: PathBuf,
        file_name: String,
    ) -> ChatRelayResult<BackupArtifactSnapshot> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        let metadata = std::fs::symlink_metadata(&path).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect relay backup retention entry",
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_PERM,
                "relay backup retention entry is not a private regular file",
            ));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "relay backup retention entry is not owner-private",
                ));
            }
        }

        let modified_at = metadata.modified().map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to inspect relay backup retention age",
            )
        })?;
        #[cfg(unix)]
        let storage_identity = BackupStorageIdentity::Unix {
            device_id: metadata.dev(),
            inode: metadata.ino(),
        };
        #[cfg(not(unix))]
        let storage_identity = BackupStorageIdentity::Portable;

        Ok(BackupArtifactSnapshot::new(
            path,
            file_name,
            metadata.len(),
            modified_at,
            storage_identity,
        ))
    }
}

impl<Namespace, Planner, Verifier> BackupInventory
    for VerifiedBackupInventory<Namespace, Planner, Verifier>
where
    Namespace: BackupArtifactNamespace,
    Planner: BackupRetentionPlanner<BackupArtifactSnapshot>,
    Verifier: BackupArtifactVerifier,
{
    fn inspect(
        &self,
        backup_directory: &Path,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayBackupRetentionInspection> {
        // [CHAT-RELAY-BACKUP-INVENTORY-DOMAIN 2026-08-27 by Codex] The scan
        // admits only the managed grammar, then authenticates complete images
        // and proves their storage identity remained stable across the read.
        let mut artifacts = Vec::new();
        let mut partials = Vec::new();
        for (index, entry) in std::fs::read_dir(backup_directory)
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to read private relay backup directory",
                )
            })?
            .enumerate()
        {
            if index >= limits.max_directory_entries {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup directory exceeds maintenance scan limit",
                ));
            }
            let entry = entry.map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect relay backup directory entry",
                )
            })?;
            let file_name = entry.file_name().into_string().map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_MISMATCH,
                    "relay backup directory contains an unsupported entry name",
                )
            })?;
            let inspected = Self::inspect_private_entry(entry.path(), file_name.clone())?;
            match self.namespace.classify(&file_name) {
                BackupArtifactKind::RecoveryImage => {
                    if inspected.size_bytes() == 0 {
                        return Err(backup_io_error(
                            rusqlite::ffi::SQLITE_CORRUPT,
                            "relay backup retention artifact is empty",
                        ));
                    }
                    artifacts.push(inspected);
                }
                BackupArtifactKind::InterruptedTemporary => partials.push(inspected),
                BackupArtifactKind::Unmanaged => {
                    return Err(backup_io_error(
                        rusqlite::ffi::SQLITE_MISMATCH,
                        "relay backup directory contains an unmanaged entry",
                    ));
                }
            }
        }

        for artifact in &artifacts {
            let verified_size = self.verifier.verify_existing(artifact.path())?;
            let rechecked =
                Self::inspect_private_entry(artifact.path_buf(), artifact.file_name_owned())?;
            if !artifact.verified_size_matches(verified_size)
                || artifact.identity_state(&rechecked) != BackupArtifactIdentityState::Stable
            {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup retention artifact changed during verification",
                ));
            }
        }

        let retention = self
            .planner
            .plan(artifacts, partials, now_unix_secs, limits.retention)
            .map_err(map_retention_policy_error)?;

        Ok(ChatRelayBackupRetentionInspection {
            receipt: ChatRelayBackupRetentionReceipt {
                retained_count: retention.retained_count,
                retained_bytes: retention.retained_bytes,
                excess_count: retention.excess_count,
                excess_bytes: retention.excess_bytes,
                partial_count: retention.partial_count,
                partial_bytes: retention.partial_bytes,
                budget_exceeded: retention.budget_exceeded,
            },
            newest_backup: retention.newest_backup,
            excess_backups: retention.excess_oldest_first,
            stale_partials: retention.stale_partials_oldest_first,
        })
    }

    fn reverify_candidate(
        &self,
        artifact: &BackupArtifactSnapshot,
        verify_sqlite: bool,
    ) -> ChatRelayResult<()> {
        let before = Self::inspect_private_entry(artifact.path_buf(), artifact.file_name_owned())?;
        if artifact.identity_state(&before) != BackupArtifactIdentityState::Stable {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup prune candidate changed after planning",
            ));
        }
        if verify_sqlite {
            let verified_size = self.verifier.verify_existing(artifact.path())?;
            let after =
                Self::inspect_private_entry(artifact.path_buf(), artifact.file_name_owned())?;
            if !artifact.verified_size_matches(verified_size)
                || artifact.identity_state(&after) != BackupArtifactIdentityState::Stable
            {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup prune candidate changed during verification",
                ));
            }
        }
        Ok(())
    }
}

/// Returns checked aggregate bytes for one private maintenance plan.
pub(super) fn checked_backup_artifact_bytes(
    artifacts: &[BackupArtifactSnapshot],
    reason: &'static str,
) -> ChatRelayResult<u64> {
    BackupArtifactSnapshot::checked_total_bytes(artifacts)
        .map_err(|error| map_artifact_accounting_error(error, reason))
}

/// Returns the complete verified-image count used by restore contracts.
pub(super) fn verified_restore_backup_count(
    inspection: &ChatRelayBackupRetentionInspection,
) -> ChatRelayResult<usize> {
    inspection
        .receipt
        .retained_count
        .checked_add(inspection.receipt.excess_count)
        .ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay restore-plan backup count overflow",
            )
        })
}

/// Inspects live custody metadata without opening or mutating SQLite state.
pub(super) fn inspect_active_restore_boundary(
    database_path: &str,
) -> ChatRelayResult<ChatRelayActiveRestoreBoundary> {
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    // [CHAT-RELAY-BACKUP-INVENTORY-DOMAIN 2026-08-27 by Codex] Opening an
    // active WAL database read-only can create SHM state, so this boundary is
    // deliberately metadata-only and rejects unsafe storage objects.
    let active_path = Path::new(database_path);
    let mut boundary = ChatRelayActiveRestoreBoundary::default();
    match std::fs::symlink_metadata(active_path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "active relay custody boundary is not a regular file",
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                if metadata.permissions().mode() & 0o077 != 0 {
                    return Err(backup_io_error(
                        rusqlite::ffi::SQLITE_PERM,
                        "active relay custody file is not owner-private",
                    ));
                }
                boundary.device_id = metadata.dev();
                boundary.inode = metadata.ino();
            }
            boundary.present = true;
            boundary.size_bytes = metadata.len();
            boundary.modified_at = Some(metadata.modified().map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect active relay custody age",
                )
            })?);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect active relay custody boundary",
            ));
        }
    }

    for suffix in ["-journal", "-wal", "-shm"] {
        let mut sidecar = active_path.as_os_str().to_os_string();
        sidecar.push(suffix);
        match std::fs::symlink_metadata(PathBuf::from(sidecar)) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    return Err(backup_io_error(
                        rusqlite::ffi::SQLITE_PERM,
                        "active relay custody sidecar boundary is unsafe",
                    ));
                }
                boundary.sidecars_present = true;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect active relay custody sidecar boundary",
                ));
            }
        }
    }

    Ok(boundary)
}

fn map_retention_policy_error(error: BackupRetentionPolicyError) -> ChatRelayError {
    match error {
        BackupRetentionPolicyError::RetainedBytesOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup retained-byte accounting overflow",
        ),
        BackupRetentionPolicyError::ExcessBytesOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup excess-byte accounting overflow",
        ),
        BackupRetentionPolicyError::PartialBytesOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup partial-byte accounting overflow",
        ),
        BackupRetentionPolicyError::PartialCutoffOutOfRange => backup_io_error(
            rusqlite::ffi::SQLITE_RANGE,
            "relay backup partial grace cutoff is out of range",
        ),
    }
}

fn map_artifact_accounting_error(
    error: BackupArtifactAccountingError,
    reason: &'static str,
) -> ChatRelayError {
    match error {
        BackupArtifactAccountingError::BytesOverflow => {
            backup_io_error(rusqlite::ffi::SQLITE_FULL, reason)
        }
    }
}
