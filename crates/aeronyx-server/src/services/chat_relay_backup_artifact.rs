// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_artifact.rs
// ============================================
// Version: 1.0.0-BackupArtifactDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-ARTIFACT-DOMAIN 2026-08-27 by Codex] Isolate the
//   immutable private recovery-artifact snapshot and exact identity checks
//   from filesystem orchestration.
//
// Main Functionality:
//   - Models one inspected private backup artifact as an immutable snapshot.
//   - Models platform storage-object identity without leaking OS conditionals.
//   - Classifies every identity mismatch through a closed enum.
//   - Performs checked aggregate byte accounting for maintenance plans.
//
// Dependencies:
//   - `chat_relay_backup_retention.rs` supplies the path-blind planner trait.
//   - `chat_relay.rs` owns metadata collection, SQLite verification, and I/O.
//
// Main Logical Flow:
//   1. The service validates a private regular file and captures its metadata.
//   2. This module stores the path, stable name, size, time, and object ID.
//   3. Reinspection produces another snapshot for exact identity comparison.
//   4. Any mismatch or byte overflow is returned as a closed fail-closed state.
//
// Important Note for Next Developer:
//   - This module must remain side-effect free and must never inspect a path.
//   - A stable identity requires every captured field to match exactly.
//   - The service must re-open SQLite and re-capture metadata before deletion.
//   - Do not expose private paths, names, times, device IDs, or inodes publicly.
//
// Last Modified:
//   v1.0.0-BackupArtifactDomain - Initial immutable artifact domain model
// ============================================

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::services::chat_relay_backup_retention::BackupRetentionArtifact;

/// Platform-specific identity for one already-inspected storage object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupStorageIdentity {
    #[cfg(unix)]
    Unix { device_id: u64, inode: u64 },
    #[cfg(not(unix))]
    Portable,
}

impl BackupStorageIdentity {
    pub(crate) const fn device_id(self) -> u64 {
        match self {
            #[cfg(unix)]
            Self::Unix { device_id, .. } => device_id,
            #[cfg(not(unix))]
            Self::Portable => 0,
        }
    }

    pub(crate) const fn inode(self) -> u64 {
        match self {
            #[cfg(unix)]
            Self::Unix { inode, .. } => inode,
            #[cfg(not(unix))]
            Self::Portable => 0,
        }
    }
}

/// Closed result of comparing two private artifact snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupArtifactIdentityState {
    Stable,
    PathChanged,
    NameChanged,
    SizeChanged,
    ModifiedAtChanged,
    StorageObjectChanged,
}

/// Closed failure vocabulary for aggregate artifact accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupArtifactAccountingError {
    BytesOverflow,
}

/// Immutable observation of one private recovery image or interrupted file.
#[derive(Debug, Clone)]
pub(crate) struct BackupArtifactSnapshot {
    path: PathBuf,
    file_name: String,
    size_bytes: u64,
    modified_at: SystemTime,
    storage_identity: BackupStorageIdentity,
}

impl BackupArtifactSnapshot {
    pub(crate) fn new(
        path: PathBuf,
        file_name: String,
        size_bytes: u64,
        modified_at: SystemTime,
        storage_identity: BackupStorageIdentity,
    ) -> Self {
        Self {
            path,
            file_name,
            size_bytes,
            modified_at,
            storage_identity,
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn path_buf(&self) -> PathBuf {
        self.path.clone()
    }

    pub(crate) fn file_name(&self) -> &str {
        &self.file_name
    }

    pub(crate) fn file_name_owned(&self) -> String {
        self.file_name.clone()
    }

    pub(crate) const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub(crate) const fn modified_at(&self) -> SystemTime {
        self.modified_at
    }

    pub(crate) const fn device_id(&self) -> u64 {
        self.storage_identity.device_id()
    }

    pub(crate) const fn inode(&self) -> u64 {
        self.storage_identity.inode()
    }

    pub(crate) fn identity_state(&self, actual: &Self) -> BackupArtifactIdentityState {
        if self.path != actual.path {
            BackupArtifactIdentityState::PathChanged
        } else if self.file_name != actual.file_name {
            BackupArtifactIdentityState::NameChanged
        } else if self.size_bytes != actual.size_bytes {
            BackupArtifactIdentityState::SizeChanged
        } else if self.modified_at != actual.modified_at {
            BackupArtifactIdentityState::ModifiedAtChanged
        } else if self.storage_identity != actual.storage_identity {
            BackupArtifactIdentityState::StorageObjectChanged
        } else {
            BackupArtifactIdentityState::Stable
        }
    }

    pub(crate) const fn verified_size_matches(&self, verified_size: u64) -> bool {
        self.size_bytes == verified_size
    }

    pub(crate) fn checked_total_bytes(
        artifacts: &[Self],
    ) -> Result<u64, BackupArtifactAccountingError> {
        artifacts.iter().try_fold(0u64, |total, artifact| {
            total
                .checked_add(artifact.size_bytes)
                .ok_or(BackupArtifactAccountingError::BytesOverflow)
        })
    }
}

impl BackupRetentionArtifact for BackupArtifactSnapshot {
    fn size_bytes(&self) -> u64 {
        self.size_bytes()
    }

    fn modified_at(&self) -> SystemTime {
        self.modified_at()
    }

    fn stable_name(&self) -> &str {
        self.file_name()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;

    fn storage_identity(device_id: u64, inode: u64) -> BackupStorageIdentity {
        #[cfg(unix)]
        {
            BackupStorageIdentity::Unix { device_id, inode }
        }
        #[cfg(not(unix))]
        {
            let _ = (device_id, inode);
            BackupStorageIdentity::Portable
        }
    }

    fn artifact(path: &str, name: &str, size: u64, modified_at: u64) -> BackupArtifactSnapshot {
        BackupArtifactSnapshot::new(
            PathBuf::from(path),
            name.to_string(),
            size,
            UNIX_EPOCH + Duration::from_secs(modified_at),
            storage_identity(7, 11),
        )
    }

    #[test]
    fn exact_snapshot_is_stable() {
        let expected = artifact("/private/backup", "backup.sqlite", 42, 100);
        let actual = expected.clone();

        assert_eq!(
            expected.identity_state(&actual),
            BackupArtifactIdentityState::Stable
        );
        assert!(expected.verified_size_matches(42));
        assert_eq!(expected.path(), Path::new("/private/backup"));
        assert_eq!(expected.file_name(), "backup.sqlite");
    }

    #[test]
    fn every_common_identity_field_fails_closed() {
        let expected = artifact("/private/backup", "backup.sqlite", 42, 100);

        let cases = [
            (
                artifact("/private/other", "backup.sqlite", 42, 100),
                BackupArtifactIdentityState::PathChanged,
            ),
            (
                artifact("/private/backup", "other.sqlite", 42, 100),
                BackupArtifactIdentityState::NameChanged,
            ),
            (
                artifact("/private/backup", "backup.sqlite", 43, 100),
                BackupArtifactIdentityState::SizeChanged,
            ),
            (
                artifact("/private/backup", "backup.sqlite", 42, 101),
                BackupArtifactIdentityState::ModifiedAtChanged,
            ),
        ];

        for (actual, state) in cases {
            assert_eq!(expected.identity_state(&actual), state);
        }
        assert!(!expected.verified_size_matches(41));
    }

    #[cfg(unix)]
    #[test]
    fn unix_storage_object_replacement_fails_closed() {
        let expected = artifact("/private/backup", "backup.sqlite", 42, 100);
        let actual = BackupArtifactSnapshot::new(
            expected.path_buf(),
            expected.file_name_owned(),
            expected.size_bytes(),
            expected.modified_at(),
            BackupStorageIdentity::Unix {
                device_id: 7,
                inode: 12,
            },
        );

        assert_eq!(
            expected.identity_state(&actual),
            BackupArtifactIdentityState::StorageObjectChanged
        );
        assert_eq!(expected.device_id(), 7);
        assert_eq!(expected.inode(), 11);
    }

    #[test]
    fn checked_accounting_reports_exact_total() {
        let artifacts = [
            artifact("/private/one", "one.sqlite", 10, 100),
            artifact("/private/two", "two.sqlite", 20, 101),
        ];

        assert_eq!(
            BackupArtifactSnapshot::checked_total_bytes(&artifacts),
            Ok(30)
        );
    }

    #[test]
    fn checked_accounting_fails_closed_on_overflow() {
        let artifacts = [
            artifact("/private/one", "one.sqlite", u64::MAX, 100),
            artifact("/private/two", "two.sqlite", 1, 101),
        ];

        assert_eq!(
            BackupArtifactSnapshot::checked_total_bytes(&artifacts),
            Err(BackupArtifactAccountingError::BytesOverflow)
        );
    }
}
