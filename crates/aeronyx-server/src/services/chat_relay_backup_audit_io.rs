// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_io.rs
// ============================================
// Version: 1.3.0-PairedLinkRecovery
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-AUDIT-IO-DOMAIN 2026-08-27 by Codex] Extract bounded
//   audit artifact reads and crash-safe publication from the relay service.
//
// Modification Reason:
//   [CHAT-RELAY-BACKUP-PAIRED-LINK-RECOVERY 2026-08-31 by Codex] Recover only
//   exact identity-proven checkpoint/temp and active/segment crash pairs.
//   [CHAT-BACKUP-AUDIT-MAINTENANCE-DOMAIN 2026-08-28 by Codex] Documented the
//   coordinator that now composes this I/O capability with chain policies.
//
// Main Functionality:
//   - Catalogs immutable audit segment and checkpoint files.
//   - Cleans bounded abandoned checkpoint temporaries under the host lock.
//   - Reads checkpoint JSON and hashes immutable segments with race detection.
//   - Publishes checkpoints and finalizes segment rotation crash windows.
//
// Dependencies:
//   - `chat_relay_backup_io` supplies private no-follow file operations.
//   - Audit catalog/checkpoint/rotation modules supply typed path-free data.
//   - Audit maintenance sequences verification, recovery, rotation, and append.
//   - `sha2` and `serde_json` preserve the existing artifact contracts.
//
// Main Logical Flow:
//   1. Catalog canonical audit artifacts from a private parent directory.
//   2. Read or hash an artifact through owner-private file descriptors.
//   3. Reserve and fsync a private temporary checkpoint.
//   4. Publish with a hard link, fsync the parent, then retire the temporary.
//   5. Recover either supported segment-publication crash window explicitly.
//
// Important Note for Next Developer:
//   - Keep filenames, size ceilings, and error messages backward compatible.
//   - Never follow symbolic links or accept changing files during verification.
//   - Publication order and both parent-directory fsync calls are required.
//   - HMAC policy and chain state transitions belong outside this I/O module.
//
// Last Modified:
//   v1.3.0-PairedLinkRecovery - Identity-bound two-link cleanup and retirement
//   v1.2.0-MaintenanceCoordinatorComposition - Documented use-case ownership
//   v1.1.0-CanonicalSegmentNaming - Exposed canonical segment naming to the
//     composed chain verifier without leaking catalog implementation details
//   v1.0.0-BackupAuditIoDomain - Initial bounded audit I/O extraction
// ============================================

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::services::chat_relay_backup_audit_catalog::{
    BackupAuditCatalogError, BackupAuditCatalogFiles, BackupAuditSegmentCatalog,
    BoundedBackupAuditSegmentCatalog,
};
use crate::services::chat_relay_backup_audit_checkpoint::ChatRelayBackupAuditCheckpoint;
use crate::services::chat_relay_backup_audit_rotation::ChatRelayBackupAuditSegmentRange;
use crate::services::chat_relay_backup_io::{
    backup_io_error, BackupFilesystem, PrivateBackupControlFileIdentity,
    PrivateBackupControlFileMode,
};
use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Private sibling file holding aggregate-only maintenance audit records.
pub(super) const BACKUP_AUDIT_FILE_NAME: &str = ".aeronyx-relay-backup-maintenance-audit.jsonl";
/// Prefix for owner-private checkpoints not yet atomically published.
pub(super) const BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX: &str =
    ".aeronyx-relay-backup-maintenance-audit.tmp-checkpoint-";
/// Hard ceiling for one active or immutable audit segment.
pub(super) const BACKUP_AUDIT_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum immutable segments accepted by one local verification.
pub(super) const BACKUP_AUDIT_MAX_SEGMENTS: usize = 16;
/// Defensive ceiling for one private checkpoint JSON document.
pub(super) const BACKUP_AUDIT_CHECKPOINT_MAX_BYTES: u64 = 4096;
/// Maximum abandoned checkpoint temporaries cleaned in one locked append.
pub(super) const BACKUP_AUDIT_CHECKPOINT_TEMP_MAX_FILES: usize = 64;

/// Path-bearing recovery action produced by authenticated chain verification.
#[derive(Debug)]
pub(super) enum ChatRelayBackupAuditPendingRotation {
    /// A checkpoint exists but the active segment has not acquired its name.
    PublishSegment {
        active_path: PathBuf,
        segment_path: PathBuf,
    },
    /// The named segment exists but the duplicate active link remains.
    RemoveDuplicateActive {
        active_path: PathBuf,
        expected_identity: PrivateBackupControlFileIdentity,
    },
}

/// Bounded host I/O required by backup-audit verification and publication.
pub(super) trait BackupAuditIo {
    /// Returns the canonical immutable segment name for one validated range.
    fn segment_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String;

    /// Returns the canonical immutable checkpoint name for one validated range.
    #[cfg(test)]
    fn checkpoint_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String;

    /// Catalogs canonical immutable segment/checkpoint names.
    fn collect_segment_files(
        &self,
        parent: &Path,
    ) -> ChatRelayResult<BTreeMap<ChatRelayBackupAuditSegmentRange, BackupAuditCatalogFiles>>;

    /// Removes bounded abandoned checkpoint temporaries and syncs the parent.
    fn cleanup_checkpoint_temporaries(&self, parent: &Path) -> ChatRelayResult<()>;

    /// Reads one stable bounded checkpoint document.
    fn read_checkpoint(&self, path: &Path) -> ChatRelayResult<ChatRelayBackupAuditCheckpoint>;

    /// Hashes one stable bounded segment from its existing private descriptor.
    fn hash_segment(&self, file: &mut File) -> ChatRelayResult<(u64, String)>;

    /// Completes one authenticated segment-publication crash window.
    fn complete_pending_rotation(
        &self,
        parent: &Path,
        pending: ChatRelayBackupAuditPendingRotation,
    ) -> ChatRelayResult<()>;

    /// Atomically publishes one bounded immutable checkpoint.
    fn publish_checkpoint(
        &self,
        parent: &Path,
        range: ChatRelayBackupAuditSegmentRange,
        checkpoint: &ChatRelayBackupAuditCheckpoint,
    ) -> ChatRelayResult<()>;
}

/// Production audit I/O composed over a replaceable private filesystem.
#[derive(Debug, Clone, Copy)]
pub(super) struct LocalBackupAuditIo<F> {
    filesystem: F,
}

impl<F> LocalBackupAuditIo<F> {
    /// Composes audit publication over the supplied private filesystem.
    pub(super) const fn new(filesystem: F) -> Self {
        Self { filesystem }
    }

    fn map_catalog_error(error: BackupAuditCatalogError) -> ChatRelayError {
        match error {
            BackupAuditCatalogError::MalformedName => backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit segment name is malformed",
            ),
            BackupAuditCatalogError::AmbiguousName => backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit segment name is ambiguous",
            ),
            BackupAuditCatalogError::InvalidRange => backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit segment range is invalid",
            ),
            BackupAuditCatalogError::DuplicateArtifact => backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit segment is duplicated",
            ),
            BackupAuditCatalogError::SegmentLimitReached => backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit segment limit reached",
            ),
        }
    }

    fn segment_catalog() -> BoundedBackupAuditSegmentCatalog {
        BoundedBackupAuditSegmentCatalog::new(BACKUP_AUDIT_MAX_SEGMENTS)
    }

    fn checkpoint_file_name(range: ChatRelayBackupAuditSegmentRange) -> String {
        Self::segment_catalog().checkpoint_file_name(range)
    }
}

impl<F: BackupFilesystem> LocalBackupAuditIo<F> {
    fn checkpoint_temporary_paths(&self, parent: &Path) -> ChatRelayResult<Vec<PathBuf>> {
        let entries = std::fs::read_dir(parent).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect relay backup maintenance audit temporaries",
            )
        })?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect relay backup maintenance audit temporary",
                )
            })?;
            let Some(file_name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Some(nonce) = file_name.strip_prefix(BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX) else {
                continue;
            };
            if !is_lower_hex(nonce, 16) {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup maintenance audit temporary name is malformed",
                ));
            }
            if paths.len() >= BACKUP_AUDIT_CHECKPOINT_TEMP_MAX_FILES {
                return Err(backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup maintenance audit temporary limit reached",
                ));
            }
            paths.push(entry.path());
        }
        Ok(paths)
    }

    fn open_checkpoint_for_read(&self, path: &Path) -> ChatRelayResult<Option<File>> {
        let parent = path.parent().ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup maintenance audit checkpoint has no private parent",
            )
        })?;
        for temporary_path in self.checkpoint_temporary_paths(parent)? {
            if let Some(pair) = self
                .filesystem
                .open_existing_control_file_pair(path, &temporary_path)?
            {
                let checkpoint = pair.first;
                drop(pair.second);
                return Ok(Some(checkpoint));
            }
        }
        self.filesystem.open_existing_control_file(path)
    }
}

impl<F: BackupFilesystem> BackupAuditIo for LocalBackupAuditIo<F> {
    fn segment_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String {
        // [CHAT-RELAY-BACKUP-AUDIT-CHAIN-DOMAIN 2026-08-27 by Codex] Keep
        // canonical artifact naming behind the I/O boundary consumed by the
        // chain verifier and service compatibility wrappers.
        Self::segment_catalog().segment_file_name(range)
    }

    #[cfg(test)]
    fn checkpoint_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String {
        Self::segment_catalog().checkpoint_file_name(range)
    }

    fn collect_segment_files(
        &self,
        parent: &Path,
    ) -> ChatRelayResult<BTreeMap<ChatRelayBackupAuditSegmentRange, BackupAuditCatalogFiles>> {
        let mut catalog = Self::segment_catalog();
        let entries = std::fs::read_dir(parent).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect relay backup maintenance audit segments",
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect relay backup maintenance audit segment",
                )
            })?;
            let Some(file_name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            catalog
                .insert_file_name(file_name)
                .map_err(Self::map_catalog_error)?;
        }
        Ok(catalog.into_files())
    }

    fn cleanup_checkpoint_temporaries(&self, parent: &Path) -> ChatRelayResult<()> {
        let checkpoint_paths = self
            .collect_segment_files(parent)?
            .into_values()
            .filter_map(|files| files.checkpoint_file_name)
            .map(|file_name| parent.join(file_name))
            .collect::<Vec<_>>();
        let temporary_paths = self.checkpoint_temporary_paths(parent)?;
        let removed = temporary_paths.len();
        for path in temporary_paths {
            let mut published_identity = None;
            for checkpoint_path in &checkpoint_paths {
                if let Some(pair) = self
                    .filesystem
                    .open_existing_control_file_pair(&path, checkpoint_path)?
                {
                    published_identity = Some(pair.identity);
                    drop(pair);
                    break;
                }
            }
            if let Some(identity) = published_identity {
                // [CHAT-RELAY-BACKUP-PAIRED-LINK-RECOVERY 2026-08-31 by Codex]
                // The canonical checkpoint is the inode's only other name.
                // Revalidate the temporary identity immediately before unlink.
                self.filesystem
                    .remove_verified_control_link(parent, &path, identity)?;
                self.filesystem.sync_backup_parent(parent)?;
                continue;
            }
            let Some(file) = self.filesystem.open_existing_control_file(&path)? else {
                continue;
            };
            drop(file);
            std::fs::remove_file(&path).map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR_DELETE,
                    "unable to remove abandoned relay backup maintenance audit temporary",
                )
            })?;
        }
        if removed > 0 {
            self.filesystem.sync_backup_parent(parent)?;
        }
        Ok(())
    }

    fn read_checkpoint(&self, path: &Path) -> ChatRelayResult<ChatRelayBackupAuditCheckpoint> {
        let Some(mut file) = self.open_checkpoint_for_read(path)? else {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint is missing",
            ));
        };
        let initial_len = file
            .metadata()
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect relay backup maintenance audit checkpoint",
                )
            })?
            .len();
        if initial_len == 0 || initial_len > BACKUP_AUDIT_CHECKPOINT_MAX_BYTES {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint size is invalid",
            ));
        }
        let mut encoded = Vec::with_capacity(usize::try_from(initial_len).unwrap_or(0));
        Read::by_ref(&mut file)
            .take(BACKUP_AUDIT_CHECKPOINT_MAX_BYTES + 1)
            .read_to_end(&mut encoded)
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to read relay backup maintenance audit checkpoint",
                )
            })?;
        let final_len = file
            .metadata()
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to re-inspect relay backup maintenance audit checkpoint",
                )
            })?
            .len();
        if encoded.len() as u64 != initial_len || final_len != initial_len {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint changed during verification",
            ));
        }
        serde_json::from_slice(&encoded).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint is malformed",
            )
        })
    }

    fn hash_segment(&self, file: &mut File) -> ChatRelayResult<(u64, String)> {
        let initial_len = file
            .metadata()
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect relay backup maintenance audit segment",
                )
            })?
            .len();
        if initial_len == 0 || initial_len > BACKUP_AUDIT_MAX_BYTES {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit segment size is invalid",
            ));
        }
        file.seek(SeekFrom::Start(0)).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to hash relay backup maintenance audit segment",
            )
        })?;
        let mut hasher = Sha256::new();
        let mut reader = Read::by_ref(file).take(BACKUP_AUDIT_MAX_BYTES + 1);
        let mut copied = 0u64;
        let mut buffer = [0u8; 16 * 1024];
        loop {
            let read = reader.read(&mut buffer).map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to hash relay backup maintenance audit segment",
                )
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            copied = copied.checked_add(read as u64).ok_or_else(|| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_FULL,
                    "relay backup maintenance audit segment hash size overflow",
                )
            })?;
        }
        let final_len = file
            .metadata()
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to re-inspect relay backup maintenance audit segment",
                )
            })?
            .len();
        if copied != initial_len || final_len != initial_len {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit segment changed during verification",
            ));
        }
        Ok((copied, hex::encode(hasher.finalize())))
    }

    fn complete_pending_rotation(
        &self,
        parent: &Path,
        pending: ChatRelayBackupAuditPendingRotation,
    ) -> ChatRelayResult<()> {
        match pending {
            ChatRelayBackupAuditPendingRotation::PublishSegment {
                active_path,
                segment_path,
            } => {
                std::fs::hard_link(&active_path, &segment_path).map_err(|_| {
                    backup_io_error(
                        rusqlite::ffi::SQLITE_IOERR,
                        "unable to publish relay backup maintenance audit segment",
                    )
                })?;
                self.filesystem.sync_backup_parent(parent)?;
                std::fs::remove_file(&active_path).map_err(|_| {
                    backup_io_error(
                        rusqlite::ffi::SQLITE_IOERR_DELETE,
                        "unable to retire active relay backup maintenance audit segment",
                    )
                })?;
                self.filesystem.sync_backup_parent(parent)
            }
            ChatRelayBackupAuditPendingRotation::RemoveDuplicateActive {
                active_path,
                expected_identity,
            } => {
                self.filesystem.remove_verified_control_link(
                    parent,
                    &active_path,
                    expected_identity,
                )?;
                self.filesystem.sync_backup_parent(parent)
            }
        }
    }

    fn publish_checkpoint(
        &self,
        parent: &Path,
        range: ChatRelayBackupAuditSegmentRange,
        checkpoint: &ChatRelayBackupAuditCheckpoint,
    ) -> ChatRelayResult<()> {
        let encoded = serde_json::to_vec(checkpoint).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_FORMAT,
                "unable to encode relay backup maintenance audit checkpoint",
            )
        })?;
        if encoded.is_empty() || encoded.len() as u64 > BACKUP_AUDIT_CHECKPOINT_MAX_BYTES {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_TOOBIG,
                "relay backup maintenance audit checkpoint exceeds its bounded size",
            ));
        }
        let checkpoint_path = parent.join(Self::checkpoint_file_name(range));
        let temporary_path = parent.join(format!(
            "{BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX}{:016x}",
            rand::random::<u64>()
        ));
        self.filesystem.reserve_private_file(&temporary_path)?;
        let mut published = false;
        let outcome = (|| -> ChatRelayResult<()> {
            let mut temporary = self
                .filesystem
                .open_control_file(&temporary_path, PrivateBackupControlFileMode::ReadWrite)?;
            temporary.write_all(&encoded).map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR_WRITE,
                    "unable to write relay backup maintenance audit checkpoint",
                )
            })?;
            temporary.sync_all().map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR_FSYNC,
                    "unable to sync relay backup maintenance audit checkpoint",
                )
            })?;
            drop(temporary);
            std::fs::hard_link(&temporary_path, &checkpoint_path).map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_CONSTRAINT,
                    "unable to publish relay backup maintenance audit checkpoint",
                )
            })?;
            published = true;
            self.filesystem.sync_backup_parent(parent)?;
            std::fs::remove_file(&temporary_path).map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR_DELETE,
                    "unable to finalize relay backup maintenance audit checkpoint",
                )
            })?;
            self.filesystem.sync_backup_parent(parent)
        })();
        let _ = std::fs::remove_file(&temporary_path);
        if published && outcome.is_err() {
            let _ = self.filesystem.sync_backup_parent(parent);
        }
        outcome
    }
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    use super::*;
    use crate::services::chat_relay_backup_io::LocalBackupFilesystem;

    fn audit_io() -> LocalBackupAuditIo<LocalBackupFilesystem> {
        LocalBackupAuditIo::new(LocalBackupFilesystem)
    }

    fn range(first: u64, last: u64) -> ChatRelayBackupAuditSegmentRange {
        ChatRelayBackupAuditSegmentRange::new(first, last).expect("valid audit range")
    }

    fn checkpoint(range: ChatRelayBackupAuditSegmentRange) -> ChatRelayBackupAuditCheckpoint {
        ChatRelayBackupAuditCheckpoint {
            version: 1,
            checkpoint_index: 1,
            segment_first_sequence: range.first_sequence,
            segment_last_sequence: range.last_sequence,
            segment_bytes: 7,
            segment_sha256: "11".repeat(32),
            cumulative_verified_bytes: 7,
            cumulative_last_recorded_at: Some(1_700_000_000),
            cumulative_dry_run_count: 1,
            cumulative_planned_count: 0,
            cumulative_completed_count: 0,
            cumulative_failed_count: 0,
            head_mac: "22".repeat(32),
            previous_checkpoint_mac: "33".repeat(32),
            checkpoint_mac: "44".repeat(32),
        }
    }

    #[test]
    fn checkpoint_publication_round_trips_without_temporary_artifacts() {
        let parent = tempfile::tempdir().expect("temporary audit parent");
        let io = audit_io();
        let range = range(1, 1);
        let expected = checkpoint(range);

        io.publish_checkpoint(parent.path(), range, &expected)
            .expect("publish checkpoint");
        let checkpoint_path = parent
            .path()
            .join(BoundedBackupAuditSegmentCatalog::new(1).checkpoint_file_name(range));
        let actual = io
            .read_checkpoint(&checkpoint_path)
            .expect("read published checkpoint");

        assert_eq!(actual, expected);
        assert!(std::fs::read_dir(parent.path())
            .expect("audit entries")
            .all(|entry| !entry
                .expect("audit entry")
                .file_name()
                .to_string_lossy()
                .starts_with(BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX)));
    }

    #[test]
    #[cfg(unix)]
    fn published_checkpoint_temporary_pair_is_readable_and_cleanup_is_idempotent() {
        use std::io::Write;
        use std::os::unix::fs::MetadataExt;

        let parent = tempfile::tempdir().expect("temporary checkpoint crash parent");
        let io = audit_io();
        let filesystem = LocalBackupFilesystem;
        let range = range(1, 1);
        let expected = checkpoint(range);
        let checkpoint_path = parent.path().join(io.checkpoint_file_name(range));
        let temporary_path = parent.path().join(format!(
            "{BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX}0123456789abcdef"
        ));
        filesystem
            .reserve_private_file(&temporary_path)
            .expect("reserve checkpoint temporary");
        let mut temporary = filesystem
            .open_control_file(&temporary_path, PrivateBackupControlFileMode::ReadWrite)
            .expect("open checkpoint temporary");
        temporary
            .write_all(&serde_json::to_vec(&expected).expect("encode checkpoint"))
            .expect("write checkpoint temporary");
        temporary.sync_all().expect("sync checkpoint temporary");
        drop(temporary);
        std::fs::hard_link(&temporary_path, &checkpoint_path)
            .expect("publish checkpoint hard link before crash");

        assert_eq!(
            io.read_checkpoint(&checkpoint_path)
                .expect("read identity-bound checkpoint pair"),
            expected
        );
        io.cleanup_checkpoint_temporaries(parent.path())
            .expect("recover checkpoint publication temporary");
        assert!(!temporary_path.exists());
        assert_eq!(
            std::fs::metadata(&checkpoint_path)
                .expect("retained checkpoint metadata")
                .nlink(),
            1
        );
        io.cleanup_checkpoint_temporaries(parent.path())
            .expect("checkpoint cleanup retry is idempotent");
        assert!(checkpoint_path.exists());
    }

    #[test]
    #[cfg(unix)]
    fn checkpoint_pair_with_third_link_or_mode_drift_has_zero_cleanup_effect() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        for drift_mode in [false, true] {
            let parent = tempfile::tempdir().expect("temporary invalid checkpoint pair parent");
            let io = audit_io();
            let filesystem = LocalBackupFilesystem;
            let range = range(1, 1);
            let checkpoint_path = parent.path().join(io.checkpoint_file_name(range));
            let temporary_path = parent.path().join(format!(
                "{BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX}fedcba9876543210"
            ));
            filesystem
                .reserve_private_file(&temporary_path)
                .expect("reserve invalid checkpoint temporary");
            std::fs::hard_link(&temporary_path, &checkpoint_path)
                .expect("publish invalid checkpoint pair");
            let third_path = parent.path().join("unmanaged-third-link");
            if drift_mode {
                std::fs::set_permissions(&checkpoint_path, std::fs::Permissions::from_mode(0o644))
                    .expect("drift checkpoint pair mode");
            } else {
                std::fs::hard_link(&temporary_path, &third_path)
                    .expect("add forbidden checkpoint third link");
            }

            assert!(io.cleanup_checkpoint_temporaries(parent.path()).is_err());
            assert!(temporary_path.exists() && checkpoint_path.exists());
            if drift_mode {
                assert_eq!(
                    std::fs::metadata(&temporary_path)
                        .expect("drifted temporary metadata")
                        .nlink(),
                    2
                );
            } else {
                assert!(third_path.exists());
            }
        }
    }

    #[test]
    fn cleanup_removes_only_canonical_abandoned_temporaries() {
        let parent = tempfile::tempdir().expect("temporary audit parent");
        let io = audit_io();
        let temporary = parent.path().join(format!(
            "{BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX}0123456789abcdef"
        ));
        let unrelated = parent.path().join("operator-note");
        LocalBackupFilesystem
            .reserve_private_file(&temporary)
            .expect("reserve abandoned temporary");
        std::fs::write(&unrelated, b"keep").expect("write unrelated file");

        io.cleanup_checkpoint_temporaries(parent.path())
            .expect("cleanup temporary");

        assert!(!temporary.exists());
        assert!(unrelated.exists());
    }

    #[test]
    fn malformed_checkpoint_temporary_fails_closed() {
        let parent = tempfile::tempdir().expect("temporary audit parent");
        let malformed = parent.path().join(format!(
            "{BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX}NOT-LOWER-HEX"
        ));
        LocalBackupFilesystem
            .reserve_private_file(&malformed)
            .expect("reserve malformed temporary");

        assert!(audit_io()
            .cleanup_checkpoint_temporaries(parent.path())
            .is_err());
        assert!(malformed.exists());
    }

    #[test]
    fn segment_hash_is_exact_and_empty_segments_fail_closed() {
        use std::io::Write;

        let parent = tempfile::tempdir().expect("temporary audit parent");
        let segment_path = parent.path().join("segment");
        let empty_path = parent.path().join("empty");
        let filesystem = LocalBackupFilesystem;
        let mut segment = filesystem
            .open_control_file(&segment_path, PrivateBackupControlFileMode::ReadWrite)
            .expect("segment file");
        segment.write_all(b"record\n").expect("segment bytes");
        segment.sync_all().expect("sync segment");
        let (bytes, digest) = audit_io().hash_segment(&mut segment).expect("hash segment");
        assert_eq!(bytes, 7);
        assert_eq!(digest, hex::encode(Sha256::digest(b"record\n")));

        let mut empty = filesystem
            .open_control_file(&empty_path, PrivateBackupControlFileMode::ReadWrite)
            .expect("empty segment file");
        assert!(audit_io().hash_segment(&mut empty).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn pending_rotation_publishes_same_inode_and_retires_active_name() {
        use std::io::Write;

        let parent = tempfile::tempdir().expect("temporary audit parent");
        let active_path = parent.path().join(BACKUP_AUDIT_FILE_NAME);
        let segment_path = parent.path().join("immutable-segment");
        let mut active = LocalBackupFilesystem
            .open_control_file(&active_path, PrivateBackupControlFileMode::ReadWrite)
            .expect("active segment");
        active.write_all(b"record\n").expect("active record");
        active.sync_all().expect("sync active segment");
        let active_inode = active.metadata().expect("active metadata").ino();
        drop(active);

        audit_io()
            .complete_pending_rotation(
                parent.path(),
                ChatRelayBackupAuditPendingRotation::PublishSegment {
                    active_path: active_path.clone(),
                    segment_path: segment_path.clone(),
                },
            )
            .expect("complete segment publication");

        assert!(!active_path.exists());
        assert_eq!(
            std::fs::metadata(segment_path)
                .expect("published segment metadata")
                .ino(),
            active_inode
        );
    }

    #[test]
    fn duplicate_active_recovery_retires_only_the_active_name() {
        let parent = tempfile::tempdir().expect("temporary audit parent");
        let active_path = parent.path().join(BACKUP_AUDIT_FILE_NAME);
        let segment_path = parent.path().join("immutable-segment");
        LocalBackupFilesystem
            .reserve_private_file(&active_path)
            .expect("reserve duplicate active segment");
        std::fs::hard_link(&active_path, &segment_path)
            .expect("publish duplicate active segment link");
        let pair = LocalBackupFilesystem
            .open_existing_control_file_pair(&active_path, &segment_path)
            .expect("inspect duplicate active pair")
            .expect("duplicate active pair is exact");
        let expected_identity = pair.identity;
        drop(pair);

        audit_io()
            .complete_pending_rotation(
                parent.path(),
                ChatRelayBackupAuditPendingRotation::RemoveDuplicateActive {
                    active_path: active_path.clone(),
                    expected_identity,
                },
            )
            .expect("retire duplicate active segment");

        assert!(!active_path.exists());
        assert!(segment_path.exists());
    }

    #[test]
    fn catalog_collects_only_canonical_audit_artifacts() {
        let parent = tempfile::tempdir().expect("temporary audit parent");
        let range = range(4, 9);
        let catalog = BoundedBackupAuditSegmentCatalog::new(1);
        std::fs::write(
            parent.path().join(catalog.segment_file_name(range)),
            b"segment",
        )
        .expect("segment artifact");
        std::fs::write(
            parent.path().join(catalog.checkpoint_file_name(range)),
            b"checkpoint",
        )
        .expect("checkpoint artifact");
        std::fs::write(parent.path().join("unmanaged"), b"ignore").expect("unmanaged artifact");

        let files = audit_io()
            .collect_segment_files(parent.path())
            .expect("collect audit catalog");

        assert_eq!(files.len(), 1);
        let pair = files.get(&range).expect("catalog range");
        assert!(pair.segment_file_name.is_some());
        assert!(pair.checkpoint_file_name.is_some());
    }
}
