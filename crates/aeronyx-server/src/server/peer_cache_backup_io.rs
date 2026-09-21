// ============================================
// File: crates/aeronyx-server/src/server/peer_cache_backup_io.rs
// ============================================
// [PEER-CACHE-BACKUP-DURABILITY 2026-09-21 by Codex] Owns the bounded,
// blocking filesystem transaction that protects the previous complete
// PeerStore cache before publishing a newly synced snapshot. Protocol bytes,
// signing, witness policy, and lifecycle decisions remain in the server root.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The existing discovery codec admits at most eight MiB. The backup path uses
/// the same independent bound before allocating or copying old primary bytes.
pub(super) const PEER_CACHE_BACKUP_MAX_BYTES: usize = 8 * 1024 * 1024;
const TEMP_CREATE_ATTEMPTS: usize = 32;
const COPY_CHECKPOINT_BYTES: usize = 64 * 1024;
const TEMP_PREFIX: &str = ".aeronyx-peer-cache";

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Coarse typed failure for the durable cache transaction.
///
/// `Ambiguous` means an atomic replace completed but its containing directory
/// could not be synced. Callers must not publish a later dependent effect and
/// must rearm the dirty generation for operator-visible retry.
#[derive(Debug)]
pub(super) enum PeerCacheBackupIoError {
    SnapshotTooLarge {
        max_bytes: usize,
    },
    OldPrimaryTooLarge {
        max_bytes: usize,
    },
    OldPrimaryChanged,
    Io {
        stage: &'static str,
        source: io::Error,
    },
    Ambiguous {
        stage: &'static str,
        source: io::Error,
    },
    WorkerFailed,
}

impl fmt::Display for PeerCacheBackupIoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SnapshotTooLarge { max_bytes } => {
                write!(formatter, "peer cache snapshot exceeds {max_bytes} bytes")
            }
            Self::OldPrimaryTooLarge { max_bytes } => {
                write!(
                    formatter,
                    "peer cache old primary exceeds {max_bytes} bytes"
                )
            }
            Self::OldPrimaryChanged => {
                formatter.write_str("peer cache old primary changed during pinned read")
            }
            Self::Io { stage, source } => {
                write!(
                    formatter,
                    "peer cache durable write failed at {stage}: {source}"
                )
            }
            Self::Ambiguous { stage, source } => write!(
                formatter,
                "peer cache durable write is ambiguous after {stage}: {source}"
            ),
            Self::WorkerFailed => formatter.write_str("peer cache blocking worker failed"),
        }
    }
}

impl std::error::Error for PeerCacheBackupIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::Ambiguous { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IoCheckpoint {
    AfterOldPrimaryMetadata,
    AfterBackupPartialWrite,
    BeforeBackupTempSync,
    BeforeBackupReplace,
    BeforeBackupParentSync,
    BeforePrimaryReplace,
    BeforePrimaryParentSync,
}

trait IoSeam: Send + Sync {
    fn checkpoint(&self, _checkpoint: IoCheckpoint, _primary: &Path) -> io::Result<()> {
        Ok(())
    }
}

struct ProductionIo;

impl IoSeam for ProductionIo {}

/// Runs every filesystem operation, including the bounded old-primary read,
/// away from Tokio workers. Cancellation of the awaiting task does not cancel
/// the blocking closure halfway through an atomic replacement sequence.
pub(super) async fn publish_peer_cache_snapshot(
    primary: PathBuf,
    snapshot: Vec<u8>,
) -> Result<(), PeerCacheBackupIoError> {
    if snapshot.len() > PEER_CACHE_BACKUP_MAX_BYTES {
        return Err(PeerCacheBackupIoError::SnapshotTooLarge {
            max_bytes: PEER_CACHE_BACKUP_MAX_BYTES,
        });
    }

    tokio::task::spawn_blocking(move || publish_sync(&primary, &snapshot, &ProductionIo))
        .await
        .map_err(|_| PeerCacheBackupIoError::WorkerFailed)?
}

fn publish_sync(
    primary: &Path,
    snapshot: &[u8],
    seam: &dyn IoSeam,
) -> Result<(), PeerCacheBackupIoError> {
    if snapshot.len() > PEER_CACHE_BACKUP_MAX_BYTES {
        return Err(PeerCacheBackupIoError::SnapshotTooLarge {
            max_bytes: PEER_CACHE_BACKUP_MAX_BYTES,
        });
    }

    let parent = parent_dir(primary);
    fs::create_dir_all(parent).map_err(|source| PeerCacheBackupIoError::Io {
        stage: "parent_create",
        source,
    })?;

    // Sync the future primary before touching the old durable backup. If any
    // later pre-replace step fails, RAII removes only this exact unique temp.
    let mut primary_temp =
        create_unique_temp(primary, "primary").map_err(|source| PeerCacheBackupIoError::Io {
            stage: "primary_temp_create",
            source,
        })?;
    write_synced_temp(&mut primary_temp, snapshot, None, seam, primary).map_err(|source| {
        PeerCacheBackupIoError::Io {
            stage: "primary_temp_sync",
            source,
        }
    })?;
    primary_temp
        .sync_and_close()
        .map_err(|source| PeerCacheBackupIoError::Io {
            stage: "primary_temp_sync",
            source,
        })?;

    let backup = backup_path(primary);
    let old_primary = read_stable_old_primary(primary, seam)?;
    if let Some(old_primary) = old_primary {
        let mut backup_temp =
            create_unique_temp(&backup, "backup").map_err(|source| PeerCacheBackupIoError::Io {
                stage: "backup_temp_create",
                source,
            })?;
        write_synced_temp(
            &mut backup_temp,
            &old_primary,
            Some(IoCheckpoint::AfterBackupPartialWrite),
            seam,
            primary,
        )
        .map_err(|source| PeerCacheBackupIoError::Io {
            stage: "backup_temp_write",
            source,
        })?;
        seam.checkpoint(IoCheckpoint::BeforeBackupTempSync, primary)
            .map_err(|source| PeerCacheBackupIoError::Io {
                stage: "backup_temp_sync",
                source,
            })?;
        backup_temp
            .sync_and_close()
            .map_err(|source| PeerCacheBackupIoError::Io {
                stage: "backup_temp_sync",
                source,
            })?;
        seam.checkpoint(IoCheckpoint::BeforeBackupReplace, primary)
            .map_err(|source| PeerCacheBackupIoError::Io {
                stage: "backup_replace",
                source,
            })?;
        atomic_replace(backup_temp.path(), &backup).map_err(|source| {
            PeerCacheBackupIoError::Io {
                stage: "backup_replace",
                source,
            }
        })?;
        backup_temp.disarm();

        seam.checkpoint(IoCheckpoint::BeforeBackupParentSync, primary)
            .and_then(|()| sync_parent_dir(&backup))
            .map_err(|source| PeerCacheBackupIoError::Ambiguous {
                stage: "backup_replace_parent_sync",
                source,
            })?;
    }

    // The old-primary handle and backup temp handle are closed before either
    // replace. The new primary is published only after backup durability.
    seam.checkpoint(IoCheckpoint::BeforePrimaryReplace, primary)
        .map_err(|source| PeerCacheBackupIoError::Io {
            stage: "primary_replace",
            source,
        })?;
    atomic_replace(primary_temp.path(), primary).map_err(|source| PeerCacheBackupIoError::Io {
        stage: "primary_replace",
        source,
    })?;
    primary_temp.disarm();
    seam.checkpoint(IoCheckpoint::BeforePrimaryParentSync, primary)
        .and_then(|()| sync_parent_dir(primary))
        .map_err(|source| PeerCacheBackupIoError::Ambiguous {
            stage: "primary_replace_parent_sync",
            source,
        })?;

    Ok(())
}

fn read_stable_old_primary(
    primary: &Path,
    seam: &dyn IoSeam,
) -> Result<Option<Vec<u8>>, PeerCacheBackupIoError> {
    let mut file = match OpenOptions::new().read(true).open(primary) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(PeerCacheBackupIoError::Io {
                stage: "old_primary_open",
                source,
            });
        }
    };
    let before = file
        .metadata()
        .map_err(|source| PeerCacheBackupIoError::Io {
            stage: "old_primary_metadata",
            source,
        })?;
    if !before.is_file() {
        return Err(PeerCacheBackupIoError::Io {
            stage: "old_primary_type",
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "old primary is not a regular file",
            ),
        });
    }
    if before.len() > PEER_CACHE_BACKUP_MAX_BYTES as u64 {
        return Err(PeerCacheBackupIoError::OldPrimaryTooLarge {
            max_bytes: PEER_CACHE_BACKUP_MAX_BYTES,
        });
    }
    seam.checkpoint(IoCheckpoint::AfterOldPrimaryMetadata, primary)
        .map_err(|source| PeerCacheBackupIoError::Io {
            stage: "old_primary_read_checkpoint",
            source,
        })?;

    let capacity =
        usize::try_from(before.len()).map_err(|_| PeerCacheBackupIoError::OldPrimaryTooLarge {
            max_bytes: PEER_CACHE_BACKUP_MAX_BYTES,
        })?;
    let mut bytes = Vec::with_capacity(capacity);
    (&mut file)
        .take((PEER_CACHE_BACKUP_MAX_BYTES as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| PeerCacheBackupIoError::Io {
            stage: "old_primary_read",
            source,
        })?;
    let after = file
        .metadata()
        .map_err(|source| PeerCacheBackupIoError::Io {
            stage: "old_primary_metadata_recheck",
            source,
        })?;
    drop(file);

    if bytes.len() > PEER_CACHE_BACKUP_MAX_BYTES
        || bytes.len() as u64 != before.len()
        || after.len() != before.len()
    {
        return Err(PeerCacheBackupIoError::OldPrimaryChanged);
    }
    Ok(Some(bytes))
}

fn write_synced_temp(
    temp: &mut DurableTempFile,
    bytes: &[u8],
    partial_checkpoint: Option<IoCheckpoint>,
    seam: &dyn IoSeam,
    primary: &Path,
) -> io::Result<()> {
    let first = bytes.len().min(COPY_CHECKPOINT_BYTES);
    temp.file_mut()?.write_all(&bytes[..first])?;
    if let Some(checkpoint) = partial_checkpoint {
        seam.checkpoint(checkpoint, primary)?;
    }
    temp.file_mut()?.write_all(&bytes[first..])?;
    temp.file_mut()?.flush()?;
    let observed = temp.file_mut()?.metadata()?.len();
    if observed != bytes.len() as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "temporary file length mismatch",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct DurableTempFile {
    path: PathBuf,
    file: Option<File>,
    armed: bool,
}

impl DurableTempFile {
    fn file_mut(&mut self) -> io::Result<&mut File> {
        self.file.as_mut().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "temporary file already closed")
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn sync_and_close(&mut self) -> io::Result<()> {
        let Some(mut file) = self.file.take() else {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "temporary file already closed",
            ));
        };
        file.flush()?;
        file.sync_all()?;
        let result = file.metadata();
        drop(file);
        result.map(|_| ())
    }

    fn disarm(&mut self) {
        self.file.take();
        self.armed = false;
    }
}

impl Drop for DurableTempFile {
    fn drop(&mut self) {
        self.file.take();
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn create_unique_temp(destination: &Path, purpose: &str) -> io::Result<DurableTempFile> {
    create_unique_temp_with_sequence(destination, purpose, &TEMP_SEQUENCE)
}

fn create_unique_temp_with_sequence(
    destination: &Path,
    purpose: &str,
    sequence: &AtomicU64,
) -> io::Result<DurableTempFile> {
    let parent = parent_dir(destination);
    for _ in 0..TEMP_CREATE_ATTEMPTS {
        let sequence = sequence.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            "{TEMP_PREFIX}-{purpose}-{}-{sequence}.tmp",
            std::process::id()
        ));
        match create_temp_at(candidate) {
            Ok(temp) => return Ok(temp),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "peer cache temporary-name attempts exhausted",
    ))
}

fn create_temp_at(path: PathBuf) -> io::Result<DurableTempFile> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    Ok(DurableTempFile {
        path,
        file: Some(file),
        armed: true,
    })
}

fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn backup_path(primary: &Path) -> PathBuf {
    PathBuf::from(format!("{}.bak", primary.display()))
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, ReplaceFileW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
        REPLACEFILE_WRITE_THROUGH,
    };

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    let source = wide(source);
    let destination_wide = wide(destination);
    let replaced = if destination.exists() {
        // SAFETY: Both buffers are NUL-terminated and live for the call; all
        // optional pointers are null and Windows owns no returned handle.
        unsafe {
            ReplaceFileW(
                destination_wide.as_ptr(),
                source.as_ptr(),
                ptr::null(),
                REPLACEFILE_WRITE_THROUGH,
                ptr::null(),
                ptr::null(),
            )
        }
    } else {
        // SAFETY: Both buffers are NUL-terminated and live for the call.
        unsafe {
            MoveFileExW(
                source.as_ptr(),
                destination_wide.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        }
    };
    if replaced == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn sync_parent_dir(path: &Path) -> io::Result<()> {
    File::open(parent_dir(path))?.sync_all()
}

#[cfg(windows)]
fn sync_parent_dir(_path: &Path) -> io::Result<()> {
    // ReplaceFileW(REPLACEFILE_WRITE_THROUGH) and
    // MoveFileExW(MOVEFILE_WRITE_THROUGH) are the Windows durability boundary.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Debug, Clone, Copy)]
    enum TestAction {
        Fail,
        GrowPrimary,
        TruncatePrimary,
    }

    struct TestSeam {
        checkpoint: IoCheckpoint,
        action: TestAction,
        fired: Mutex<bool>,
    }

    impl TestSeam {
        fn new(checkpoint: IoCheckpoint, action: TestAction) -> Self {
            Self {
                checkpoint,
                action,
                fired: Mutex::new(false),
            }
        }
    }

    impl IoSeam for TestSeam {
        fn checkpoint(&self, checkpoint: IoCheckpoint, primary: &Path) -> io::Result<()> {
            if checkpoint != self.checkpoint {
                return Ok(());
            }
            let mut fired = self.fired.lock().unwrap();
            if *fired {
                return Ok(());
            }
            *fired = true;
            drop(fired);
            match self.action {
                TestAction::Fail => Err(io::Error::other("injected failure")),
                TestAction::GrowPrimary => OpenOptions::new()
                    .append(true)
                    .open(primary)?
                    .write_all(b"grown"),
                TestAction::TruncatePrimary => OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(primary)
                    .map(|_| ()),
            }
        }
    }

    fn setup_files() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("peer-cache.json");
        let backup = backup_path(&primary);
        fs::write(&primary, b"old-primary").unwrap();
        fs::write(&backup, b"older-backup").unwrap();
        (directory, primary, backup)
    }

    fn assert_no_temps(directory: &Path) {
        let entries = fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(TEMP_PREFIX))
            .collect::<Vec<_>>();
        assert!(entries.is_empty(), "leaked temps: {entries:?}");
    }

    #[test]
    fn pre_replace_failures_preserve_old_backup_byte_for_byte() {
        for checkpoint in [
            IoCheckpoint::AfterBackupPartialWrite,
            IoCheckpoint::BeforeBackupTempSync,
            IoCheckpoint::BeforeBackupReplace,
        ] {
            let (directory, primary, backup) = setup_files();
            let expected_primary = if checkpoint == IoCheckpoint::AfterBackupPartialWrite {
                let bytes = vec![0x5au8; COPY_CHECKPOINT_BYTES * 2];
                fs::write(&primary, &bytes).unwrap();
                bytes
            } else {
                b"old-primary".to_vec()
            };
            let result = publish_sync(
                &primary,
                b"new-primary",
                &TestSeam::new(checkpoint, TestAction::Fail),
            );
            assert!(matches!(result, Err(PeerCacheBackupIoError::Io { .. })));
            assert_eq!(fs::read(&primary).unwrap(), expected_primary);
            assert_eq!(fs::read(&backup).unwrap(), b"older-backup");
            assert_no_temps(directory.path());
        }
    }

    #[test]
    fn backup_parent_sync_failure_is_typed_ambiguous_and_blocks_primary() {
        let (directory, primary, backup) = setup_files();
        let result = publish_sync(
            &primary,
            b"new-primary",
            &TestSeam::new(IoCheckpoint::BeforeBackupParentSync, TestAction::Fail),
        );
        assert!(matches!(
            result,
            Err(PeerCacheBackupIoError::Ambiguous {
                stage: "backup_replace_parent_sync",
                ..
            })
        ));
        assert_eq!(fs::read(&primary).unwrap(), b"old-primary");
        assert_eq!(fs::read(&backup).unwrap(), b"old-primary");
        assert_no_temps(directory.path());
    }

    #[test]
    fn absent_backup_is_created_before_new_primary_is_published() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("peer-cache.json");
        let backup = backup_path(&primary);
        fs::write(&primary, b"old-primary").unwrap();

        publish_sync(&primary, b"new-primary", &ProductionIo).unwrap();
        assert_eq!(fs::read(primary).unwrap(), b"new-primary");
        assert_eq!(fs::read(backup).unwrap(), b"old-primary");
        assert_no_temps(directory.path());
    }

    #[test]
    fn first_publish_without_primary_leaves_existing_backup_unchanged() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("peer-cache.json");
        let backup = backup_path(&primary);
        fs::write(&backup, b"retained-backup").unwrap();

        publish_sync(&primary, b"first-primary", &ProductionIo).unwrap();
        assert_eq!(fs::read(primary).unwrap(), b"first-primary");
        assert_eq!(fs::read(backup).unwrap(), b"retained-backup");
        assert_no_temps(directory.path());
    }

    #[test]
    fn primary_parent_sync_failure_is_typed_ambiguous_after_complete_replace() {
        let (directory, primary, backup) = setup_files();
        let result = publish_sync(
            &primary,
            b"new-primary",
            &TestSeam::new(IoCheckpoint::BeforePrimaryParentSync, TestAction::Fail),
        );
        assert!(matches!(
            result,
            Err(PeerCacheBackupIoError::Ambiguous {
                stage: "primary_replace_parent_sync",
                ..
            })
        ));
        assert_eq!(fs::read(&primary).unwrap(), b"new-primary");
        assert_eq!(fs::read(&backup).unwrap(), b"old-primary");
        assert_no_temps(directory.path());
    }

    #[test]
    fn create_new_collision_retries_without_truncating_existing_temp() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("peer-cache.json");
        let sequence = AtomicU64::new(9_000);
        let temp = directory.path().join(format!(
            "{TEMP_PREFIX}-backup-{}-9000.tmp",
            std::process::id()
        ));
        fs::write(&temp, b"owned-by-another-transaction").unwrap();
        let created = create_unique_temp_with_sequence(&primary, "backup", &sequence).unwrap();
        assert_ne!(created.path(), temp);
        assert_eq!(fs::read(&temp).unwrap(), b"owned-by-another-transaction");
        drop(created);
        assert_eq!(fs::read(temp).unwrap(), b"owned-by-another-transaction");
    }

    #[test]
    fn oversized_old_primary_is_rejected_without_backup_or_primary_mutation() {
        let (directory, primary, backup) = setup_files();
        let file = OpenOptions::new().write(true).open(&primary).unwrap();
        file.set_len(PEER_CACHE_BACKUP_MAX_BYTES as u64 + 1)
            .unwrap();
        drop(file);
        let old_backup = fs::read(&backup).unwrap();

        let result = publish_sync(&primary, b"new-primary", &ProductionIo);
        assert!(matches!(
            result,
            Err(PeerCacheBackupIoError::OldPrimaryTooLarge { .. })
        ));
        assert_eq!(fs::metadata(&primary).unwrap().len(), 8 * 1024 * 1024 + 1);
        assert_eq!(fs::read(backup).unwrap(), old_backup);
        assert_no_temps(directory.path());
    }

    #[test]
    fn maximum_sized_old_primary_is_accepted_without_unbounded_growth() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("peer-cache.json");
        let old_primary = vec![0x41; PEER_CACHE_BACKUP_MAX_BYTES];
        fs::write(&primary, &old_primary).unwrap();

        publish_sync(&primary, b"new-primary", &ProductionIo).unwrap();
        assert_eq!(
            fs::metadata(backup_path(&primary)).unwrap().len(),
            8 * 1024 * 1024
        );
        assert_eq!(fs::read(primary).unwrap(), b"new-primary");
    }

    #[test]
    fn oversized_new_snapshot_has_zero_filesystem_effect() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory
            .path()
            .join("missing-parent")
            .join("peer-cache.json");
        let snapshot = vec![0u8; PEER_CACHE_BACKUP_MAX_BYTES + 1];
        let result = publish_sync(&primary, &snapshot, &ProductionIo);
        assert!(matches!(
            result,
            Err(PeerCacheBackupIoError::SnapshotTooLarge { .. })
        ));
        assert!(!primary.parent().unwrap().exists());
    }

    #[test]
    fn old_primary_growth_and_truncation_fail_closed() {
        for action in [TestAction::GrowPrimary, TestAction::TruncatePrimary] {
            let (directory, primary, backup) = setup_files();
            let result = publish_sync(
                &primary,
                b"new-primary",
                &TestSeam::new(IoCheckpoint::AfterOldPrimaryMetadata, action),
            );
            assert!(matches!(
                result,
                Err(PeerCacheBackupIoError::OldPrimaryChanged)
            ));
            assert_eq!(fs::read(backup).unwrap(), b"older-backup");
            assert_ne!(fs::read(primary).unwrap(), b"new-primary");
            assert_no_temps(directory.path());
        }
    }

    #[test]
    fn backup_is_always_one_complete_generation_at_failure_boundaries() {
        for checkpoint in [
            IoCheckpoint::BeforeBackupReplace,
            IoCheckpoint::BeforeBackupParentSync,
            IoCheckpoint::BeforePrimaryReplace,
        ] {
            let (directory, primary, backup) = setup_files();
            let _ = publish_sync(
                &primary,
                b"new-primary",
                &TestSeam::new(checkpoint, TestAction::Fail),
            );
            let bytes = fs::read(&backup).unwrap();
            assert!(bytes == b"older-backup" || bytes == b"old-primary");
            assert_no_temps(directory.path());
        }
    }

    #[cfg(unix)]
    #[test]
    fn unix_atomic_replace_replaces_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.tmp");
        let destination = directory.path().join("destination");
        fs::write(&source, b"new").unwrap();
        fs::write(&destination, b"old").unwrap();
        atomic_replace(&source, &destination).unwrap();
        sync_parent_dir(&destination).unwrap();
        assert_eq!(fs::read(destination).unwrap(), b"new");
        assert!(!source.exists());
    }

    #[cfg(windows)]
    #[test]
    fn windows_atomic_replace_replaces_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.tmp");
        let destination = directory.path().join("destination");
        fs::write(&source, b"new").unwrap();
        fs::write(&destination, b"old").unwrap();
        atomic_replace(&source, &destination).unwrap();
        assert_eq!(fs::read(destination).unwrap(), b"new");
        assert!(!source.exists());
    }

    #[tokio::test]
    async fn async_boundary_publishes_complete_primary_and_backup() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("peer-cache.json");
        fs::write(&primary, b"old-primary").unwrap();
        publish_peer_cache_snapshot(primary.clone(), b"new-primary".to_vec())
            .await
            .unwrap();
        assert_eq!(fs::read(&primary).unwrap(), b"new-primary");
        assert_eq!(fs::read(backup_path(&primary)).unwrap(), b"old-primary");
    }
}
