// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_chain.rs
// ============================================
// Version: 1.1.0-MaintenanceCoordinatorComposition
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-AUDIT-CHAIN-DOMAIN 2026-08-27 by Codex] Extract the
//   authenticated multi-segment audit-chain verifier from the relay service.
//
// Modification Reason:
//   [CHAT-BACKUP-AUDIT-MAINTENANCE-DOMAIN 2026-08-28 by Codex] Documented the
//   dedicated coordinator that consumes authenticated state and recovery work.
//
// Main Functionality:
//   - Verifies bounded newline-delimited HMAC audit records.
//   - Authenticates immutable segment checkpoints and cumulative accounting.
//   - Classifies both supported active-tail crash-recovery windows.
//   - Returns a typed verified state plus an explicit pending rotation action.
//
// Dependencies:
//   - `chat_relay_backup_audit_io` supplies bounded artifact I/O and naming.
//   - `chat_relay_backup_io` supplies private no-follow control-file access.
//   - Audit verification/rotation modules supply pure transition policies.
//   - Audit record/checkpoint modules supply HMAC authenticators.
//   - Audit maintenance owns recovery execution, rotation, and durable append.
//
// Main Logical Flow:
//   1. Catalog canonical immutable segments and checkpoints.
//   2. Verify every record and checkpoint in monotonic range order.
//   3. Compare the active artifact with the latest immutable fingerprint.
//   4. Finalize aggregate accounting and return explicit recovery work.
//
// Important Note for Next Developer:
//   - Never expose record MACs, node secrets, paths, or checkpoint MACs.
//   - Every size, sequence, and aggregate transition must remain bounded.
//   - Files changing during verification must fail closed as corruption.
//   - Recovery actions may be returned only after full authentication.
//
// Last Modified:
//   v1.1.0-MaintenanceCoordinatorComposition - Documented use-case ownership
//   v1.0.0-BackupAuditChainDomain - Initial composed chain verifier extraction
// ============================================

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::services::chat_relay_backup_audit::{
    BackupAuditRecordAuthenticator, BackupAuditRecordError, ChatRelayBackupMaintenanceAuditRecord,
    HmacBackupAuditRecordAuthenticator,
};
use crate::services::chat_relay_backup_audit_checkpoint::{
    BackupAuditCheckpointAuthenticator, BackupAuditCheckpointError, ChatRelayBackupAuditCheckpoint,
    HmacBackupAuditCheckpointAuthenticator,
};
use crate::services::chat_relay_backup_audit_io::{
    BackupAuditIo, ChatRelayBackupAuditPendingRotation, LocalBackupAuditIo, BACKUP_AUDIT_FILE_NAME,
};
use crate::services::chat_relay_backup_audit_rotation::BoundedBackupAuditRotationPolicy;
use crate::services::chat_relay_backup_audit_rotation::{
    BackupAuditActiveTailDisposition, BackupAuditRotationPolicy, BackupAuditSegmentFingerprint,
    ChatRelayBackupAuditSegmentRange,
};
use crate::services::chat_relay_backup_audit_verification::{
    BackupAuditCheckpointAdmission, BackupAuditRecordAdmission, BackupAuditSegmentBaseline,
    BackupAuditVerificationError, BackupAuditVerificationPolicy,
    BoundedBackupAuditVerificationPolicy, ChatRelayBackupAuditVerificationState,
};
use crate::services::chat_relay_backup_io::{
    backup_io_error, BackupFilesystem, LocalBackupFilesystem,
};
use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Resource ceiling owned by the encoded-record reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BackupAuditChainLimits {
    pub(super) max_record_bytes: usize,
    pub(super) max_segment_bytes: u64,
}

/// Fully authenticated chain state and optional crash-recovery action.
#[derive(Debug)]
pub(super) struct ChatRelayBackupAuditChainVerification {
    pub(super) state: ChatRelayBackupAuditVerificationState,
    pub(super) pending_rotation: Option<ChatRelayBackupAuditPendingRotation>,
}

/// Capability boundary for verifying a complete host-local audit chain.
pub(super) trait BackupAuditChainVerifier {
    fn verify_chain(
        &self,
        parent: &Path,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditChainVerification>;
}

/// Authenticated verifier composed from host I/O and pure transition policies.
#[derive(Debug, Clone, Copy)]
pub(super) struct AuthenticatedBackupAuditChainVerifier<F, I, V, R> {
    filesystem: F,
    audit_io: I,
    verification_policy: V,
    rotation_policy: R,
    limits: BackupAuditChainLimits,
}

/// Production composition used by the audit maintenance coordinator.
pub(super) type LocalBackupAuditChainVerifier = AuthenticatedBackupAuditChainVerifier<
    LocalBackupFilesystem,
    LocalBackupAuditIo<LocalBackupFilesystem>,
    BoundedBackupAuditVerificationPolicy,
    BoundedBackupAuditRotationPolicy,
>;

impl<F, I, V, R> AuthenticatedBackupAuditChainVerifier<F, I, V, R> {
    pub(super) const fn new(
        filesystem: F,
        audit_io: I,
        verification_policy: V,
        rotation_policy: R,
        limits: BackupAuditChainLimits,
    ) -> Self {
        Self {
            filesystem,
            audit_io,
            verification_policy,
            rotation_policy,
            limits,
        }
    }
}

impl<F, I, V, R> AuthenticatedBackupAuditChainVerifier<F, I, V, R>
where
    F: BackupFilesystem,
    I: BackupAuditIo,
    V: BackupAuditVerificationPolicy,
    R: BackupAuditRotationPolicy,
{
    /// Verifies one standalone active audit log for compatibility tests.
    #[cfg(test)]
    pub(super) fn verify_log(
        &self,
        file: &mut File,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationState> {
        self.verify_segment(file, node_secret, self.verification_policy.empty_state())
    }

    fn read_next_record(
        &self,
        reader: &mut BufReader<File>,
        line: &mut String,
        state: &ChatRelayBackupAuditVerificationState,
        baseline: BackupAuditSegmentBaseline,
    ) -> ChatRelayResult<
        Option<(
            ChatRelayBackupMaintenanceAuditRecord,
            BackupAuditRecordAdmission,
        )>,
    > {
        line.clear();
        // Metadata is not a read bound because a concurrent writer can grow
        // the file after inspection. Bound each allocation independently.
        let read_limit = self.limits.max_record_bytes.checked_add(1).ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit record size limit overflow",
            )
        })?;
        let bytes = Read::by_ref(reader)
            .take(read_limit as u64)
            .read_line(line)
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to read relay backup maintenance audit",
                )
            })?;
        if bytes == 0 {
            return Ok(None);
        }
        if bytes > self.limits.max_record_bytes || !line.ends_with('\n') {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit record is malformed",
            ));
        }
        let admission = self
            .verification_policy
            .prepare_record(state, baseline, bytes)
            .map_err(map_backup_audit_verification_error)?;
        let record = serde_json::from_str(line.trim_end_matches('\n')).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit record is malformed",
            )
        })?;
        Ok(Some((record, admission)))
    }

    fn verify_segment(
        &self,
        file: &mut File,
        node_secret: &[u8; 32],
        mut state: ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationState> {
        let metadata = file.metadata().map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to inspect relay backup maintenance audit",
            )
        })?;
        if metadata.len() > self.limits.max_segment_bytes {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit exceeds its bounded size",
            ));
        }
        let initial_len = metadata.len();
        let baseline = self.verification_policy.begin_segment(&state);
        let initial_verified_bytes = state.receipt().verified_bytes;
        file.seek(SeekFrom::Start(0)).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to read relay backup maintenance audit",
            )
        })?;

        let mut reader = BufReader::new(file.try_clone().map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "unable to read relay backup maintenance audit",
            )
        })?);
        let mut line = String::new();
        while let Some((record, admission)) =
            self.read_next_record(&mut reader, &mut line, &state, baseline)?
        {
            let phase = HmacBackupAuditRecordAuthenticator
                .authenticate(
                    node_secret,
                    &record,
                    admission.expected_sequence(),
                    admission.expected_previous_mac(),
                )
                .map_err(map_backup_audit_record_error)?;
            self.verification_policy
                .commit_record(&mut state, admission, phase, record.timestamp, record.mac)
                .map_err(map_backup_audit_verification_error)?;
        }
        let final_len = file
            .metadata()
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to re-inspect relay backup maintenance audit",
                )
            })?
            .len();
        if final_len != initial_len
            || state
                .receipt()
                .verified_bytes
                .checked_sub(initial_verified_bytes)
                != Some(initial_len)
        {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit changed during verification",
            ));
        }
        Ok(state)
    }

    fn apply_checkpoint(
        &self,
        node_secret: &[u8; 32],
        range: ChatRelayBackupAuditSegmentRange,
        segment_bytes: u64,
        segment_sha256: &str,
        previous_record_count: u64,
        checkpoint: ChatRelayBackupAuditCheckpoint,
        state: &mut ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<()> {
        let admission: BackupAuditCheckpointAdmission = self
            .verification_policy
            .prepare_checkpoint(
                state,
                range,
                segment_bytes,
                segment_sha256.to_string(),
                previous_record_count,
            )
            .map_err(map_backup_audit_verification_error)?;
        HmacBackupAuditCheckpointAuthenticator
            .authenticate(
                node_secret,
                &checkpoint,
                admission.expected(),
                admission.expected_record_count(),
            )
            .map_err(map_backup_audit_checkpoint_error)?;
        self.verification_policy
            .commit_checkpoint(state, admission, checkpoint.checkpoint_mac)
            .map_err(map_backup_audit_verification_error)
    }

    fn verify_active_tail(
        &self,
        active_path: &Path,
        latest_segment_fingerprint: Option<&BackupAuditSegmentFingerprint>,
        node_secret: &[u8; 32],
        state: ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<(
        ChatRelayBackupAuditVerificationState,
        Option<ChatRelayBackupAuditPendingRotation>,
    )> {
        let Some(mut active) = self.filesystem.open_existing_control_file(active_path)? else {
            return Ok((state, None));
        };
        let active_len = active
            .metadata()
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect active relay backup maintenance audit",
                )
            })?
            .len();
        let active_fingerprint = if active_len > 0 {
            let (bytes, sha256) = self.audit_io.hash_segment(&mut active)?;
            Some(BackupAuditSegmentFingerprint { bytes, sha256 })
        } else {
            None
        };
        match self
            .rotation_policy
            .classify_active_tail(active_fingerprint.as_ref(), latest_segment_fingerprint)
        {
            BackupAuditActiveTailDisposition::RetirePublishedDuplicate => {
                return Ok((
                    state,
                    Some(ChatRelayBackupAuditPendingRotation::RemoveDuplicateActive {
                        active_path: active_path.to_path_buf(),
                    }),
                ));
            }
            BackupAuditActiveTailDisposition::VerifyActiveTail => {}
        }
        Ok((self.verify_segment(&mut active, node_secret, state)?, None))
    }
}

impl<F, I, V, R> BackupAuditChainVerifier for AuthenticatedBackupAuditChainVerifier<F, I, V, R>
where
    F: BackupFilesystem,
    I: BackupAuditIo,
    V: BackupAuditVerificationPolicy,
    R: BackupAuditRotationPolicy,
{
    fn verify_chain(
        &self,
        parent: &Path,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditChainVerification> {
        let active_path = parent.join(BACKUP_AUDIT_FILE_NAME);
        let segments = self.audit_io.collect_segment_files(parent)?;
        let segment_count = segments.len();
        let mut state = self.verification_policy.empty_state();
        let mut pending_rotation = None;
        let mut latest_segment_fingerprint: Option<BackupAuditSegmentFingerprint> = None;

        for (index, (range, files)) in segments.into_iter().enumerate() {
            let checkpoint_path = parent.join(files.checkpoint_file_name.ok_or_else(|| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "relay backup maintenance audit segment checkpoint is missing",
                )
            })?);
            let previous_record_count = state.receipt().record_count;
            let (mut segment, uses_active) =
                if let Some(segment_file_name) = files.segment_file_name {
                    let segment_path = parent.join(segment_file_name);
                    let segment = self
                        .filesystem
                        .open_existing_control_file(&segment_path)?
                        .ok_or_else(|| {
                            backup_io_error(
                                rusqlite::ffi::SQLITE_CORRUPT,
                                "relay backup maintenance audit segment is missing",
                            )
                        })?;
                    (segment, false)
                } else {
                    if index + 1 != segment_count || pending_rotation.is_some() {
                        return Err(backup_io_error(
                            rusqlite::ffi::SQLITE_CORRUPT,
                            "relay backup maintenance audit segment publication is incomplete",
                        ));
                    }
                    let segment = self
                        .filesystem
                        .open_existing_control_file(&active_path)?
                        .ok_or_else(|| {
                            backup_io_error(
                                rusqlite::ffi::SQLITE_CORRUPT,
                                "relay backup maintenance audit pending segment is missing",
                            )
                        })?;
                    pending_rotation = Some(ChatRelayBackupAuditPendingRotation::PublishSegment {
                        active_path: active_path.clone(),
                        segment_path: parent.join(self.audit_io.segment_file_name(range)),
                    });
                    (segment, true)
                };
            state = self.verify_segment(&mut segment, node_secret, state)?;
            let (segment_bytes, segment_sha256) = self.audit_io.hash_segment(&mut segment)?;
            let checkpoint = self.audit_io.read_checkpoint(&checkpoint_path)?;
            self.apply_checkpoint(
                node_secret,
                range,
                segment_bytes,
                &segment_sha256,
                previous_record_count,
                checkpoint,
                &mut state,
            )?;
            latest_segment_fingerprint = Some(BackupAuditSegmentFingerprint {
                bytes: segment_bytes,
                sha256: segment_sha256,
            });
            if uses_active {
                break;
            }
        }

        if pending_rotation.is_none() {
            (state, pending_rotation) = self.verify_active_tail(
                &active_path,
                latest_segment_fingerprint.as_ref(),
                node_secret,
                state,
            )?;
        }

        self.verification_policy
            .finalize_chain(&mut state, pending_rotation.is_some())
            .map_err(map_backup_audit_verification_error)?;
        Ok(ChatRelayBackupAuditChainVerification {
            state,
            pending_rotation,
        })
    }
}

pub(super) fn map_backup_audit_record_error(error: BackupAuditRecordError) -> ChatRelayError {
    match error {
        BackupAuditRecordError::CountOutOfRange => backup_io_error(
            rusqlite::ffi::SQLITE_TOOBIG,
            "relay backup maintenance count exceeds audit format",
        ),
        BackupAuditRecordError::EncodingFailed => backup_io_error(
            rusqlite::ffi::SQLITE_FORMAT,
            "unable to encode relay backup maintenance audit",
        ),
        BackupAuditRecordError::AuthenticatorInitFailed => backup_io_error(
            rusqlite::ffi::SQLITE_AUTH,
            "unable to initialize relay backup maintenance audit",
        ),
        BackupAuditRecordError::InvalidRecord => backup_io_error(
            rusqlite::ffi::SQLITE_CORRUPT,
            "relay backup maintenance audit verification failed",
        ),
    }
}

pub(super) fn map_backup_audit_checkpoint_error(
    error: BackupAuditCheckpointError,
) -> ChatRelayError {
    match error {
        BackupAuditCheckpointError::EncodingFailed => backup_io_error(
            rusqlite::ffi::SQLITE_FORMAT,
            "unable to encode relay backup maintenance audit checkpoint",
        ),
        BackupAuditCheckpointError::AuthenticatorInitFailed => backup_io_error(
            rusqlite::ffi::SQLITE_AUTH,
            "unable to initialize relay backup maintenance audit checkpoint",
        ),
        BackupAuditCheckpointError::InvalidCheckpoint => backup_io_error(
            rusqlite::ffi::SQLITE_CORRUPT,
            "relay backup maintenance audit checkpoint verification failed",
        ),
    }
}

pub(super) fn map_backup_audit_verification_error(
    error: BackupAuditVerificationError,
) -> ChatRelayError {
    match error {
        BackupAuditVerificationError::RecordLimitReached => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit record limit reached",
        ),
        BackupAuditVerificationError::RecordSizeOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit byte count overflow",
        ),
        BackupAuditVerificationError::SegmentSizeExceeded => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit exceeds its bounded size",
        ),
        BackupAuditVerificationError::TotalSizeExceeded => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit chain exceeds its bounded size",
        ),
        BackupAuditVerificationError::ByteCountOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit byte count overflow",
        ),
        BackupAuditVerificationError::RecordCountOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit sequence overflow",
        ),
        BackupAuditVerificationError::PhaseCountOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit phase count overflow",
        ),
        BackupAuditVerificationError::CheckpointIndexOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit checkpoint index overflow",
        ),
        BackupAuditVerificationError::InvalidSegmentRange => backup_io_error(
            rusqlite::ffi::SQLITE_CORRUPT,
            "relay backup maintenance audit checkpoint verification failed",
        ),
        BackupAuditVerificationError::ArchivedByteCountOverflow => backup_io_error(
            rusqlite::ffi::SQLITE_FULL,
            "relay backup maintenance audit archived byte count overflow",
        ),
        BackupAuditVerificationError::InvalidRecordAccounting => backup_io_error(
            rusqlite::ffi::SQLITE_CORRUPT,
            "relay backup maintenance audit record accounting is invalid",
        ),
        BackupAuditVerificationError::StaleRecordAdmission
        | BackupAuditVerificationError::StaleCheckpointAdmission => backup_io_error(
            rusqlite::ffi::SQLITE_CORRUPT,
            "relay backup maintenance audit verification state changed unexpectedly",
        ),
    }
}
