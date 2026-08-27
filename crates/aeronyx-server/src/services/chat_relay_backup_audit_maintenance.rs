// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_maintenance.rs
// ============================================
// Version: 1.0.0-BackupAuditMaintenanceCoordinator
//
// Creation Reason:
//   [CHAT-BACKUP-AUDIT-MAINTENANCE-DOMAIN 2026-08-28 by Codex] Compose audit
//   verification, crash recovery, bounded rotation, checkpoint publication,
//   and durable append outside the oversized relay orchestration service.
//
// Main Functionality:
//   - Builds one consistent production audit policy composition.
//   - Verifies the complete private audit chain before mutation.
//   - Completes an authenticated interrupted rotation before appending.
//   - Rotates a bounded active segment through checkpoint-first publication.
//   - Encodes, bounds, appends, and fsyncs one authenticated audit record.
//
// Dependencies:
//   - `chat_relay_backup_audit*.rs` own records, checkpoints, policies, and I/O.
//   - `chat_relay_backup_io.rs` owns private no-follow filesystem operations.
//   - `chat_relay_error.rs` provides stable path-free storage failures.
//   - The relay service retains cross-process lock ownership and API wrappers.
//
// Main Logical Flow:
//   1. Clean bounded abandoned checkpoint temporaries.
//   2. Authenticate immutable segments, checkpoints, and the active tail.
//   3. Complete any checkpoint-first crash window under the caller-held lock.
//   4. Build and bound the next authenticated maintenance record.
//   5. Rotate when required, then append and fsync the active audit segment.
//
// Important Note for Next Developer:
//   - The caller must hold the cross-process maintenance lock for every write.
//   - Preserve checkpoint-first publication and deterministic crash recovery.
//   - Never log paths, private MACs, operation ids, or artifact identities.
//   - All capacity, sequence, corruption, and ownership ambiguity fails closed.
//   - Keep the v1 record/checkpoint signing order byte-for-byte compatible.
//
// Last Modified:
//   v1.0.0-BackupAuditMaintenanceCoordinator - Initial use-case composition
// ============================================

#[cfg(test)]
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::chat_relay_backup_audit::{
    BackupAuditPhase, BackupAuditRecordAuthenticator, ChatRelayBackupMaintenanceAuditCounts,
    HmacBackupAuditRecordAuthenticator,
};
use super::chat_relay_backup_audit_chain::{
    map_backup_audit_checkpoint_error, map_backup_audit_record_error,
    map_backup_audit_verification_error, AuthenticatedBackupAuditChainVerifier,
    BackupAuditChainLimits, BackupAuditChainVerifier, ChatRelayBackupAuditChainVerification,
    LocalBackupAuditChainVerifier,
};
use super::chat_relay_backup_audit_checkpoint::{
    BackupAuditCheckpointAuthenticator, BackupAuditCheckpointState,
    HmacBackupAuditCheckpointAuthenticator,
};
use super::chat_relay_backup_audit_io::{
    BackupAuditIo, ChatRelayBackupAuditPendingRotation, LocalBackupAuditIo, BACKUP_AUDIT_FILE_NAME,
};
#[cfg(test)]
use super::chat_relay_backup_audit_rotation::ChatRelayBackupAuditSegmentRange;
use super::chat_relay_backup_audit_rotation::{
    BackupAuditRotationError, BackupAuditRotationLimits, BackupAuditRotationPolicy,
    BackupAuditRotationState, BoundedBackupAuditRotationPolicy,
};
use super::chat_relay_backup_audit_verification::{
    BackupAuditVerificationLimits, BackupAuditVerificationPolicy,
    BoundedBackupAuditVerificationPolicy, ChatRelayBackupAuditVerificationState,
};
use super::chat_relay_backup_io::{
    backup_io_error, BackupFilesystem, LocalBackupFilesystem, PrivateBackupControlFileMode,
};
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Fixed resource ceilings shared by verification, rotation, and append.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BackupAuditMaintenanceLimits {
    pub(super) max_record_bytes: usize,
    pub(super) max_records_per_segment: u64,
    pub(super) max_segment_bytes: u64,
    pub(super) max_segments: u64,
    pub(super) max_total_bytes: u64,
}

/// Production backup-audit verification, recovery, rotation, and append flow.
pub(super) struct BackupAuditMaintenance {
    filesystem: LocalBackupFilesystem,
    audit_io: LocalBackupAuditIo<LocalBackupFilesystem>,
    chain_verifier: LocalBackupAuditChainVerifier,
    verification_policy: BoundedBackupAuditVerificationPolicy,
    rotation_policy: BoundedBackupAuditRotationPolicy,
    limits: BackupAuditMaintenanceLimits,
}

impl BackupAuditMaintenance {
    /// Composes every policy from one immutable resource-limit contract.
    pub(super) fn new(limits: BackupAuditMaintenanceLimits) -> Self {
        let filesystem = LocalBackupFilesystem;
        let audit_io = LocalBackupAuditIo::new(filesystem);
        let verification_policy =
            BoundedBackupAuditVerificationPolicy::new(BackupAuditVerificationLimits {
                max_records_per_segment: limits.max_records_per_segment,
                max_bytes_per_segment: limits.max_segment_bytes,
                max_total_bytes: limits.max_total_bytes,
            });
        let rotation_policy = BoundedBackupAuditRotationPolicy::new(BackupAuditRotationLimits {
            max_records_per_segment: limits.max_records_per_segment,
            max_bytes_per_segment: limits.max_segment_bytes,
            max_segments: limits.max_segments,
        });
        let chain_verifier = AuthenticatedBackupAuditChainVerifier::new(
            filesystem,
            audit_io,
            verification_policy,
            rotation_policy,
            BackupAuditChainLimits {
                max_record_bytes: limits.max_record_bytes,
                max_segment_bytes: limits.max_segment_bytes,
            },
        );
        Self {
            filesystem,
            audit_io,
            chain_verifier,
            verification_policy,
            rotation_policy,
            limits,
        }
    }

    /// Returns the canonical immutable segment name for a validated range.
    #[cfg(test)]
    pub(super) fn segment_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String {
        self.audit_io.segment_file_name(range)
    }

    /// Returns the canonical checkpoint name for compatibility fixtures.
    #[cfg(test)]
    pub(super) fn checkpoint_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String {
        self.audit_io.checkpoint_file_name(range)
    }

    /// Verifies one standalone active log for compatibility fixtures.
    #[cfg(test)]
    pub(super) fn verify_log(
        &self,
        file: &mut File,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationState> {
        self.chain_verifier.verify_log(file, node_secret)
    }

    /// Authenticates all immutable checkpoints/segments and the active tail.
    pub(super) fn verify_chain(
        &self,
        parent: &Path,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditChainVerification> {
        self.chain_verifier.verify_chain(parent, node_secret)
    }

    /// Returns whether the next record requires bounded active-tail rotation.
    pub(super) fn segment_needs_rotation(
        &self,
        active_record_count: u64,
        active_bytes: u64,
        next_record_bytes: usize,
    ) -> ChatRelayResult<bool> {
        self.rotation_policy
            .should_rotate(active_record_count, active_bytes, next_record_bytes)
            .map_err(Self::map_rotation_error)
    }

    /// Publishes one authenticated immutable checkpoint and segment.
    pub(super) fn rotate_segment(
        &self,
        parent: &Path,
        node_secret: &[u8; 32],
        state: &ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<()> {
        let receipt = state.receipt();
        let rotation_state = BackupAuditRotationState {
            active_record_count: receipt.active_record_count,
            archived_record_count: receipt.archived_record_count,
            record_count: receipt.record_count,
            checkpoint_count: receipt.checkpoint_count,
        };
        self.rotation_policy
            .validate_admission(rotation_state)
            .map_err(Self::map_rotation_error)?;
        let active_path = parent.join(BACKUP_AUDIT_FILE_NAME);
        let Some(mut active) = self.filesystem.open_existing_control_file(&active_path)? else {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "active relay backup maintenance audit segment is missing",
            ));
        };
        active.sync_all().map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_FSYNC,
                "unable to sync active relay backup maintenance audit segment",
            )
        })?;
        let rotation_plan = self
            .rotation_policy
            .plan_rotation(rotation_state)
            .map_err(Self::map_rotation_error)?;
        let range = rotation_plan.range;
        let (segment_bytes, segment_sha256) = self.audit_io.hash_segment(&mut active)?;
        let checkpoint = HmacBackupAuditCheckpointAuthenticator
            .build(
                node_secret,
                BackupAuditCheckpointState {
                    checkpoint_index: rotation_plan.checkpoint_index,
                    segment_first_sequence: range.first_sequence,
                    segment_last_sequence: range.last_sequence,
                    segment_bytes,
                    segment_sha256,
                    cumulative_verified_bytes: receipt.verified_bytes,
                    cumulative_last_recorded_at: receipt.last_recorded_at,
                    cumulative_dry_run_count: receipt.dry_run_count,
                    cumulative_planned_count: receipt.planned_count,
                    cumulative_completed_count: receipt.completed_count,
                    cumulative_failed_count: receipt.failed_count,
                    head_mac: state.head_mac().to_string(),
                    previous_checkpoint_mac: state.checkpoint_head_mac().to_string(),
                },
            )
            .map_err(map_backup_audit_checkpoint_error)?;
        drop(active);
        self.audit_io
            .publish_checkpoint(parent, range, &checkpoint)?;
        self.audit_io.complete_pending_rotation(
            parent,
            ChatRelayBackupAuditPendingRotation::PublishSegment {
                active_path,
                segment_path: parent.join(self.audit_io.segment_file_name(range)),
            },
        )
    }

    /// Recovers, rotates if needed, and durably appends one audit record.
    pub(super) fn append(
        &self,
        backup_directory: &Path,
        node_secret: &[u8; 32],
        phase: BackupAuditPhase,
        timestamp: u64,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> ChatRelayResult<()> {
        let parent = backup_directory.parent().ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private audit parent",
            )
        })?;
        let audit_path = parent.join(BACKUP_AUDIT_FILE_NAME);
        self.audit_io.cleanup_checkpoint_temporaries(parent)?;
        let mut chain = self.verify_chain(parent, node_secret)?;
        if let Some(pending) = chain.pending_rotation.take() {
            // [CHAT-BACKUP-AUDIT-MAINTENANCE-DOMAIN 2026-08-28 by Codex]
            // Checkpoint-first publication is completed under the caller-held
            // cross-process lock before a new authenticated record is built.
            self.audit_io.complete_pending_rotation(parent, pending)?;
            self.verification_policy
                .mark_rotation_recovered(&mut chain.state);
        }
        let verification = chain.state;
        let next_sequence = verification
            .next_sequence()
            .map_err(map_backup_audit_verification_error)?;
        let record = HmacBackupAuditRecordAuthenticator
            .build(
                node_secret,
                next_sequence,
                verification.head_mac().to_string(),
                phase,
                timestamp,
                counts,
            )
            .map_err(map_backup_audit_record_error)?;
        let mut encoded = serde_json::to_vec(&record).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_FORMAT,
                "unable to encode relay backup maintenance audit",
            )
        })?;
        encoded.push(b'\n');
        if encoded.len() > self.limits.max_record_bytes {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit capacity exhausted",
            ));
        }
        if verification
            .receipt()
            .verified_bytes
            .checked_add(encoded.len() as u64)
            .map_or(true, |bytes| bytes > self.limits.max_total_bytes)
        {
            return Err(backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit chain capacity exhausted",
            ));
        }
        let mut file = self
            .filesystem
            .open_control_file(&audit_path, PrivateBackupControlFileMode::Append)?;
        let current_len = file
            .metadata()
            .map_err(|_| {
                backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect active relay backup maintenance audit",
                )
            })?
            .len();
        if self.segment_needs_rotation(
            verification.receipt().active_record_count,
            current_len,
            encoded.len(),
        )? {
            drop(file);
            self.rotate_segment(parent, node_secret, &verification)?;
            file = self
                .filesystem
                .open_control_file(&audit_path, PrivateBackupControlFileMode::Append)?;
        }
        file.write_all(&encoded).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_WRITE,
                "unable to append relay backup maintenance audit",
            )
        })?;
        file.sync_all().map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_FSYNC,
                "unable to durably sync relay backup maintenance audit",
            )
        })
    }

    fn map_rotation_error(error: BackupAuditRotationError) -> ChatRelayError {
        match error {
            BackupAuditRotationError::EmptyActiveSegment => backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "empty relay backup maintenance audit segment cannot be rotated",
            ),
            BackupAuditRotationError::SegmentLimitReached => backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit segment limit reached",
            ),
            BackupAuditRotationError::SequenceOverflow => backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit sequence overflow",
            ),
            BackupAuditRotationError::InvalidSegmentRange => backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit segment range is invalid",
            ),
            BackupAuditRotationError::CheckpointIndexOverflow => backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit checkpoint index overflow",
            ),
            BackupAuditRotationError::RecordSizeOverflow => backup_io_error(
                rusqlite::ffi::SQLITE_TOOBIG,
                "relay backup maintenance audit record size exceeds platform limits",
            ),
            BackupAuditRotationError::ByteCountOverflow => backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit byte count overflow",
            ),
        }
    }
}
