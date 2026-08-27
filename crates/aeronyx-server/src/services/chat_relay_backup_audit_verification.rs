// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_verification.rs
// ============================================
// Version: 1.0.0-VerificationDomain
//
// Creation Reason:
//   [CHAT-RELAY-AUDIT-VERIFICATION-DOMAIN 2026-08-27 by Codex] Isolate the
//   bounded verification state machine from private filesystem orchestration.
//
// Main Functionality:
//   - Models aggregate verification state and its public path-free receipt.
//   - Prepares and atomically commits authenticated record admissions.
//   - Prepares and atomically commits authenticated checkpoint admissions.
//   - Finalizes active/archive accounting and rotation-recovery state.
//   - Rejects stale, overflowing, discontinuous, or unbounded transitions.
//
// Dependencies:
//   - `chat_relay_backup_audit.rs` owns authenticated record/phase contracts.
//   - `chat_relay_backup_audit_checkpoint.rs` owns checkpoint authentication.
//   - `chat_relay_backup_audit_rotation.rs` owns validated sequence ranges.
//   - `chat_relay.rs` owns paths, locks, bounded reads, HMAC calls, and fsync.
//
// Main Logical Flow:
//   1. Capture an immutable baseline before reading one segment.
//   2. Prepare a bounded record admission without mutating verification state.
//   3. Commit it only after the service authenticates the exact record.
//   4. Prepare a checkpoint expectation from the resulting cumulative state.
//   5. Commit it only after checkpoint authentication, then finalize accounting.
//
// Important Note for Next Developer:
//   - Every error must leave the supplied state unchanged.
//   - Never add paths, file names, identities, payloads, routes, or ciphertext.
//   - Authentication and side effects remain outside this pure policy module.
//   - Preserve the public receipt fields and serialization contract.
//
// Last Modified:
//   v1.0.0-VerificationDomain - Initial bounded trait-based state machine
// ============================================

use serde::Serialize;

use crate::services::chat_relay_backup_audit::BackupAuditPhase;
use crate::services::chat_relay_backup_audit_checkpoint::BackupAuditCheckpointState;
use crate::services::chat_relay_backup_audit_rotation::ChatRelayBackupAuditSegmentRange;

const EMPTY_CHAIN_MAC: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Aggregate result of verifying the private maintenance audit HMAC chain.
///
/// No chain MAC, path, artifact identity, message metadata, wallet identity,
/// or ciphertext may cross this host-local boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ChatRelayBackupAuditVerificationReceipt {
    /// Always true on success; corruption is returned as an error.
    pub verified: bool,
    /// Number of authenticated audit records.
    pub record_count: u64,
    /// Timestamp of the final authenticated record, absent for an empty chain.
    pub last_recorded_at: Option<u64>,
    /// Number of authenticated retention dry-run records.
    pub dry_run_count: u64,
    /// Number of authenticated destructive plans.
    pub planned_count: u64,
    /// Number of authenticated completed destructive operations.
    pub completed_count: u64,
    /// Number of authenticated failed destructive operations.
    pub failed_count: u64,
    /// Exact bytes consumed by authenticated newline-delimited records.
    pub verified_bytes: u64,
    /// Number of authenticated immutable segment checkpoints.
    pub checkpoint_count: u64,
    /// Records covered by immutable segment checkpoints.
    pub archived_record_count: u64,
    /// Records still held by the active append-only segment.
    pub active_record_count: u64,
    /// Bytes covered by immutable segment checkpoints.
    pub archived_bytes: u64,
    /// Whether a crash-safe rotation publication still needs housekeeping.
    pub rotation_pending: bool,
}

/// Immutable bounds applied to every verification transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackupAuditVerificationLimits {
    pub(crate) max_records_per_segment: u64,
    pub(crate) max_bytes_per_segment: u64,
    pub(crate) max_total_bytes: u64,
}

/// Closed failure vocabulary mapped to stable storage errors by the I/O owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditVerificationError {
    RecordLimitReached,
    RecordSizeOverflow,
    SegmentSizeExceeded,
    TotalSizeExceeded,
    ByteCountOverflow,
    RecordCountOverflow,
    PhaseCountOverflow,
    CheckpointIndexOverflow,
    InvalidSegmentRange,
    ArchivedByteCountOverflow,
    InvalidRecordAccounting,
    StaleRecordAdmission,
    StaleCheckpointAdmission,
}

/// Immutable cumulative boundary captured before reading one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackupAuditSegmentBaseline {
    verified_bytes: u64,
    record_count: u64,
}

/// Prepared, non-mutating admission for one exact encoded record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackupAuditRecordAdmission {
    record_count_before: u64,
    verified_bytes_before: u64,
    expected_sequence: u64,
    expected_previous_mac: String,
    verified_bytes_after: u64,
}

impl BackupAuditRecordAdmission {
    pub(crate) const fn expected_sequence(&self) -> u64 {
        self.expected_sequence
    }

    pub(crate) fn expected_previous_mac(&self) -> &str {
        &self.expected_previous_mac
    }
}

/// Prepared expectation committed only after checkpoint authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackupAuditCheckpointAdmission {
    expected: BackupAuditCheckpointState,
    expected_record_count: u64,
}

impl BackupAuditCheckpointAdmission {
    pub(crate) const fn expected(&self) -> &BackupAuditCheckpointState {
        &self.expected
    }

    pub(crate) const fn expected_record_count(&self) -> u64 {
        self.expected_record_count
    }
}

/// Private cumulative state; callers can advance it only through the policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatRelayBackupAuditVerificationState {
    receipt: ChatRelayBackupAuditVerificationReceipt,
    head_mac: String,
    checkpoint_head_mac: String,
}

impl ChatRelayBackupAuditVerificationState {
    pub(crate) const fn receipt(&self) -> &ChatRelayBackupAuditVerificationReceipt {
        &self.receipt
    }

    pub(crate) fn into_receipt(self) -> ChatRelayBackupAuditVerificationReceipt {
        self.receipt
    }

    pub(crate) fn head_mac(&self) -> &str {
        &self.head_mac
    }

    pub(crate) fn checkpoint_head_mac(&self) -> &str {
        &self.checkpoint_head_mac
    }

    pub(crate) fn next_sequence(&self) -> Result<u64, BackupAuditVerificationError> {
        self.receipt
            .record_count
            .checked_add(1)
            .ok_or(BackupAuditVerificationError::RecordCountOverflow)
    }
}

/// Replaceable pure policy for bounded audit verification transitions.
pub(crate) trait BackupAuditVerificationPolicy {
    fn empty_state(&self) -> ChatRelayBackupAuditVerificationState;

    fn begin_segment(
        &self,
        state: &ChatRelayBackupAuditVerificationState,
    ) -> BackupAuditSegmentBaseline;

    fn prepare_record(
        &self,
        state: &ChatRelayBackupAuditVerificationState,
        baseline: BackupAuditSegmentBaseline,
        encoded_bytes: usize,
    ) -> Result<BackupAuditRecordAdmission, BackupAuditVerificationError>;

    fn commit_record(
        &self,
        state: &mut ChatRelayBackupAuditVerificationState,
        admission: BackupAuditRecordAdmission,
        phase: BackupAuditPhase,
        timestamp: u64,
        record_mac: String,
    ) -> Result<(), BackupAuditVerificationError>;

    fn prepare_checkpoint(
        &self,
        state: &ChatRelayBackupAuditVerificationState,
        range: ChatRelayBackupAuditSegmentRange,
        segment_bytes: u64,
        segment_sha256: String,
        previous_record_count: u64,
    ) -> Result<BackupAuditCheckpointAdmission, BackupAuditVerificationError>;

    fn commit_checkpoint(
        &self,
        state: &mut ChatRelayBackupAuditVerificationState,
        admission: BackupAuditCheckpointAdmission,
        checkpoint_mac: String,
    ) -> Result<(), BackupAuditVerificationError>;

    fn finalize_chain(
        &self,
        state: &mut ChatRelayBackupAuditVerificationState,
        rotation_pending: bool,
    ) -> Result<(), BackupAuditVerificationError>;

    fn mark_rotation_recovered(&self, state: &mut ChatRelayBackupAuditVerificationState);
}

/// Production verification policy enforcing fixed per-segment and chain bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BoundedBackupAuditVerificationPolicy {
    limits: BackupAuditVerificationLimits,
}

impl BoundedBackupAuditVerificationPolicy {
    pub(crate) const fn new(limits: BackupAuditVerificationLimits) -> Self {
        Self { limits }
    }

    fn increment_phase(
        receipt: &mut ChatRelayBackupAuditVerificationReceipt,
        phase: BackupAuditPhase,
    ) -> Result<(), BackupAuditVerificationError> {
        let count = match phase {
            BackupAuditPhase::DryRun => &mut receipt.dry_run_count,
            BackupAuditPhase::Planned => &mut receipt.planned_count,
            BackupAuditPhase::Completed => &mut receipt.completed_count,
            BackupAuditPhase::Failed => &mut receipt.failed_count,
        };
        *count = count
            .checked_add(1)
            .ok_or(BackupAuditVerificationError::PhaseCountOverflow)?;
        Ok(())
    }
}

impl BackupAuditVerificationPolicy for BoundedBackupAuditVerificationPolicy {
    fn empty_state(&self) -> ChatRelayBackupAuditVerificationState {
        ChatRelayBackupAuditVerificationState {
            receipt: ChatRelayBackupAuditVerificationReceipt {
                verified: true,
                ..Default::default()
            },
            head_mac: EMPTY_CHAIN_MAC.to_string(),
            checkpoint_head_mac: EMPTY_CHAIN_MAC.to_string(),
        }
    }

    fn begin_segment(
        &self,
        state: &ChatRelayBackupAuditVerificationState,
    ) -> BackupAuditSegmentBaseline {
        BackupAuditSegmentBaseline {
            verified_bytes: state.receipt.verified_bytes,
            record_count: state.receipt.record_count,
        }
    }

    fn prepare_record(
        &self,
        state: &ChatRelayBackupAuditVerificationState,
        baseline: BackupAuditSegmentBaseline,
        encoded_bytes: usize,
    ) -> Result<BackupAuditRecordAdmission, BackupAuditVerificationError> {
        let segment_records = state
            .receipt
            .record_count
            .checked_sub(baseline.record_count)
            .ok_or(BackupAuditVerificationError::InvalidRecordAccounting)?;
        if segment_records >= self.limits.max_records_per_segment {
            return Err(BackupAuditVerificationError::RecordLimitReached);
        }
        let encoded_bytes = u64::try_from(encoded_bytes)
            .map_err(|_| BackupAuditVerificationError::RecordSizeOverflow)?;
        let verified_bytes_after = state
            .receipt
            .verified_bytes
            .checked_add(encoded_bytes)
            .ok_or(BackupAuditVerificationError::ByteCountOverflow)?;
        let segment_bytes = verified_bytes_after
            .checked_sub(baseline.verified_bytes)
            .ok_or(BackupAuditVerificationError::InvalidRecordAccounting)?;
        if segment_bytes > self.limits.max_bytes_per_segment {
            return Err(BackupAuditVerificationError::SegmentSizeExceeded);
        }
        if verified_bytes_after > self.limits.max_total_bytes {
            return Err(BackupAuditVerificationError::TotalSizeExceeded);
        }
        Ok(BackupAuditRecordAdmission {
            record_count_before: state.receipt.record_count,
            verified_bytes_before: state.receipt.verified_bytes,
            expected_sequence: state.next_sequence()?,
            expected_previous_mac: state.head_mac.clone(),
            verified_bytes_after,
        })
    }

    fn commit_record(
        &self,
        state: &mut ChatRelayBackupAuditVerificationState,
        admission: BackupAuditRecordAdmission,
        phase: BackupAuditPhase,
        timestamp: u64,
        record_mac: String,
    ) -> Result<(), BackupAuditVerificationError> {
        if state.receipt.record_count != admission.record_count_before
            || state.receipt.verified_bytes != admission.verified_bytes_before
            || state.head_mac != admission.expected_previous_mac
            || state.next_sequence()? != admission.expected_sequence
        {
            return Err(BackupAuditVerificationError::StaleRecordAdmission);
        }
        let mut next_receipt = state.receipt;
        Self::increment_phase(&mut next_receipt, phase)?;
        next_receipt.record_count = admission.expected_sequence;
        next_receipt.verified_bytes = admission.verified_bytes_after;
        next_receipt.last_recorded_at = Some(timestamp);
        state.receipt = next_receipt;
        state.head_mac = record_mac;
        Ok(())
    }

    fn prepare_checkpoint(
        &self,
        state: &ChatRelayBackupAuditVerificationState,
        range: ChatRelayBackupAuditSegmentRange,
        segment_bytes: u64,
        segment_sha256: String,
        previous_record_count: u64,
    ) -> Result<BackupAuditCheckpointAdmission, BackupAuditVerificationError> {
        let checkpoint_index = state
            .receipt
            .checkpoint_count
            .checked_add(1)
            .ok_or(BackupAuditVerificationError::CheckpointIndexOverflow)?;
        if previous_record_count.checked_add(1) != Some(range.first_sequence)
            || range.last_sequence != state.receipt.record_count
        {
            return Err(BackupAuditVerificationError::InvalidSegmentRange);
        }
        Ok(BackupAuditCheckpointAdmission {
            expected: BackupAuditCheckpointState {
                checkpoint_index,
                segment_first_sequence: range.first_sequence,
                segment_last_sequence: range.last_sequence,
                segment_bytes,
                segment_sha256,
                cumulative_verified_bytes: state.receipt.verified_bytes,
                cumulative_last_recorded_at: state.receipt.last_recorded_at,
                cumulative_dry_run_count: state.receipt.dry_run_count,
                cumulative_planned_count: state.receipt.planned_count,
                cumulative_completed_count: state.receipt.completed_count,
                cumulative_failed_count: state.receipt.failed_count,
                head_mac: state.head_mac.clone(),
                previous_checkpoint_mac: state.checkpoint_head_mac.clone(),
            },
            expected_record_count: state.receipt.record_count,
        })
    }

    fn commit_checkpoint(
        &self,
        state: &mut ChatRelayBackupAuditVerificationState,
        admission: BackupAuditCheckpointAdmission,
        checkpoint_mac: String,
    ) -> Result<(), BackupAuditVerificationError> {
        let expected = &admission.expected;
        if state.receipt.record_count != admission.expected_record_count
            || state.receipt.checkpoint_count.checked_add(1) != Some(expected.checkpoint_index)
            || state.receipt.verified_bytes != expected.cumulative_verified_bytes
            || state.receipt.last_recorded_at != expected.cumulative_last_recorded_at
            || state.receipt.dry_run_count != expected.cumulative_dry_run_count
            || state.receipt.planned_count != expected.cumulative_planned_count
            || state.receipt.completed_count != expected.cumulative_completed_count
            || state.receipt.failed_count != expected.cumulative_failed_count
            || state.head_mac != expected.head_mac
            || state.checkpoint_head_mac != expected.previous_checkpoint_mac
        {
            return Err(BackupAuditVerificationError::StaleCheckpointAdmission);
        }
        let archived_bytes = state
            .receipt
            .archived_bytes
            .checked_add(expected.segment_bytes)
            .ok_or(BackupAuditVerificationError::ArchivedByteCountOverflow)?;
        state.receipt.checkpoint_count = expected.checkpoint_index;
        state.receipt.archived_record_count = state.receipt.record_count;
        state.receipt.archived_bytes = archived_bytes;
        state.checkpoint_head_mac = checkpoint_mac;
        Ok(())
    }

    fn finalize_chain(
        &self,
        state: &mut ChatRelayBackupAuditVerificationState,
        rotation_pending: bool,
    ) -> Result<(), BackupAuditVerificationError> {
        let active_record_count = state
            .receipt
            .record_count
            .checked_sub(state.receipt.archived_record_count)
            .ok_or(BackupAuditVerificationError::InvalidRecordAccounting)?;
        state.receipt.active_record_count = active_record_count;
        state.receipt.rotation_pending = rotation_pending;
        Ok(())
    }

    fn mark_rotation_recovered(&self, state: &mut ChatRelayBackupAuditVerificationState) {
        state.receipt.rotation_pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> BoundedBackupAuditVerificationPolicy {
        BoundedBackupAuditVerificationPolicy::new(BackupAuditVerificationLimits {
            max_records_per_segment: 2,
            max_bytes_per_segment: 16,
            max_total_bytes: 24,
        })
    }

    fn range(first_sequence: u64, last_sequence: u64) -> ChatRelayBackupAuditSegmentRange {
        ChatRelayBackupAuditSegmentRange::new(first_sequence, last_sequence).expect("valid range")
    }

    fn commit_record(
        policy: &BoundedBackupAuditVerificationPolicy,
        state: &mut ChatRelayBackupAuditVerificationState,
        baseline: BackupAuditSegmentBaseline,
        bytes: usize,
        phase: BackupAuditPhase,
    ) {
        let admission = policy
            .prepare_record(state, baseline, bytes)
            .expect("prepare record");
        let record_mac = format!("{:064x}", state.receipt().record_count + 1);
        policy
            .commit_record(state, admission, phase, 1_000, record_mac)
            .expect("commit record");
    }

    #[test]
    fn commits_authenticated_records_atomically() {
        let policy = policy();
        let mut state = policy.empty_state();
        let baseline = policy.begin_segment(&state);
        let admission = policy
            .prepare_record(&state, baseline, 8)
            .expect("prepare first record");

        assert_eq!(state.receipt().record_count, 0);
        assert_eq!(admission.expected_sequence(), 1);
        assert_eq!(admission.expected_previous_mac(), EMPTY_CHAIN_MAC);
        policy
            .commit_record(
                &mut state,
                admission,
                BackupAuditPhase::Completed,
                1_000,
                "a".repeat(64),
            )
            .expect("commit first record");

        assert_eq!(state.receipt().record_count, 1);
        assert_eq!(state.receipt().completed_count, 1);
        assert_eq!(state.receipt().verified_bytes, 8);
        assert_eq!(state.head_mac(), "a".repeat(64));
    }

    #[test]
    fn rejects_stale_and_over_limit_records_without_mutation() {
        let policy = policy();
        let mut state = policy.empty_state();
        let baseline = policy.begin_segment(&state);
        let stale = policy
            .prepare_record(&state, baseline, 8)
            .expect("prepare stale record");
        commit_record(&policy, &mut state, baseline, 8, BackupAuditPhase::Planned);
        let snapshot = state.clone();

        assert_eq!(
            policy.commit_record(
                &mut state,
                stale,
                BackupAuditPhase::Failed,
                2_000,
                "f".repeat(64),
            ),
            Err(BackupAuditVerificationError::StaleRecordAdmission)
        );
        assert_eq!(state, snapshot);
        assert_eq!(
            policy.prepare_record(&state, baseline, 9),
            Err(BackupAuditVerificationError::SegmentSizeExceeded)
        );
        assert_eq!(state, snapshot);
    }

    #[test]
    fn commits_checkpoint_then_finalizes_active_accounting() {
        let policy = policy();
        let mut state = policy.empty_state();
        let baseline = policy.begin_segment(&state);
        commit_record(&policy, &mut state, baseline, 8, BackupAuditPhase::DryRun);
        let checkpoint = policy
            .prepare_checkpoint(&state, range(1, 1), 8, "b".repeat(64), 0)
            .expect("prepare checkpoint");
        assert_eq!(checkpoint.expected().checkpoint_index, 1);
        policy
            .commit_checkpoint(&mut state, checkpoint, "c".repeat(64))
            .expect("commit checkpoint");
        let active_baseline = policy.begin_segment(&state);
        commit_record(
            &policy,
            &mut state,
            active_baseline,
            4,
            BackupAuditPhase::Completed,
        );
        policy
            .finalize_chain(&mut state, true)
            .expect("finalize chain");

        assert_eq!(state.receipt().archived_record_count, 1);
        assert_eq!(state.receipt().active_record_count, 1);
        assert_eq!(state.receipt().archived_bytes, 8);
        assert!(state.receipt().rotation_pending);
        policy.mark_rotation_recovered(&mut state);
        assert!(!state.receipt().rotation_pending);
    }

    #[test]
    fn rejects_discontinuous_and_stale_checkpoints_without_mutation() {
        let policy = policy();
        let mut state = policy.empty_state();
        let baseline = policy.begin_segment(&state);
        commit_record(&policy, &mut state, baseline, 8, BackupAuditPhase::DryRun);
        assert_eq!(
            policy.prepare_checkpoint(&state, range(2, 2), 8, "b".repeat(64), 0),
            Err(BackupAuditVerificationError::InvalidSegmentRange)
        );
        let stale = policy
            .prepare_checkpoint(&state, range(1, 1), 8, "b".repeat(64), 0)
            .expect("prepare stale checkpoint");
        let active_baseline = policy.begin_segment(&state);
        commit_record(
            &policy,
            &mut state,
            active_baseline,
            4,
            BackupAuditPhase::Completed,
        );
        let snapshot = state.clone();
        assert_eq!(
            policy.commit_checkpoint(&mut state, stale, "c".repeat(64)),
            Err(BackupAuditVerificationError::StaleCheckpointAdmission)
        );
        assert_eq!(state, snapshot);
    }
}
