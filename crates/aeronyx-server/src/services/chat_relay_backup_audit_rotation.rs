// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_rotation.rs
// ============================================
// Version: 1.0.0-RotationDomain
//
// Creation Reason:
//   [CHAT-RELAY-AUDIT-ROTATION-DOMAIN 2026-08-26 by Codex] Separate bounded
//   audit-segment rotation and crash-tail classification from filesystem I/O.
//
// Main Functionality:
//   - Models immutable segment ranges, fingerprints, limits, state, and plans.
//   - Decides whether the next record requires segment rotation.
//   - Validates rotation admission and computes the next range/checkpoint.
//   - Classifies a verified active file as a live tail or published duplicate.
//
// Dependencies:
//   - `chat_relay.rs` owns file handles, hashing, HMAC verification, hard links,
//     fsync ordering, removal, and crash-safe publication recovery.
//
// Main Logical Flow:
//   1. Check fixed record, byte, and segment limits without touching storage.
//   2. Build a monotonic segment range and checkpoint index with checked math.
//   3. Compare path-free fingerprints to classify the active crash window.
//   4. Return a closed plan/disposition for the service to execute.
//
// Important Note for Next Developer:
//   - This module must remain path-free and side-effect-free.
//   - Do not move hard-link, fsync, deletion, or file-lock behavior here.
//   - Preserve exact-boundary behavior: the last legal byte/record is accepted.
//   - New recovery states require an explicit enum variant and service mapping.
//
// Last Modified:
//   v1.0.0-RotationDomain - Initial trait-based extraction
// ============================================

/// Inclusive sequence range committed by one immutable audit segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ChatRelayBackupAuditSegmentRange {
    pub(crate) first_sequence: u64,
    pub(crate) last_sequence: u64,
}

impl ChatRelayBackupAuditSegmentRange {
    pub(crate) fn new(
        first_sequence: u64,
        last_sequence: u64,
    ) -> Result<Self, BackupAuditRotationError> {
        if first_sequence == 0 || last_sequence < first_sequence {
            return Err(BackupAuditRotationError::InvalidSegmentRange);
        }
        Ok(Self {
            first_sequence,
            last_sequence,
        })
    }
}

/// Path-free content fingerprint used to recognize a published duplicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackupAuditSegmentFingerprint {
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

/// Fixed resource ceilings for the append-only audit chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackupAuditRotationLimits {
    pub(crate) max_records_per_segment: u64,
    pub(crate) max_bytes_per_segment: u64,
    pub(crate) max_segments: u64,
}

/// Current cumulative counters required to plan one rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackupAuditRotationState {
    pub(crate) active_record_count: u64,
    pub(crate) archived_record_count: u64,
    pub(crate) record_count: u64,
    pub(crate) checkpoint_count: u64,
}

/// Immutable result consumed by checkpoint construction and publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackupAuditRotationPlan {
    pub(crate) range: ChatRelayBackupAuditSegmentRange,
    pub(crate) checkpoint_index: u64,
}

/// Closed active-tail states after archived segments have been verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditActiveTailDisposition {
    VerifyActiveTail,
    RetirePublishedDuplicate,
}

/// Closed policy failures mapped to stable SQLite-compatible service errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditRotationError {
    EmptyActiveSegment,
    SegmentLimitReached,
    SequenceOverflow,
    InvalidSegmentRange,
    CheckpointIndexOverflow,
    RecordSizeOverflow,
    ByteCountOverflow,
}

/// Replaceable pure policy boundary for audit-segment rotation decisions.
pub(crate) trait BackupAuditRotationPolicy {
    fn validate_admission(
        &self,
        state: BackupAuditRotationState,
    ) -> Result<(), BackupAuditRotationError>;

    fn plan_rotation(
        &self,
        state: BackupAuditRotationState,
    ) -> Result<BackupAuditRotationPlan, BackupAuditRotationError>;

    fn should_rotate(
        &self,
        active_record_count: u64,
        active_bytes: u64,
        next_record_bytes: usize,
    ) -> Result<bool, BackupAuditRotationError>;

    fn classify_active_tail(
        &self,
        active_fingerprint: Option<&BackupAuditSegmentFingerprint>,
        latest_segment_fingerprint: Option<&BackupAuditSegmentFingerprint>,
    ) -> BackupAuditActiveTailDisposition;
}

/// Production bounded policy preserving the existing exact-limit semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BoundedBackupAuditRotationPolicy {
    limits: BackupAuditRotationLimits,
}

impl BoundedBackupAuditRotationPolicy {
    pub(crate) const fn new(limits: BackupAuditRotationLimits) -> Self {
        Self { limits }
    }
}

impl BackupAuditRotationPolicy for BoundedBackupAuditRotationPolicy {
    fn validate_admission(
        &self,
        state: BackupAuditRotationState,
    ) -> Result<(), BackupAuditRotationError> {
        if state.active_record_count == 0 {
            return Err(BackupAuditRotationError::EmptyActiveSegment);
        }
        if state.checkpoint_count >= self.limits.max_segments {
            return Err(BackupAuditRotationError::SegmentLimitReached);
        }
        Ok(())
    }

    fn plan_rotation(
        &self,
        state: BackupAuditRotationState,
    ) -> Result<BackupAuditRotationPlan, BackupAuditRotationError> {
        self.validate_admission(state)?;
        let first_sequence = state
            .archived_record_count
            .checked_add(1)
            .ok_or(BackupAuditRotationError::SequenceOverflow)?;
        let range = ChatRelayBackupAuditSegmentRange::new(first_sequence, state.record_count)?;
        let checkpoint_index = state
            .checkpoint_count
            .checked_add(1)
            .ok_or(BackupAuditRotationError::CheckpointIndexOverflow)?;
        Ok(BackupAuditRotationPlan {
            range,
            checkpoint_index,
        })
    }

    fn should_rotate(
        &self,
        active_record_count: u64,
        active_bytes: u64,
        next_record_bytes: usize,
    ) -> Result<bool, BackupAuditRotationError> {
        let next_record_bytes = u64::try_from(next_record_bytes)
            .map_err(|_| BackupAuditRotationError::RecordSizeOverflow)?;
        let prospective_bytes = active_bytes
            .checked_add(next_record_bytes)
            .ok_or(BackupAuditRotationError::ByteCountOverflow)?;
        Ok(active_record_count >= self.limits.max_records_per_segment
            || prospective_bytes > self.limits.max_bytes_per_segment)
    }

    fn classify_active_tail(
        &self,
        active_fingerprint: Option<&BackupAuditSegmentFingerprint>,
        latest_segment_fingerprint: Option<&BackupAuditSegmentFingerprint>,
    ) -> BackupAuditActiveTailDisposition {
        if active_fingerprint.is_some() && active_fingerprint == latest_segment_fingerprint {
            BackupAuditActiveTailDisposition::RetirePublishedDuplicate
        } else {
            BackupAuditActiveTailDisposition::VerifyActiveTail
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> BoundedBackupAuditRotationPolicy {
        BoundedBackupAuditRotationPolicy::new(BackupAuditRotationLimits {
            max_records_per_segment: 4,
            max_bytes_per_segment: 100,
            max_segments: 2,
        })
    }

    fn state() -> BackupAuditRotationState {
        BackupAuditRotationState {
            active_record_count: 2,
            archived_record_count: 3,
            record_count: 5,
            checkpoint_count: 1,
        }
    }

    #[test]
    fn plans_monotonic_range_and_checkpoint() {
        assert_eq!(
            policy().plan_rotation(state()),
            Ok(BackupAuditRotationPlan {
                range: ChatRelayBackupAuditSegmentRange {
                    first_sequence: 4,
                    last_sequence: 5,
                },
                checkpoint_index: 2,
            })
        );
    }

    #[test]
    fn rejects_empty_limit_and_invalid_accounting() {
        let mut current = state();
        current.active_record_count = 0;
        assert_eq!(
            policy().validate_admission(current),
            Err(BackupAuditRotationError::EmptyActiveSegment)
        );

        current = state();
        current.checkpoint_count = 2;
        assert_eq!(
            policy().validate_admission(current),
            Err(BackupAuditRotationError::SegmentLimitReached)
        );

        current = state();
        current.archived_record_count = current.record_count;
        assert_eq!(
            policy().plan_rotation(current),
            Err(BackupAuditRotationError::InvalidSegmentRange)
        );
    }

    #[test]
    fn preserves_exact_record_and_byte_boundaries() {
        assert_eq!(policy().should_rotate(3, 90, 10), Ok(false));
        assert_eq!(policy().should_rotate(4, 0, 1), Ok(true));
        assert_eq!(policy().should_rotate(1, 91, 10), Ok(true));
        assert_eq!(
            policy().should_rotate(1, u64::MAX, 1),
            Err(BackupAuditRotationError::ByteCountOverflow)
        );
    }

    #[test]
    fn classifies_only_exact_nonempty_fingerprint_as_duplicate() {
        let fingerprint = BackupAuditSegmentFingerprint {
            bytes: 42,
            sha256: "a".repeat(64),
        };
        let different = BackupAuditSegmentFingerprint {
            bytes: 43,
            sha256: "b".repeat(64),
        };
        assert_eq!(
            policy().classify_active_tail(Some(&fingerprint), Some(&fingerprint)),
            BackupAuditActiveTailDisposition::RetirePublishedDuplicate
        );
        assert_eq!(
            policy().classify_active_tail(None, Some(&fingerprint)),
            BackupAuditActiveTailDisposition::VerifyActiveTail
        );
        assert_eq!(
            policy().classify_active_tail(Some(&different), Some(&fingerprint)),
            BackupAuditActiveTailDisposition::VerifyActiveTail
        );
    }
}
