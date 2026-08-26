// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_retention.rs
// ============================================
// Version: 1.0.0-BoundedRetentionPolicy
//
// Creation Reason:
//   [CHAT-BACKUP-RETENTION-DOMAIN 2026-08-26 by Codex] Isolate the pure
//   recovery-image retention decision from filesystem inspection and deletion
//   so policy can be tested without private paths or mutable storage.
//
// Main Functionality:
//   - Models immutable count, byte, and interrupted-file grace limits.
//   - Selects retained recovery images newest-first.
//   - Emits complete and partial deletion candidates oldest-first.
//   - Uses checked accounting and fails closed on time/byte overflow.
//
// Dependencies:
//   - `chat_relay.rs` verifies private artifacts and performs locked deletion.
//   - `ChatRelayConfig` supplies validated limits to the composed policy.
//
// Main Logical Flow:
//   1. Sort verified complete images by stable newest-first identity.
//   2. Preserve at least the newest recovery point, then apply count/byte caps.
//   3. Reverse excess images into deterministic oldest-first deletion order.
//   4. Select grace-expired partials in deterministic oldest-first order.
//
// Important Note for Next Developer:
//   - This module must remain path-blind and side-effect free.
//   - The caller must verify artifact identity before and during deletion.
//   - Never change the newest-image safety exception without a migration plan.
//   - Deletion candidates must remain oldest-first so partial failure preserves
//     the strongest available recovery history.
//
// Last Modified:
//   v1.0.0-BoundedRetentionPolicy - Initial trait-based policy extraction
// ============================================

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Immutable limits for one verified backup-retention decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackupRetentionLimits {
    pub(crate) target_artifacts: usize,
    pub(crate) target_bytes: u64,
    pub(crate) partial_grace_secs: u64,
}

impl BackupRetentionLimits {
    pub(crate) const fn new(
        target_artifacts: usize,
        target_bytes: u64,
        partial_grace_secs: u64,
    ) -> Self {
        Self {
            target_artifacts,
            target_bytes,
            partial_grace_secs,
        }
    }
}

/// Closed failure vocabulary for pure retention planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupRetentionPolicyError {
    RetainedBytesOverflow,
    ExcessBytesOverflow,
    PartialBytesOverflow,
    PartialCutoffOutOfRange,
}

/// Minimal artifact view consumed by the path-blind retention policy.
pub(crate) trait BackupRetentionArtifact: Clone {
    fn size_bytes(&self) -> u64;
    fn modified_at(&self) -> SystemTime;
    fn stable_name(&self) -> &str;
}

/// Side-effect-free result consumed by the filesystem-owning relay service.
#[derive(Debug)]
pub(crate) struct BackupRetentionPlan<Artifact> {
    pub(crate) retained_count: usize,
    pub(crate) retained_bytes: u64,
    pub(crate) excess_count: usize,
    pub(crate) excess_bytes: u64,
    pub(crate) partial_count: usize,
    pub(crate) partial_bytes: u64,
    pub(crate) budget_exceeded: bool,
    pub(crate) newest_backup: Option<Artifact>,
    pub(crate) excess_oldest_first: Vec<Artifact>,
    pub(crate) stale_partials_oldest_first: Vec<Artifact>,
}

/// Replaceable retention-policy boundary for verified opaque artifacts.
pub(crate) trait BackupRetentionPlanner<Artifact>
where
    Artifact: BackupRetentionArtifact,
{
    fn plan(
        &self,
        complete: Vec<Artifact>,
        partials: Vec<Artifact>,
        now_unix_secs: u64,
        limits: BackupRetentionLimits,
    ) -> Result<BackupRetentionPlan<Artifact>, BackupRetentionPolicyError>;
}

/// Production bounded retention policy.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BoundedBackupRetentionPlanner;

impl<Artifact> BackupRetentionPlanner<Artifact> for BoundedBackupRetentionPlanner
where
    Artifact: BackupRetentionArtifact,
{
    fn plan(
        &self,
        mut complete: Vec<Artifact>,
        mut partials: Vec<Artifact>,
        now_unix_secs: u64,
        limits: BackupRetentionLimits,
    ) -> Result<BackupRetentionPlan<Artifact>, BackupRetentionPolicyError> {
        complete.sort_by(|left, right| {
            right
                .modified_at()
                .cmp(&left.modified_at())
                .then_with(|| right.stable_name().cmp(left.stable_name()))
        });
        let newest_backup = complete.first().cloned();
        let mut retained_count = 0usize;
        let mut retained_bytes = 0u64;
        let mut excess_bytes = 0u64;
        let mut excess_newest_first = Vec::new();

        for artifact in complete {
            let next_bytes = retained_bytes.checked_add(artifact.size_bytes());
            let fits = retained_count < limits.target_artifacts
                && next_bytes
                    .map(|bytes| bytes <= limits.target_bytes)
                    .unwrap_or(false);
            if retained_count == 0 || fits {
                retained_count += 1;
                retained_bytes = retained_bytes
                    .checked_add(artifact.size_bytes())
                    .ok_or(BackupRetentionPolicyError::RetainedBytesOverflow)?;
            } else {
                excess_bytes = excess_bytes
                    .checked_add(artifact.size_bytes())
                    .ok_or(BackupRetentionPolicyError::ExcessBytesOverflow)?;
                excess_newest_first.push(artifact);
            }
        }

        let excess_count = excess_newest_first.len();
        excess_newest_first.reverse();

        let partial_count = partials.len();
        let partial_bytes = partials.iter().try_fold(0u64, |total, partial| {
            total
                .checked_add(partial.size_bytes())
                .ok_or(BackupRetentionPolicyError::PartialBytesOverflow)
        })?;
        let partial_cutoff = UNIX_EPOCH
            .checked_add(Duration::from_secs(
                now_unix_secs.saturating_sub(limits.partial_grace_secs),
            ))
            .ok_or(BackupRetentionPolicyError::PartialCutoffOutOfRange)?;
        partials.sort_by(|left, right| {
            left.modified_at()
                .cmp(&right.modified_at())
                .then_with(|| left.stable_name().cmp(right.stable_name()))
        });
        let stale_partials_oldest_first = partials
            .into_iter()
            .filter(|partial| partial.modified_at() <= partial_cutoff)
            .collect();

        Ok(BackupRetentionPlan {
            retained_count,
            retained_bytes,
            excess_count,
            excess_bytes,
            partial_count,
            partial_bytes,
            budget_exceeded: excess_count > 0
                || retained_count > limits.target_artifacts
                || retained_bytes > limits.target_bytes,
            newest_backup,
            excess_oldest_first: excess_newest_first,
            stale_partials_oldest_first,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestArtifact {
        name: &'static str,
        size: u64,
        modified_at: SystemTime,
    }

    impl TestArtifact {
        fn at(name: &'static str, size: u64, modified_secs: u64) -> Self {
            Self {
                name,
                size,
                modified_at: UNIX_EPOCH + Duration::from_secs(modified_secs),
            }
        }
    }

    impl BackupRetentionArtifact for TestArtifact {
        fn size_bytes(&self) -> u64 {
            self.size
        }

        fn modified_at(&self) -> SystemTime {
            self.modified_at
        }

        fn stable_name(&self) -> &str {
            self.name
        }
    }

    fn names(artifacts: &[TestArtifact]) -> Vec<&str> {
        artifacts.iter().map(|artifact| artifact.name).collect()
    }

    #[test]
    fn keeps_newest_recovery_point_even_when_it_exceeds_byte_target() {
        let plan = BoundedBackupRetentionPlanner
            .plan(
                vec![TestArtifact::at("newest", 11, 30)],
                Vec::new(),
                40,
                BackupRetentionLimits::new(1, 10, 5),
            )
            .unwrap();

        assert_eq!(plan.retained_count, 1);
        assert_eq!(plan.retained_bytes, 11);
        assert!(plan.budget_exceeded);
        assert_eq!(plan.newest_backup.as_ref().unwrap().name, "newest");
    }

    #[test]
    fn emits_complete_deletion_candidates_oldest_first() {
        let plan = BoundedBackupRetentionPlanner
            .plan(
                vec![
                    TestArtifact::at("middle", 10, 20),
                    TestArtifact::at("oldest", 10, 10),
                    TestArtifact::at("newest", 10, 30),
                ],
                Vec::new(),
                40,
                BackupRetentionLimits::new(1, 100, 5),
            )
            .unwrap();

        assert_eq!(plan.retained_count, 1);
        assert_eq!(names(&plan.excess_oldest_first), vec!["oldest", "middle"]);
    }

    #[test]
    fn emits_only_grace_expired_partials_in_stable_oldest_first_order() {
        let plan = BoundedBackupRetentionPlanner
            .plan(
                Vec::new(),
                vec![
                    TestArtifact::at("fresh", 1, 98),
                    TestArtifact::at("old-b", 2, 80),
                    TestArtifact::at("old-a", 3, 80),
                ],
                100,
                BackupRetentionLimits::new(1, 100, 10),
            )
            .unwrap();

        assert_eq!(plan.partial_count, 3);
        assert_eq!(plan.partial_bytes, 6);
        assert_eq!(
            names(&plan.stale_partials_oldest_first),
            vec!["old-a", "old-b"]
        );
    }

    #[test]
    fn checked_accounting_fails_closed_on_partial_byte_overflow() {
        let error = BoundedBackupRetentionPlanner
            .plan(
                Vec::new(),
                vec![
                    TestArtifact::at("first", u64::MAX, 1),
                    TestArtifact::at("second", 1, 2),
                ],
                10,
                BackupRetentionLimits::new(1, u64::MAX, 1),
            )
            .unwrap_err();

        assert_eq!(error, BackupRetentionPolicyError::PartialBytesOverflow);
    }
}
