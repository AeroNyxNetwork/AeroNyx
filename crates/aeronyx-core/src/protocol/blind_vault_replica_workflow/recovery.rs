// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/recovery.rs
// ============================================
//! Fail-closed restart recovery decisions for replica workflow attempts.
//!
//! ## Creation Reason
//! A sealed workflow snapshot intentionally excludes terminal requests, reply
//! sessions, capabilities, replacement credentials, and provisioning secrets.
//! Restoring `AwaitingEvidence` therefore cannot imply that retransmission is
//! safe; the recovery path depends on the immutable planner action.
//!
//! ## Main Functionality
//! - Classifies every restored in-flight action without performing network I/O.
//! - Distinguishes read-only re-observation from private-journal recovery.
//! - Reports whether the original evidence window remains open or has expired.
//! - Keeps identifiers and recovery material out of telemetry and wire formats.
//!
//! ## Dependencies
//! - `blind_vault_replica_workflow.rs`: source-owned domain state.
//! - `blind_vault_replica_workflow/snapshot.rs`: authenticated restoration.
//! - `protocol::blind_vault`: immutable replica planner actions.
//!
//! ## Important Note For The Next Developer
//! - These values are local orchestration decisions, never network frames.
//! - Do not infer successful mutation from transport completion or timeout.
//! - Replacement/provisioning recovery requires the separately sealed private
//!   attempt journal; do not reconstruct credentials from public metadata.
//! - A fresh inventory or status observation must still pass normal evidence
//!   verification before it affects a new planner generation.
//!
//! Last Modified: v1.1.0-PrivacySafeRecoveryTaskDebug - Redacted source-local
//! work identity and absolute timing from standard task diagnostics.
//! v1.0.0-RestartRecoveryPlan - Initial bounded classification.
//! ============================================

use std::fmt;

use super::{
    require_timestamp, BlindVaultReplicaRestoredExecution, BlindVaultReplicaWorkId,
    BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError,
};
use crate::protocol::blind_vault::BlindVaultReplicaAction;

/// Safe source-side recovery strategy for one restored in-flight action.
///
/// [BLIND-VAULT-RESTART-RECOVERY-PLAN 2026-08-29 by Codex] The strategy is
/// derived from immutable action semantics. It never claims that a timed-out
/// mutating request failed, because the terminal may have committed it before
/// the source lost the encrypted reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaRestartRecoveryKind {
    /// Re-read the exact lease inventory; this operation has no storage side effect.
    ReobserveInventory,
    /// Read current lease status/inventory and build a fresh complete plan.
    RefreshPlanFromLeaseStatus,
    /// Restore the separately sealed replacement request/session/credential journal.
    RestoreReplacementAttemptJournal,
    /// Restore every separately sealed provisioning attempt journal.
    RestoreProvisioningAttemptJournal {
        /// Exact number of independent replica admissions authorized by the plan.
        replica_count: u8,
    },
}

impl BlindVaultReplicaRestartRecoveryKind {
    /// Whether safe continuation depends on private attempt material that the
    /// workflow snapshot deliberately does not contain.
    #[must_use]
    pub const fn requires_private_attempt_journal(self) -> bool {
        matches!(
            self,
            Self::RestoreReplacementAttemptJournal | Self::RestoreProvisioningAttemptJournal { .. }
        )
    }
}

/// Position of one recovered attempt relative to its original evidence window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaRestartRecoveryTiming {
    /// The original evidence window is still open, including its exact boundary.
    EvidenceWindowOpen { remaining_ms: u64 },
    /// The original evidence window elapsed without accepted terminal evidence.
    EvidenceWindowExpired { overdue_ms: u64 },
}

/// One bounded source-local recovery task derived from restored workflow state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaRestartRecoveryTask {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    dispatched_at_ms: u64,
    evidence_deadline_ms: u64,
    timing: BlindVaultReplicaRestartRecoveryTiming,
    kind: BlindVaultReplicaRestartRecoveryKind,
}

// [BLIND-VAULT-RECOVERY-TASK-DIAGNOSTICS 2026-08-30 by Codex] Recovery
// identity and absolute source times can correlate private work across logs.
// Keep only bounded operational state in standard diagnostics.
impl fmt::Debug for BlindVaultReplicaRestartRecoveryTask {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRestartRecoveryTask")
            .field("work_id", &"<redacted>")
            .field("attempt", &self.attempt)
            .field("timing", &self.timing)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl BlindVaultReplicaRestartRecoveryTask {
    /// Stable source-local identity of the ambiguous in-flight action.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    /// Original bounded dispatch attempt number.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    /// Source time at which the ambiguous attempt was dispatched.
    #[must_use]
    pub const fn dispatched_at_ms(&self) -> u64 {
        self.dispatched_at_ms
    }

    /// Last source time at which matching terminal evidence may be accepted.
    #[must_use]
    pub const fn evidence_deadline_ms(&self) -> u64 {
        self.evidence_deadline_ms
    }

    /// Whether the original evidence window is still open or already expired.
    #[must_use]
    pub const fn timing(&self) -> BlindVaultReplicaRestartRecoveryTiming {
        self.timing
    }

    /// Action-specific fail-closed recovery strategy.
    #[must_use]
    pub const fn kind(&self) -> BlindVaultReplicaRestartRecoveryKind {
        self.kind
    }
}

impl BlindVaultReplicaRestoredExecution {
    /// Derives recovery work for every restored in-flight action.
    ///
    /// The result preserves deterministic planner order. Empty output means no
    /// action was in flight when the authenticated snapshot was written; it
    /// does not imply whole-set convergence.
    pub fn restart_recovery_tasks(
        &self,
        now_ms: u64,
    ) -> Result<Vec<BlindVaultReplicaRestartRecoveryTask>, BlindVaultReplicaWorkflowError> {
        require_timestamp(now_ms)?;
        let mut tasks = Vec::with_capacity(self.execution.in_flight_count());
        for item in self.execution.items() {
            let BlindVaultReplicaWorkState::AwaitingEvidence {
                attempt,
                dispatched_at_ms,
                evidence_deadline_ms,
            } = item.state()
            else {
                continue;
            };
            if now_ms < dispatched_at_ms {
                return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
            }
            let timing = if now_ms <= evidence_deadline_ms {
                BlindVaultReplicaRestartRecoveryTiming::EvidenceWindowOpen {
                    remaining_ms: evidence_deadline_ms - now_ms,
                }
            } else {
                BlindVaultReplicaRestartRecoveryTiming::EvidenceWindowExpired {
                    overdue_ms: now_ms - evidence_deadline_ms,
                }
            };
            tasks.push(BlindVaultReplicaRestartRecoveryTask {
                work_id: item.id(),
                attempt,
                dispatched_at_ms,
                evidence_deadline_ms,
                timing,
                kind: recovery_kind(item.action()),
            });
        }
        Ok(tasks)
    }
}

const fn recovery_kind(action: BlindVaultReplicaAction) -> BlindVaultReplicaRestartRecoveryKind {
    match action {
        BlindVaultReplicaAction::RenewLease { .. } => {
            BlindVaultReplicaRestartRecoveryKind::RefreshPlanFromLeaseStatus
        }
        BlindVaultReplicaAction::ReconcileInventory { .. }
        | BlindVaultReplicaAction::RetryObservation { .. } => {
            BlindVaultReplicaRestartRecoveryKind::ReobserveInventory
        }
        BlindVaultReplicaAction::ReplaceReplica { .. } => {
            BlindVaultReplicaRestartRecoveryKind::RestoreReplacementAttemptJournal
        }
        BlindVaultReplicaAction::ProvisionReplicas { count } => {
            BlindVaultReplicaRestartRecoveryKind::RestoreProvisioningAttemptJournal {
                replica_count: count,
            }
        }
    }
}
