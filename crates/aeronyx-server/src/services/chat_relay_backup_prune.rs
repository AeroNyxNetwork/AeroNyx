// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_prune.rs
// ============================================
// Version: 1.0.0-AuditedBackupPruneDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-PRUNE-DOMAIN 2026-08-27 by Codex] Extract the complete
//   audited prune command from the oversized relay orchestration service.
//
// Main Functionality:
//   - Admits dry-run or explicitly confirmed destructive prune commands.
//   - Builds checked aggregate plans from a verified private inventory.
//   - Revalidates candidates before ordered complete/partial artifact removal.
//   - Brackets mutation with planned, failed, and completed audit records.
//
// Dependencies:
//   - `chat_relay_backup_inventory` supplies verified plans and identity checks.
//   - `chat_relay_backup_io` supplies durable directory synchronization.
//   - `chat_relay_backup_audit` supplies path-free maintenance audit contracts.
//   - `chat_relay_backup_contract` supplies public request/receipt contracts.
//
// Main Logical Flow:
//   1. Classify the request before any mutable filesystem operation.
//   2. Inspect the complete private inventory and build checked plan totals.
//   3. Return an audited dry-run, or publish the destructive planned phase.
//   4. Reverify and remove candidates oldest-first, then fsync the directory.
//   5. Reinspect remaining custody and publish completed or failed evidence.
//
// Important Note for Next Developer:
//   - The caller must hold the cross-process maintenance lock for this call.
//   - Keep candidate order and every existing error string stable.
//   - A failed audit write must never turn failed deletion into success.
//   - Never add paths, filenames, identities, or payload metadata to receipts.
//
// Last Modified:
//   v1.0.0-AuditedBackupPruneDomain - Initial composed command extraction
// ============================================

use std::path::Path;

use crate::services::chat_relay_backup_artifact::BackupArtifactSnapshot;
use crate::services::chat_relay_backup_audit::{
    BackupAuditPhase, ChatRelayBackupMaintenanceAuditCounts,
};
use crate::services::chat_relay_backup_contract::{
    BackupPruneAdmission, ChatRelayBackupPruneReceipt, ChatRelayBackupPruneRequest,
};
use crate::services::chat_relay_backup_inventory::{
    checked_backup_artifact_bytes, BackupInventory, BackupInventoryLimits,
    ChatRelayBackupRetentionInspection,
};
use crate::services::chat_relay_backup_io::{
    backup_io_error, BackupFilesystem, LocalBackupFilesystem,
};
use crate::services::chat_relay_error::ChatRelayResult;

/// Path-free immutable totals selected by one verified retention inspection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BackupPrunePlan {
    backup_count: usize,
    backup_bytes: u64,
    partial_count: usize,
    partial_bytes: u64,
}

impl BackupPrunePlan {
    fn from_inspection(inspection: &ChatRelayBackupRetentionInspection) -> ChatRelayResult<Self> {
        Ok(Self {
            backup_count: inspection.excess_backups.len(),
            backup_bytes: checked_backup_artifact_bytes(
                &inspection.excess_backups,
                "relay backup prune-plan byte accounting overflow",
            )?,
            partial_count: inspection.stale_partials.len(),
            partial_bytes: checked_backup_artifact_bytes(
                &inspection.stale_partials,
                "relay backup partial-prune byte accounting overflow",
            )?,
        })
    }

    const fn audit_counts(
        self,
        progress: BackupPruneProgress,
    ) -> ChatRelayBackupMaintenanceAuditCounts {
        ChatRelayBackupMaintenanceAuditCounts {
            planned_backup_count: self.backup_count,
            planned_backup_bytes: self.backup_bytes,
            planned_partial_count: self.partial_count,
            planned_partial_bytes: self.partial_bytes,
            completed_backup_count: progress.backup_count,
            completed_backup_bytes: progress.backup_bytes,
            completed_partial_count: progress.partial_count,
            completed_partial_bytes: progress.partial_bytes,
        }
    }
}

/// Checked aggregate mutation progress used for failure and completion audit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BackupPruneProgress {
    backup_count: usize,
    backup_bytes: u64,
    partial_count: usize,
    partial_bytes: u64,
}

impl BackupPruneProgress {
    fn record_backup(&mut self, size_bytes: u64) -> ChatRelayResult<()> {
        self.backup_count += 1;
        self.backup_bytes = self.backup_bytes.checked_add(size_bytes).ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup deletion byte accounting overflow",
            )
        })?;
        Ok(())
    }

    fn record_partial(&mut self, size_bytes: u64) -> ChatRelayResult<()> {
        self.partial_count += 1;
        self.partial_bytes = self.partial_bytes.checked_add(size_bytes).ok_or_else(|| {
            backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup partial deletion byte accounting overflow",
            )
        })?;
        Ok(())
    }
}

/// Replaceable aggregate-only maintenance audit sink.
pub(super) trait BackupPruneAuditSink {
    fn record(
        &self,
        phase: BackupAuditPhase,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> ChatRelayResult<()>;
}

impl<F> BackupPruneAuditSink for F
where
    F: Fn(BackupAuditPhase, ChatRelayBackupMaintenanceAuditCounts) -> ChatRelayResult<()>,
{
    fn record(
        &self,
        phase: BackupAuditPhase,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> ChatRelayResult<()> {
        self(phase, counts)
    }
}

/// Replaceable removal and publication-durability boundary.
pub(super) trait BackupArtifactRemoval {
    fn remove_verified_backup(&self, artifact: &BackupArtifactSnapshot) -> ChatRelayResult<()>;
    fn remove_stale_partial(&self, artifact: &BackupArtifactSnapshot) -> ChatRelayResult<()>;
    fn sync_directory(&self, backup_directory: &Path) -> ChatRelayResult<()>;
}

/// Host filesystem implementation preserving stable deletion error semantics.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LocalBackupArtifactRemoval;

impl BackupArtifactRemoval for LocalBackupArtifactRemoval {
    fn remove_verified_backup(&self, artifact: &BackupArtifactSnapshot) -> ChatRelayResult<()> {
        std::fs::remove_file(artifact.path()).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_DELETE,
                "unable to remove verified relay backup artifact",
            )
        })
    }

    fn remove_stale_partial(&self, artifact: &BackupArtifactSnapshot) -> ChatRelayResult<()> {
        std::fs::remove_file(artifact.path()).map_err(|_| {
            backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_DELETE,
                "unable to remove grace-expired relay backup partial",
            )
        })
    }

    fn sync_directory(&self, backup_directory: &Path) -> ChatRelayResult<()> {
        LocalBackupFilesystem.sync_backup_directory(backup_directory)
    }
}

/// Capability boundary for one lock-protected backup prune command.
pub(super) trait BackupPruneExecutor {
    fn execute(
        &self,
        backup_directory: &Path,
        admission: BackupPruneAdmission,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt>;
}

/// Admits a prune request before the caller resolves or creates storage paths.
pub(super) fn admit_backup_prune_request(
    request: &ChatRelayBackupPruneRequest,
) -> ChatRelayResult<BackupPruneAdmission> {
    request.admission().map_err(|_| {
        backup_io_error(
            rusqlite::ffi::SQLITE_AUTH,
            "relay backup prune confirmation is incomplete",
        )
    })
}

/// Composed audited command over inventory, removal, and audit capabilities.
#[derive(Debug)]
pub(super) struct AuditedBackupPruneExecutor<Inventory, Removal, Audit> {
    inventory: Inventory,
    removal: Removal,
    audit: Audit,
}

impl<Inventory, Removal, Audit> AuditedBackupPruneExecutor<Inventory, Removal, Audit> {
    pub(super) const fn new(inventory: Inventory, removal: Removal, audit: Audit) -> Self {
        Self {
            inventory,
            removal,
            audit,
        }
    }
}

impl<Inventory, Removal, Audit> BackupPruneExecutor
    for AuditedBackupPruneExecutor<Inventory, Removal, Audit>
where
    Inventory: BackupInventory,
    Removal: BackupArtifactRemoval,
    Audit: BackupPruneAuditSink,
{
    fn execute(
        &self,
        backup_directory: &Path,
        admission: BackupPruneAdmission,
        now_unix_secs: u64,
        limits: BackupInventoryLimits,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        let inspection = self
            .inventory
            .inspect(backup_directory, now_unix_secs, limits)?;
        let plan = BackupPrunePlan::from_inspection(&inspection)?;

        if admission == BackupPruneAdmission::DryRun {
            self.audit.record(
                BackupAuditPhase::DryRun,
                plan.audit_counts(Default::default()),
            )?;
            return Ok(ChatRelayBackupPruneReceipt {
                executed: false,
                planned_backup_count: plan.backup_count,
                planned_backup_bytes: plan.backup_bytes,
                planned_partial_count: plan.partial_count,
                planned_partial_bytes: plan.partial_bytes,
                remaining: inspection.receipt,
                ..Default::default()
            });
        }

        // [CHAT-RELAY-BACKUP-PRUNE-DOMAIN 2026-08-27 by Codex] Publication of
        // `planned` is a mandatory precondition for every destructive effect.
        self.audit.record(
            BackupAuditPhase::Planned,
            plan.audit_counts(Default::default()),
        )?;

        let mut progress = BackupPruneProgress::default();
        let deletion_result = (|| -> ChatRelayResult<()> {
            for artifact in &inspection.excess_backups {
                self.inventory.reverify_candidate(artifact, true)?;
                self.removal.remove_verified_backup(artifact)?;
                progress.record_backup(artifact.size_bytes())?;
            }
            for partial in &inspection.stale_partials {
                self.inventory.reverify_candidate(partial, false)?;
                self.removal.remove_stale_partial(partial)?;
                progress.record_partial(partial.size_bytes())?;
            }
            self.removal.sync_directory(backup_directory)
        })();

        if let Err(error) = deletion_result {
            let _ = self
                .audit
                .record(BackupAuditPhase::Failed, plan.audit_counts(progress));
            return Err(error);
        }

        let remaining = match self
            .inventory
            .inspect(backup_directory, now_unix_secs, limits)
        {
            Ok(inspection) => inspection.receipt,
            Err(error) => {
                let _ = self
                    .audit
                    .record(BackupAuditPhase::Failed, plan.audit_counts(progress));
                return Err(error);
            }
        };
        self.audit
            .record(BackupAuditPhase::Completed, plan.audit_counts(progress))?;

        Ok(ChatRelayBackupPruneReceipt {
            executed: true,
            planned_backup_count: plan.backup_count,
            planned_backup_bytes: plan.backup_bytes,
            planned_partial_count: plan.partial_count,
            planned_partial_bytes: plan.partial_bytes,
            deleted_backup_count: progress.backup_count,
            deleted_backup_bytes: progress.backup_bytes,
            deleted_partial_count: progress.partial_count,
            deleted_partial_bytes: progress.partial_bytes,
            remaining,
        })
    }
}
