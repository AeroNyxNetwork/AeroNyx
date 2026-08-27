// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_contract.rs
// ============================================
// Version: 1.0.0-BackupContractDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-CONTRACT-DOMAIN 2026-08-27 by Codex] Extract the
//   path-free backup command and receipt contracts from the oversized relay
//   orchestration service while preserving every existing public type path.
//
// Main Functionality:
//   - Defines aggregate backup, retention, prune, and restore-readiness receipts.
//   - Defines the backward-compatible host-local prune request.
//   - Models dry-run and destructive prune admission as explicit states.
//   - Keeps private artifact paths and identity metadata outside public output.
//
// Dependencies:
//   - `serde` serializes aggregate-only management-plane receipts.
//
// Main Logical Flow:
//   1. A management caller constructs a prune request or invokes an audit.
//   2. The request is admitted as dry-run, execute, or rejected fail-closed.
//   3. The relay I/O coordinator performs only the admitted operation.
//   4. The caller receives an aggregate receipt with no artifact identity.
//
// Important Note for Next Developer:
//   - Preserve public field names and serialization shape for compatibility.
//   - Never add paths, filenames, operation IDs, identities, or payload data.
//   - Destructive admission must continue to require both exact confirmations.
//   - Keep filesystem, SQLite, clocks, locks, and process state out of this file.
//
// Last Modified:
//   v1.0.0-BackupContractDomain - Initial contract and admission extraction
// ============================================

use serde::Serialize;

/// Exact phrase required before a host-local command may delete backup files.
pub const CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION: &str = "PRUNE-VERIFIED-RELAY-BACKUPS";

/// Aggregate result of one audited, idempotent custody backup operation.
///
/// [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] This intentionally
/// excludes the filesystem path and opaque artifact key. Management callers
/// may report only whether a verified image was created or reused and its size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChatRelayBackupReceipt {
    /// Size of the verified SQLite recovery image.
    pub(crate) size_bytes: u64,
    /// `true` only when this invocation published the artifact.
    pub(crate) created: bool,
}

/// Aggregate, path-free result of a verified custody-backup retention audit.
///
/// [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] No artifact name,
/// operation ID, filesystem timestamp, identity, route, or payload-derived
/// value may cross this boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ChatRelayBackupRetentionReceipt {
    /// Verified recovery images modeled as retained under the planning target.
    pub retained_count: usize,
    /// Aggregate verified bytes modeled as retained under the planning target.
    pub retained_bytes: u64,
    /// Verified recovery images exceeding the configured retention policy.
    pub excess_count: usize,
    /// Aggregate verified bytes exceeding the configured retention policy.
    pub excess_bytes: u64,
    /// Incomplete private SQLite entries observed after interrupted runs.
    pub partial_count: usize,
    /// Aggregate incomplete bytes observed after interrupted runs.
    pub partial_bytes: u64,
    /// The inventory or newest recovery point exceeds the byte target.
    pub budget_exceeded: bool,
}

/// Explicit host-local backup-prune request.
///
/// `execute=false` is a dry-run. Execution requires the exact public
/// confirmation phrase and an operator assertion that the node process is
/// stopped. These gates supplement, rather than replace, filesystem locking.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatRelayBackupPruneRequest {
    /// Whether eligible artifacts should actually be deleted.
    pub execute: bool,
    /// Exact confirmation phrase required for execution.
    pub confirmation: Option<String>,
    /// Operator assertion that the serving node process has been stopped.
    pub node_stopped_confirmed: bool,
}

/// Admitted behavior for one host-local backup prune request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupPruneAdmission {
    /// Inspect and audit the plan without deleting any artifact.
    DryRun,
    /// Delete only artifacts selected by the verified retention plan.
    Execute,
}

/// Fail-closed reason preventing destructive backup-prune admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum BackupPruneAdmissionError {
    /// One or both destructive-operation confirmations are absent or invalid.
    #[error("relay backup prune confirmation is incomplete")]
    IncompleteConfirmation,
}

impl ChatRelayBackupPruneRequest {
    /// Classifies this request without performing storage or filesystem work.
    pub(crate) fn admission(&self) -> Result<BackupPruneAdmission, BackupPruneAdmissionError> {
        if !self.execute {
            return Ok(BackupPruneAdmission::DryRun);
        }
        if self.confirmation.as_deref() == Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION)
            && self.node_stopped_confirmed
        {
            return Ok(BackupPruneAdmission::Execute);
        }
        Err(BackupPruneAdmissionError::IncompleteConfirmation)
    }
}

/// Aggregate, path-free result of one host-local prune command.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ChatRelayBackupPruneReceipt {
    /// Whether this invocation performed deletion rather than a dry-run.
    pub executed: bool,
    /// Verified complete recovery images selected by policy.
    pub planned_backup_count: usize,
    /// Aggregate bytes of selected complete recovery images.
    pub planned_backup_bytes: u64,
    /// Grace-expired interrupted files selected by policy.
    pub planned_partial_count: usize,
    /// Aggregate bytes of selected interrupted files.
    pub planned_partial_bytes: u64,
    /// Complete recovery images deleted by this invocation.
    pub deleted_backup_count: usize,
    /// Aggregate complete recovery-image bytes deleted.
    pub deleted_backup_bytes: u64,
    /// Interrupted files deleted by this invocation.
    pub deleted_partial_count: usize,
    /// Aggregate interrupted-file bytes deleted.
    pub deleted_partial_bytes: u64,
    /// Verified post-command retention state.
    pub remaining: ChatRelayBackupRetentionReceipt,
}

/// Aggregate, path-free result of a read-only recovery preflight.
///
/// [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] This contract never
/// identifies an artifact and never replaces or removes active/backup storage.
/// A ready result means an operator may evaluate a separately approved restore
/// flow; it does not mean a restore has happened.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ChatRelayRestoreReadinessReceipt {
    /// Whether all preflight gates needed by a future restore are satisfied.
    pub ready: bool,
    /// Number of fully verified recovery images in the private boundary.
    pub verified_backup_count: usize,
    /// Size of the newest fully verified recovery image.
    pub selected_backup_bytes: u64,
    /// Whether the configured active main database currently exists.
    pub active_database_present: bool,
    /// Size of the active main database, or zero when absent.
    pub active_database_bytes: u64,
    /// Whether any active SQLite journal/WAL/SHM sidecar exists.
    pub active_sidecars_present: bool,
    /// Stable aggregate blocker code; absent when `ready=true`.
    pub blocker: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use super::{
        BackupPruneAdmission, BackupPruneAdmissionError, ChatRelayBackupPruneRequest,
        CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION,
    };

    #[test]
    fn default_prune_request_is_dry_run() {
        assert_eq!(
            ChatRelayBackupPruneRequest::default().admission(),
            Ok(BackupPruneAdmission::DryRun)
        );
    }

    #[test]
    fn destructive_prune_requires_both_exact_confirmations() {
        let requests = [
            ChatRelayBackupPruneRequest {
                execute: true,
                confirmation: None,
                node_stopped_confirmed: true,
            },
            ChatRelayBackupPruneRequest {
                execute: true,
                confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
                node_stopped_confirmed: false,
            },
            ChatRelayBackupPruneRequest {
                execute: true,
                confirmation: Some("prune-verified-relay-backups".to_string()),
                node_stopped_confirmed: true,
            },
        ];
        for request in requests {
            assert_eq!(
                request.admission(),
                Err(BackupPruneAdmissionError::IncompleteConfirmation)
            );
        }
    }

    #[test]
    fn fully_confirmed_prune_is_admitted_for_execution() {
        let request = ChatRelayBackupPruneRequest {
            execute: true,
            confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
            node_stopped_confirmed: true,
        };
        assert_eq!(request.admission(), Ok(BackupPruneAdmission::Execute));
    }
}
