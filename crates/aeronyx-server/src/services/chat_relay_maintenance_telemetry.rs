// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_maintenance_telemetry.rs
// ============================================
// Version: 1.0.0-MaintenanceTelemetryDomain
//
// Creation Reason:
//   [CHAT-MAINTENANCE-TELEMETRY-DOMAIN 2026-08-28 by Codex] Extract the
//   maintenance status contract and all mutation rules from relay orchestration.
//
// Main Functionality:
//   - Defines the privacy-safe serialized maintenance status contract.
//   - Owns the single lock protecting complete maintenance snapshots.
//   - Records quarantine, cleanup, backlog, and worker-failure evidence.
//   - Uses saturating aggregate counters and stable failure reason buckets.
//
// Dependencies:
//   - `chat_relay_cleanup.rs` supplies one bounded run summary.
//   - `chat_relay_error.rs` supplies stable privacy-safe reason buckets.
//   - `parking_lot` provides atomic snapshot and update ownership.
//   - `serde` preserves the existing heartbeat and API wire contract.
//
// Main Logical Flow:
//   1. Convert process-sized counts into saturating public counters.
//   2. Apply all fields for one event while holding exactly one write lock.
//   3. Return complete cloned snapshots under one read lock.
//
// Important Note for Next Developer:
//   - Never add identities, routes, endpoints, payloads, or per-user labels.
//   - Preserve `serde(default)` for backward-compatible status snapshots.
//   - Keep all multi-field transitions under this module's single lock.
//   - Worker reasons must remain fixed privacy-safe buckets, never panic text.
//
// Last Modified:
//   v1.0.0-MaintenanceTelemetryDomain - Initial state-domain extraction
// ============================================

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::chat_relay_cleanup::CleanupRunSummary;
use super::chat_relay_error::ChatRelayError;

const CLEANUP_SUCCEEDED: &str = "succeeded";
const CLEANUP_FAILED: &str = "failed";

/// Aggregate TTL maintenance evidence safe for heartbeat and node health APIs.
///
/// This snapshot intentionally excludes message IDs, wallet keys, routes,
/// endpoints, payloads, and per-user counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayMaintenanceStatus {
    /// Total cleanup attempts, including failed transactions.
    pub cleanup_runs_total: u64,
    /// Cleanup attempts that returned an error.
    pub cleanup_failures_total: u64,
    /// Successfully committed bounded cleanup transactions.
    pub cleanup_batches_total: u64,
    /// Runs that reached their transaction budget with work still pending.
    pub cleanup_backlog_deferred_total: u64,
    /// Pending message rows removed by successfully committed batches.
    pub expired_messages_total: u64,
    /// Encrypted blob rows removed by successfully committed batches.
    pub expired_blobs_total: u64,
    /// Delivered or stale expiry-notification rows removed by committed batches.
    pub expired_notifications_removed_total: u64,
    /// Corrupt pending-message rows atomically isolated from active delivery.
    pub quarantined_pending_messages_total: u64,
    /// Corrupt expiry-notification rows atomically isolated from delivery.
    pub quarantined_expired_notifications_total: u64,
    /// De-identified quarantine event rows removed by bounded retention.
    pub quarantine_events_removed_total: u64,
    /// Current durable de-identified quarantine event rows.
    pub quarantine_events_retained: u64,
    /// Unix timestamp of the most recent poison-row isolation.
    pub last_quarantine_at: Option<u64>,
    /// Unix timestamp of the most recent cleanup attempt.
    pub last_cleanup_at: Option<u64>,
    /// Stable state bucket: `succeeded` or `failed`.
    pub last_cleanup_status: Option<String>,
    /// Stable aggregate failure bucket from [`ChatRelayError::reason_bucket`].
    pub last_cleanup_failure_reason: Option<String>,
    /// Number of successfully committed transactions in the latest run.
    pub last_cleanup_batches: u64,
    /// Whether the latest run deferred remaining work to the next timer tick.
    pub last_cleanup_backlog_deferred: bool,
    /// Corrupt pending-message rows isolated by the latest cleanup run.
    pub last_cleanup_quarantined_pending_messages: u64,
}

/// Thread-safe owner of the complete relay maintenance status state machine.
#[derive(Debug, Default)]
pub(crate) struct RelayMaintenanceTelemetry {
    status: RwLock<ChatRelayMaintenanceStatus>,
}

impl RelayMaintenanceTelemetry {
    /// Returns one internally consistent privacy-safe status snapshot.
    pub(crate) fn snapshot(&self) -> ChatRelayMaintenanceStatus {
        self.status.read().clone()
    }

    /// Initializes the retained quarantine count after durable reconciliation.
    pub(crate) fn set_retained_quarantine_events(&self, retained_events: usize) {
        self.status.write().quarantine_events_retained = counter(retained_events);
    }

    /// Records one atomic poison-row replacement result.
    pub(crate) fn record_quarantine(
        &self,
        now: u64,
        pending_messages: usize,
        expired_notifications: usize,
        removed_events: usize,
        retained_events: usize,
    ) {
        let mut status = self.status.write();
        status.quarantined_pending_messages_total = status
            .quarantined_pending_messages_total
            .saturating_add(counter(pending_messages));
        status.quarantined_expired_notifications_total = status
            .quarantined_expired_notifications_total
            .saturating_add(counter(expired_notifications));
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(counter(removed_events));
        status.quarantine_events_retained = counter(retained_events);
        status.last_quarantine_at = Some(now);
    }

    /// Records one bounded cleanup result, including committed partial progress.
    pub(crate) fn record_cleanup(
        &self,
        now: u64,
        summary: CleanupRunSummary,
        failure: Option<&ChatRelayError>,
    ) {
        let mut status = self.status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_batches_total = status
            .cleanup_batches_total
            .saturating_add(counter(summary.successful_batches));
        if summary.backlog_deferred {
            status.cleanup_backlog_deferred_total =
                status.cleanup_backlog_deferred_total.saturating_add(1);
        }
        status.expired_messages_total = status
            .expired_messages_total
            .saturating_add(counter(summary.expired_messages));
        status.expired_blobs_total = status
            .expired_blobs_total
            .saturating_add(counter(summary.expired_blobs));
        status.expired_notifications_removed_total = status
            .expired_notifications_removed_total
            .saturating_add(counter(summary.removed_notifications));
        status.quarantined_pending_messages_total = status
            .quarantined_pending_messages_total
            .saturating_add(counter(summary.quarantined_pending_messages));
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(counter(summary.removed_quarantine_events));
        if summary.successful_batches > 0 {
            status.quarantine_events_retained = counter(summary.retained_quarantine_events);
        }
        if summary.quarantined_pending_messages > 0 {
            status.last_quarantine_at = Some(now);
        }
        status.last_cleanup_at = Some(now);
        status.last_cleanup_batches = counter(summary.successful_batches);
        status.last_cleanup_backlog_deferred = summary.backlog_deferred;
        status.last_cleanup_quarantined_pending_messages =
            counter(summary.quarantined_pending_messages);
        match failure {
            None => {
                status.last_cleanup_status = Some(CLEANUP_SUCCEEDED.to_string());
                status.last_cleanup_failure_reason = None;
            }
            Some(error) => {
                status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
                status.last_cleanup_status = Some(CLEANUP_FAILED.to_string());
                status.last_cleanup_failure_reason = Some(error.reason_bucket().to_string());
            }
        }
    }

    /// Records a stable failure for a cleanup worker that did not return a run.
    pub(crate) fn record_worker_failure(&self, now: u64, reason: &'static str) {
        let mut status = self.status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
        status.last_cleanup_at = Some(now);
        status.last_cleanup_status = Some(CLEANUP_FAILED.to_string());
        status.last_cleanup_failure_reason = Some(reason.to_string());
        status.last_cleanup_batches = 0;
        status.last_cleanup_backlog_deferred = false;
    }
}

fn counter(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
