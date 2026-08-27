// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_status_facade.rs
// ============================================
// Version: 1.0.0-OperatorStatusFacade
//
// Creation Reason:
//   [CHAT-STATUS-FACADE-DOMAIN 2026-08-28 by Codex] Move configuration,
//   maintenance health, worker-failure recording, and durable storage usage
//   APIs out of the relay composition root without changing their contracts.
//
// Main Functionality:
//   - Exposes the effective immutable chat-relay configuration.
//   - Returns privacy-safe aggregate maintenance execution evidence.
//   - Records allowlisted blocking-worker failure reasons.
//   - Reads reconciled aggregate durable storage usage fail-closed.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns the composed service and stable re-exports.
//   - Maintenance telemetry owns snapshot and worker-failure transitions.
//   - Storage usage repository owns reconciled SQLite counter reads.
//
// Main Logical Flow:
//   1. Receive a status request through the existing service method.
//   2. Delegate to the narrowly scoped composed capability.
//   3. Return only configuration or aggregate privacy-safe status data.
//   4. Propagate unavailable storage telemetry as an error, never as zero.
//
// Important Note for Next Developer:
//   - Do not add message, wallet, peer, route, endpoint, or payload dimensions.
//   - Worker failures must remain a closed, allowlisted reason vocabulary.
//   - Storage accounting failures must remain unavailable, not synthetic zero.
//   - Keep these methods source-compatible with existing operator consumers.
//
// Last Modified:
//   v1.0.0-OperatorStatusFacade - Initial operator status facade extraction
// ============================================

use crate::config::ChatRelayConfig;
use crate::services::chat_relay_storage_usage::RelayStorageUsageRepository;

use super::{
    now_secs, ChatRelayMaintenanceStatus, ChatRelayResult, ChatRelayService, ChatRelayStorageUsage,
};

impl ChatRelayService {
    /// Returns the immutable effective relay configuration.
    #[must_use]
    pub fn config(&self) -> &ChatRelayConfig {
        &self.config
    }

    /// Returns aggregate TTL cleanup execution evidence.
    #[must_use]
    pub fn maintenance_status(&self) -> ChatRelayMaintenanceStatus {
        self.maintenance_telemetry.snapshot()
    }

    /// Records a blocking-worker failure that occurred outside `run_cleanup`.
    ///
    /// Tokio join failures are deliberately converted to stable buckets so a
    /// heartbeat never exposes panic payloads or other runtime internals.
    pub(crate) fn record_maintenance_worker_failure(&self, reason: &'static str) {
        self.maintenance_telemetry
            .record_worker_failure(now_secs(), reason);
    }

    /// Returns aggregate durable queue usage maintained by `SQLite` triggers.
    ///
    /// The result contains no message, wallet, sender, receiver, route, or
    /// payload identifiers and is safe for operator-capacity telemetry.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error if the reconciled singleton usage row cannot be
    /// read. Callers must treat that as unavailable telemetry, not zero usage.
    pub fn storage_usage(&self) -> ChatRelayResult<ChatRelayStorageUsage> {
        let conn = self.conn.lock();
        self.storage_usage_repository.read(&conn)
    }
}
