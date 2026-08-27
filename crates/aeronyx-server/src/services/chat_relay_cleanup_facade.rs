// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_cleanup_facade.rs
// ============================================
// Version: 1.0.0-CleanupFacade
//
// Creation Reason:
//   [CHAT-CLEANUP-FACADE-DOMAIN 2026-08-28 by Codex] Move bounded cleanup
//   orchestration, aggregate telemetry, and stable failure propagation out of
//   the relay composition root while preserving deterministic test seams.
//
// Main Functionality:
//   - Runs one bounded production cleanup cycle.
//   - Supports a controlled batch budget for deterministic maintenance tests.
//   - Records aggregate cleanup outcomes and returns the first typed failure.
//   - Emits privacy-safe operational summaries after transaction execution.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns the composed service and private fields.
//   - Cleanup execution owns bounded SQLite transactions and partial progress.
//   - Maintenance telemetry owns aggregate status transitions and counters.
//
// Main Logical Flow:
//   1. Capture one wall-clock cleanup boundary for execution and telemetry.
//   2. Execute at most the configured number of independently committed batches.
//   3. Record aggregate progress and stable failure evidence.
//   4. Return committed message/blob counts or the typed execution failure.
//
// Important Note for Next Developer:
//   - Earlier committed batches remain success when a later batch fails.
//   - Never collapse unavailable or failed cleanup into a zero-success result.
//   - Keep logs aggregate-only; never add identifiers, rows, keys, or payloads.
//   - `*_at` and budget methods are `pub(super)` only for sibling tests.
//
// Last Modified:
//   v1.0.0-CleanupFacade - Initial bounded-cleanup facade extraction
// ============================================

use tracing::{debug, info, warn};

use crate::services::chat_relay_cleanup::{CleanupRunSummary, CLEANUP_MAX_BATCHES_PER_RUN};
use crate::services::chat_relay_cleanup_execution::RelayCleanupExecution;

use super::{now_secs, ChatRelayError, ChatRelayResult, ChatRelayService};

impl ChatRelayService {
    /// Runs one TTL cleanup cycle (synchronous - call from `spawn_blocking`).
    ///
    /// Mutations run in a bounded sequence of `SQLite` IMMEDIATE transactions.
    /// Each committed batch releases the connection before the next begins.
    /// Returns `(expired_messages, expired_blobs)`.
    ///
    /// # Errors
    ///
    /// Returns a storage, serialization, or durable-data integrity error. A
    /// failed batch is rolled back and counted in maintenance evidence. Earlier
    /// committed batches remain durable and are included in aggregate counters.
    pub fn run_cleanup(&self) -> ChatRelayResult<(usize, usize)> {
        self.run_cleanup_with_batch_budget(CLEANUP_MAX_BATCHES_PER_RUN)
    }

    pub(super) fn run_cleanup_with_batch_budget(
        &self,
        max_batches: usize,
    ) -> ChatRelayResult<(usize, usize)> {
        let now = now_secs();
        let cleanup_now = i64::try_from(now).unwrap_or(i64::MAX);
        let (summary, failure) = self.run_cleanup_at(cleanup_now, max_batches);

        self.maintenance_telemetry
            .record_cleanup(now, summary, failure.as_ref());

        let Some(error) = failure else {
            return Ok((summary.expired_messages, summary.expired_blobs));
        };
        Err(error)
    }

    pub(super) fn run_cleanup_at(
        &self,
        now: i64,
        max_batches: usize,
    ) -> (CleanupRunSummary, Option<ChatRelayError>) {
        let execution =
            self.cleanup_execution
                .execute(&self.conn, &self.durable_quarantine, now, max_batches);
        let summary = execution.summary;
        let failure = execution.failure;

        if summary.removed_anything() || summary.backlog_deferred {
            info!(
                expired_messages = summary.expired_messages,
                expired_blobs = summary.expired_blobs,
                removed_notifications = summary.removed_notifications,
                quarantined_pending_messages = summary.quarantined_pending_messages,
                removed_quarantine_events = summary.removed_quarantine_events,
                removed_verified_submit_responses = summary.removed_verified_submit_responses,
                removed_verified_submit_reservations = summary.removed_verified_submit_reservations,
                removed_blind_route_responses = summary.removed_blind_route_responses,
                removed_blind_route_reservations = summary.removed_blind_route_reservations,
                retained_quarantine_events = summary.retained_quarantine_events,
                committed_batches = summary.successful_batches,
                backlog_deferred = summary.backlog_deferred,
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Bounded cleanup run complete"
            );
        } else {
            debug!(
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Cleanup: nothing to expire"
            );
        }
        if summary.quarantined_pending_messages > 0 {
            warn!(
                quarantined_pending_messages = summary.quarantined_pending_messages,
                "[CHAT_RELAY] Corrupt pending rows isolated during cleanup"
            );
        }

        (summary, failure)
    }
}
