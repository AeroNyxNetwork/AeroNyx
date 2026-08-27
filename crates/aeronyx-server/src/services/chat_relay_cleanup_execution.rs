// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_cleanup_execution.rs
// ============================================
// Version: 1.0.0-BoundedCleanupExecution
//
// Creation Reason:
//   [CHAT-CLEANUP-EXECUTION-DOMAIN 2026-08-28 by Codex] Extract bounded
//   multi-transaction cleanup execution from the relay orchestration service.
//
// Main Functionality:
//   - Defines a replaceable cleanup execution capability.
//   - Freezes retention cutoffs once for an entire maintenance run.
//   - Executes bounded IMMEDIATE transactions with lock release between batches.
//   - Preserves committed partial progress when a later batch fails.
//   - Reports stable summary, failure, and deferred-backlog state to telemetry.
//
// Dependencies:
//   - `chat_relay_cleanup.rs` owns retention policy and one-batch mechanics.
//   - `chat_relay_quarantine.rs` owns de-identified poison-row evidence.
//   - `chat_relay_error.rs` owns the stable typed failure contract.
//   - `parking_lot` and `rusqlite` supply locking and transactions.
//
// Main Logical Flow:
//   1. Clamp the transaction budget to at least one batch.
//   2. Freeze all TTL cutoffs at the caller-provided run timestamp.
//   3. Lock, execute, and commit exactly one immediate transaction.
//   4. Release the connection before deciding whether another batch is needed.
//   5. Stop on completion, budget exhaustion, or the first fail-closed error.
//
// Important Note for Next Developer:
//   - Never wrap the entire run in one transaction; lock release is intentional.
//   - Earlier committed batches remain valid when a later batch fails.
//   - Set `backlog_deferred` only when the final allowed batch still has work.
//   - Do not log identities, row keys, replay keys, blobs, or payloads here.
//   - Scheduling, logs, and aggregate health status remain service-owned.
//
// Last Modified:
//   v1.0.0-BoundedCleanupExecution - Initial execution capability
// ============================================

use parking_lot::Mutex;
use rusqlite::{Connection, TransactionBehavior};

use super::chat_relay_cleanup::{
    CleanupBatchOutcome, CleanupRunSummary, RelayCleanupCutoffs, RelayCleanupDomain,
};
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};
use super::chat_relay_quarantine::DurableQuarantineDomain;
use crate::config::ChatRelayConfig;

/// Completed bounded run, including a possible failure after partial progress.
pub(crate) struct CleanupExecutionResult {
    pub(crate) summary: CleanupRunSummary,
    pub(crate) failure: Option<ChatRelayError>,
}

/// Replaceable capability for executing one bounded cleanup run.
pub(crate) trait RelayCleanupExecution {
    fn execute(
        &self,
        connection: &Mutex<Connection>,
        quarantine: &DurableQuarantineDomain,
        now: i64,
        max_batches: usize,
    ) -> CleanupExecutionResult;
}

/// Production bounded SQLite cleanup executor.
pub(crate) struct BoundedRelayCleanupExecutor {
    cleanup: RelayCleanupDomain,
}

impl BoundedRelayCleanupExecutor {
    pub(crate) fn new(
        config: &ChatRelayConfig,
        verified_submit_ttl_secs: u64,
        blind_route_ttl_secs: u64,
    ) -> Self {
        Self {
            cleanup: RelayCleanupDomain::new(
                config,
                verified_submit_ttl_secs,
                blind_route_ttl_secs,
            ),
        }
    }

    fn run_transaction(
        &self,
        connection: &mut Connection,
        quarantine: &DurableQuarantineDomain,
        now: i64,
        cutoffs: RelayCleanupCutoffs,
    ) -> ChatRelayResult<CleanupBatchOutcome> {
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let result = self
            .cleanup
            .run_batch(&transaction, quarantine, now, cutoffs);
        match result {
            Ok(outcome) => {
                transaction.commit()?;
                Ok(outcome)
            }
            Err(error) => Err(error),
        }
    }
}

impl RelayCleanupExecution for BoundedRelayCleanupExecutor {
    fn execute(
        &self,
        connection: &Mutex<Connection>,
        quarantine: &DurableQuarantineDomain,
        now: i64,
        max_batches: usize,
    ) -> CleanupExecutionResult {
        let max_batches = max_batches.max(1);
        let cutoffs = self.cleanup.cutoffs(now);
        let mut summary = CleanupRunSummary::default();
        let mut failure = None;

        for batch_index in 0..max_batches {
            let batch_result = {
                let mut connection = connection.lock();
                self.run_transaction(&mut connection, quarantine, now, cutoffs)
            };
            match batch_result {
                Ok(batch) => {
                    let has_more = batch.has_more;
                    summary.absorb(batch);
                    if !has_more {
                        break;
                    }
                    if batch_index + 1 == max_batches {
                        summary.backlog_deferred = true;
                    }
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }

        CleanupExecutionResult { summary, failure }
    }
}
