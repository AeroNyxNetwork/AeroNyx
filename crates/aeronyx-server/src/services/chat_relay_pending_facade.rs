// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_facade.rs
// ============================================
// Version: 1.1.0-PendingCursorTestSeam
//
// Creation Reason:
//   [CHAT-PENDING-FACADE-DOMAIN 2026-08-28 by Codex] Move offline-message
//   custody, delivery, quarantine telemetry, and acknowledgement APIs out of
//   the relay composition root without widening service field visibility.
//
// Modification Reason:
//   [CHAT-PENDING-CURSOR-SEAM-DOMAIN 2026-08-28 by Codex] Co-locate the
//   deterministic test-only v2 cursor decoder with pending delivery APIs.
//
// Main Functionality:
//   - Stores bounded encrypted envelopes for offline receivers.
//   - Delivers legacy cursor pages and stable v2 snapshot pages.
//   - Records aggregate quarantine outcomes without message identifiers.
//   - Atomically acknowledges receiver-bound message batches.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns the composed service and private fields.
//   - Pending custody owns admission, quotas, sequence allocation, and writes.
//   - Pending delivery owns reads, cursor semantics, and poison-row isolation.
//
// Main Logical Flow:
//   1. Validate and prepare a domain command before acquiring the DB lock.
//   2. Execute the bounded custody or delivery operation transactionally.
//   3. Convert quarantine outcomes into aggregate-only maintenance telemetry.
//   4. Return stable public contracts without exposing private durable rows.
//
// Important Note for Next Developer:
//   - V1 pagination must remain ordered by `message_id`, matching its cursor.
//   - V2 pagination must preserve the receiver-bound snapshot ceiling.
//   - Corrupt rows must be quarantined atomically; never skip them silently.
//   - ACK deletion must remain receiver-bound and batch-limited.
//   - Never log message IDs, wallet keys, ciphertext, routes, or raw rows.
//
// Last Modified:
//   v1.1.0-PendingCursorTestSeam - Co-located test-only cursor decoding
//   v1.0.0-PendingMessageFacade - Initial pending-message facade extraction
// ============================================

use aeronyx_core::protocol::chat::ChatEnvelope;
use tracing::{debug, warn};

use crate::services::chat_relay_pending_contract::{PendingMessage, PendingMessagePageV2};
use crate::services::chat_relay_pending_custody::PendingMessageStoreOutcome;
use crate::services::chat_relay_pending_delivery::PendingPullQuarantineSummary;
#[cfg(test)]
use crate::services::chat_relay_pull_cursor::PullCursorV2;

use super::{now_secs, ChatRelayResult, ChatRelayService};

impl ChatRelayService {
    /// Decodes one opaque v2 cursor for deterministic in-module tests.
    #[cfg(test)]
    pub(super) fn decode_pull_cursor_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2> {
        self.pending_delivery
            .decode_cursor(receiver, after_timestamp, encoded)
    }

    /// Stores a pending offline message for a receiver that is not currently online.
    ///
    /// # Errors
    ///
    /// Returns an item-size or durable-capacity error before insertion, or a
    /// serialization/SQLite error if encoding or the atomic write fails.
    pub fn store_pending(&self, envelope: &ChatEnvelope) -> ChatRelayResult<()> {
        let write = self.pending_custody.prepare_store(envelope, now_secs())?;
        let mut conn = self.conn.lock();
        let outcome = self.pending_custody.store(&mut conn, write)?;
        drop(conn);

        if let PendingMessageStoreOutcome::Stored { encoded_bytes } = outcome {
            debug!(encoded_bytes, "[CHAT_RELAY] Message stored pending");
        }
        Ok(())
    }

    fn record_pending_pull_quarantine(&self, summary: PendingPullQuarantineSummary) {
        let PendingPullQuarantineSummary::Replaced {
            quarantined_at,
            quarantined_rows,
            removed_events,
            retained_events,
        } = summary
        else {
            return;
        };
        self.maintenance_telemetry.record_quarantine(
            quarantined_at,
            quarantined_rows,
            0,
            removed_events,
            retained_events,
        );
        warn!(
            quarantined_pending_messages = quarantined_rows,
            "[CHAT_RELAY] Corrupt pending rows isolated during pull"
        );
    }

    /// Retrieves a page of pending messages for the given receiver wallet.
    ///
    /// The v1 wire cursor contains only `message_id`, so rows must be ordered
    /// by that same key. Ordering by timestamp first can permanently skip a
    /// later row whose random ID sorts below the previous page's cursor.
    ///
    /// # Errors
    ///
    /// Corrupt rows are atomically replaced by de-identified quarantine events
    /// so one poison row cannot permanently block a receiver's mailbox.
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: &[u8; 16],
        limit: u32,
    ) -> ChatRelayResult<(Vec<PendingMessage>, bool)> {
        let delivery = self.pending_delivery.pull_legacy(
            &self.conn,
            &self.durable_quarantine,
            receiver,
            after_timestamp,
            cursor,
            limit,
        )?;
        self.record_pending_pull_quarantine(delivery.quarantine);
        Ok((delivery.messages, delivery.has_more))
    }

    /// Retrieves one stable monotonic snapshot page for ChatPullV2.
    ///
    /// An empty cursor captures the current receiver-specific sequence ceiling.
    /// Later inserts receive larger sequences and cannot move into that snapshot,
    /// preventing duplicate/skip behavior while the client paginates. The
    /// sequence and ceiling remain node-internal inside an AEAD-protected cursor.
    ///
    /// # Errors
    ///
    /// Returns [`super::ChatRelayError::InvalidPullCursor`] for tampered,
    /// cross-wallet, cross-filter, malformed, or foreign-node cursors. Corrupt
    /// durable rows are atomically quarantined using the same path as v1 pulls.
    pub fn pull_pending_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded_cursor: &[u8],
        limit: u32,
    ) -> ChatRelayResult<PendingMessagePageV2> {
        let delivery = self.pending_delivery.pull_snapshot(
            &self.conn,
            &self.durable_quarantine,
            receiver,
            after_timestamp,
            encoded_cursor,
            limit,
        )?;
        self.record_pending_pull_quarantine(delivery.quarantine);
        Ok(delivery.page)
    }

    /// Acknowledges delivery of a batch of messages, deleting them from the store.
    ///
    /// Only deletes rows where `receiver = receiver_wallet`.
    ///
    /// # Errors
    ///
    /// Returns an oversized-batch or `SQLite` error. The transaction is atomic.
    pub fn ack_messages(
        &self,
        message_ids: &[[u8; 16]],
        receiver_wallet: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        let Some(batch) = self.pending_custody.prepare_acknowledgement(message_ids)? else {
            return Ok(0);
        };
        let deleted =
            self.pending_custody
                .acknowledge(&mut self.conn.lock(), &batch, receiver_wallet)?;

        debug!(count = deleted, "[CHAT_RELAY] Messages ACKed and deleted");
        Ok(deleted)
    }
}
