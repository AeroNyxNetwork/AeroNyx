// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_expired_facade.rs
// ============================================
// Version: 1.0.0-ExpiredNotificationFacade
//
// Creation Reason:
//   [CHAT-EXPIRED-FACADE-DOMAIN 2026-08-28 by Codex] Move expiry-control
//   delivery, poison-row isolation, compatibility reads, and pushed-state ACK
//   APIs out of the relay composition root without changing durable semantics.
//
// Main Functionality:
//   - Reads one bounded page of sender-bound expiry notifications.
//   - Replaces malformed durable rows with de-identified quarantine evidence.
//   - Preserves the legacy non-pageable compatibility wrapper.
//   - Atomically marks one validated notification batch as pushed.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns the composed service and private fields.
//   - Expired delivery owns bounded reads, row validation, and pushed writes.
//   - Durable quarantine owns atomic poison-row replacement and retention.
//
// Main Logical Flow:
//   1. Read and validate one bounded sender-specific page under the DB lock.
//   2. Atomically replace any corrupt source rows with anonymous evidence.
//   3. Record aggregate quarantine telemetry after releasing the DB lock.
//   4. Return valid notifications and pagination state, or ACK a sent batch.
//
// Important Note for Next Developer:
//   - Never skip malformed rows without durable quarantine evidence.
//   - Never expose raw rows, sender keys, message IDs, or decoding details.
//   - Preserve the extra-row `has_more` contract and bounded ACK semantics.
//   - Keep telemetry updates outside the SQLite lock scope.
//
// Last Modified:
//   v1.0.0-ExpiredNotificationFacade - Initial expiry-control facade extraction
// ============================================

use tracing::warn;

use crate::services::chat_relay_expired_contract::ExpiredNotification;
use crate::services::chat_relay_quarantine::QuarantineRowTarget;

use super::{now_secs, ChatRelayResult, ChatRelayService};

impl ChatRelayService {
    /// Retrieves one bounded page of expiry notifications for a sender.
    ///
    /// The extra row is used only to compute `has_more`; it is never returned.
    /// Invalid durable rows are atomically replaced by de-identified quarantine
    /// evidence so one poison row cannot permanently block sender control flow.
    ///
    /// # Errors
    ///
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<(Vec<ExpiredNotification>, bool)> {
        let mut conn = self.conn.lock();
        let page = self
            .expired_notification_delivery
            .read_page(&conn, sender)?;

        if page.corrupt_rows.is_empty() {
            drop(conn);
        } else {
            let quarantine_now = now_secs();
            let outcome = self.durable_quarantine.replace_rows(
                &mut conn,
                QuarantineRowTarget::ExpiredNotification,
                &page.corrupt_rows,
                quarantine_now,
            )?;
            drop(conn);

            self.maintenance_telemetry.record_quarantine(
                quarantine_now,
                0,
                outcome.quarantined_rows,
                outcome.removed_events,
                outcome.retained_events,
            );
            warn!(
                quarantined_expired_notifications = outcome.quarantined_rows,
                "[CHAT_RELAY] Corrupt expiry notifications isolated during pull"
            );
        }

        Ok((page.notifications, page.has_more))
    }

    /// Compatibility wrapper for callers that do not consume pagination yet.
    ///
    /// New runtime code should use [`Self::pull_pending_notifications`] and
    /// propagate its `has_more` flag.
    ///
    /// # Errors
    ///
    /// Returns a storage, decoding, or durable-data integrity error.
    pub fn get_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<Vec<ExpiredNotification>> {
        self.pull_pending_notifications(sender)
            .map(|(notifications, _)| notifications)
    }

    /// Atomically marks a successfully written notification page as pushed.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error and rolls back the whole page on failure.
    pub fn mark_notifications_pushed(&self, ids: &[i64]) -> ChatRelayResult<()> {
        let Some(batch) = self
            .expired_notification_delivery
            .prepare_acknowledgement(ids)
        else {
            return Ok(());
        };
        self.expired_notification_delivery
            .mark_pushed(&mut self.conn.lock(), &batch)
    }
}
