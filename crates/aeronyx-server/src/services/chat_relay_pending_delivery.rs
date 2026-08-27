// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_delivery.rs
// ============================================
// Version: 1.0.0-PendingDeliveryDomain
//
// Creation Reason:
//   [CHAT-PENDING-DELIVERY-DOMAIN 2026-08-28 by Codex] Extract complete legacy
//   and snapshot pull use cases from the oversized relay orchestration service.
//
// Main Functionality:
//   - Composes pending-row validation with authenticated cursor protection.
//   - Owns the bounded SQLite lock scope for v1 and v2 pull reads.
//   - Atomically replaces poison rows with de-identified quarantine evidence.
//   - Finalizes stable pagination only after releasing the database lock.
//   - Returns typed quarantine counters for service-owned telemetry.
//
// Dependencies:
//   - `chat_relay_pending_pull.rs` owns ordered reads and row authentication.
//   - `chat_relay_pull_cursor.rs` owns the stable encrypted cursor wire format.
//   - `chat_relay_quarantine.rs` owns atomic poison-row replacement.
//   - `chat_relay.rs` owns public API models, logging, and aggregate status.
//
// Main Logical Flow:
//   1. Clamp the requested page size to the existing protocol bounds.
//   2. Decode or capture the receiver-bound snapshot cursor.
//   3. Read and authenticate a bounded page under one connection lock.
//   4. Replace any corrupt rows before releasing that lock.
//   5. Finalize `has_more`, progress, and the next opaque cursor in memory.
//
// Important Note for Next Developer:
//   - V1 ordering stays message-id based; v2 ordering stays queue-sequence based.
//   - Cursor bytes, version, AAD, timestamp binding, and bounds are wire ABI.
//   - A corrupt row must be quarantined before any valid page is returned.
//   - Never log receiver keys, message ids, cursor bytes, or row contents here.
//   - Keep final pagination outside the connection-lock scope.
//
// Last Modified:
//   v1.0.0-PendingDeliveryDomain - Initial pull use-case composition
// ============================================

use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use rusqlite::Connection;

use super::chat_relay::{PendingMessage, PendingMessagePageV2};
use super::chat_relay_error::ChatRelayResult;
use super::chat_relay_pending_pull::PendingMessagePullDomain;
use super::chat_relay_pull_cursor::{ChatPullCursorCodec, PullCursorV2};
use super::chat_relay_quarantine::{
    CorruptDurableRow, DurableQuarantineDomain, QuarantineReplaceOutcome, QuarantineRowTarget,
};

/// De-identified quarantine counters emitted by one pending pull.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum PendingPullQuarantineSummary {
    #[default]
    Clean,
    Replaced {
        quarantined_at: u64,
        quarantined_rows: usize,
        removed_events: usize,
        retained_events: usize,
    },
}

impl PendingPullQuarantineSummary {
    fn replaced(quarantined_at: u64, outcome: QuarantineReplaceOutcome) -> Self {
        Self::Replaced {
            quarantined_at,
            quarantined_rows: outcome.quarantined_rows,
            removed_events: outcome.removed_events,
            retained_events: outcome.retained_events,
        }
    }
}

/// Completed legacy page plus privacy-minimised maintenance evidence.
pub(crate) struct LegacyPendingDeliveryPage {
    pub(crate) messages: Vec<PendingMessage>,
    pub(crate) has_more: bool,
    pub(crate) quarantine: PendingPullQuarantineSummary,
}

/// Completed v2 snapshot page plus privacy-minimised maintenance evidence.
pub(crate) struct SnapshotPendingDeliveryPage {
    pub(crate) page: PendingMessagePageV2,
    pub(crate) quarantine: PendingPullQuarantineSummary,
}

/// Composed pending-message delivery use cases.
pub(crate) struct PendingMessageDeliveryDomain {
    pull: PendingMessagePullDomain,
    cursor: ChatPullCursorCodec,
}

impl PendingMessageDeliveryDomain {
    pub(crate) fn new(node_secret: &[u8; 32]) -> ChatRelayResult<Self> {
        Ok(Self {
            pull: PendingMessagePullDomain::new(),
            cursor: ChatPullCursorCodec::new(node_secret)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn decode_cursor(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2> {
        self.cursor.decode(receiver, after_timestamp, encoded)
    }

    pub(crate) fn pull_legacy(
        &self,
        connection: &Mutex<Connection>,
        quarantine: &DurableQuarantineDomain,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: &[u8; 16],
        limit: u32,
    ) -> ChatRelayResult<LegacyPendingDeliveryPage> {
        let page_limit = bounded_page_limit(limit);
        let (page, quarantine) = {
            let mut connection = connection.lock();
            let page = self.pull.read_legacy_page(
                &connection,
                receiver,
                after_timestamp,
                cursor,
                page_limit,
            )?;
            let quarantine =
                self.quarantine_corrupt_rows(&mut connection, quarantine, &page.corrupt_rows)?;
            (page, quarantine)
        };

        let mut messages = page.messages;
        let has_more = page.raw_has_more || messages.len() > page_limit;
        messages.truncate(page_limit);
        Ok(LegacyPendingDeliveryPage {
            messages,
            has_more,
            quarantine,
        })
    }

    pub(crate) fn pull_snapshot(
        &self,
        connection: &Mutex<Connection>,
        quarantine: &DurableQuarantineDomain,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded_cursor: &[u8],
        limit: u32,
    ) -> ChatRelayResult<SnapshotPendingDeliveryPage> {
        let page_limit = bounded_page_limit(limit);
        let decoded_cursor = if encoded_cursor.is_empty() {
            None
        } else {
            Some(
                self.cursor
                    .decode(receiver, after_timestamp, encoded_cursor)?,
            )
        };
        let (page, cursor, quarantine) = {
            let mut connection = connection.lock();
            let cursor = match decoded_cursor {
                Some(cursor) => cursor,
                None => PullCursorV2 {
                    position: 0,
                    ceiling: self.pull.capture_snapshot_ceiling(
                        &connection,
                        receiver,
                        after_timestamp,
                    )?,
                },
            };
            let page = self.pull.read_snapshot_page(
                &connection,
                receiver,
                after_timestamp,
                cursor.position,
                cursor.ceiling,
                page_limit,
            )?;
            let quarantine =
                self.quarantine_corrupt_rows(&mut connection, quarantine, &page.corrupt_rows)?;
            (page, cursor, quarantine)
        };

        let mut valid_messages = page.messages;
        let valid_overflow = valid_messages.len() > page_limit;
        let has_more = page.raw_has_more || valid_overflow;
        let next_position = if valid_overflow {
            valid_messages
                .get(page_limit.saturating_sub(1))
                .map(|(sequence, _)| *sequence)
                .unwrap_or(cursor.position)
        } else if has_more {
            page.raw_max_sequence.unwrap_or(cursor.position)
        } else {
            cursor.ceiling
        };
        valid_messages.truncate(page_limit);
        let messages = valid_messages
            .into_iter()
            .map(|(_, message)| message)
            .collect();
        let next_cursor = self.cursor.encode(
            receiver,
            after_timestamp,
            PullCursorV2 {
                position: next_position,
                ceiling: cursor.ceiling,
            },
        )?;

        Ok(SnapshotPendingDeliveryPage {
            page: PendingMessagePageV2 {
                messages,
                next_cursor,
                has_more,
            },
            quarantine,
        })
    }

    fn quarantine_corrupt_rows(
        &self,
        connection: &mut Connection,
        quarantine: &DurableQuarantineDomain,
        corrupt_rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<PendingPullQuarantineSummary> {
        if corrupt_rows.is_empty() {
            return Ok(PendingPullQuarantineSummary::Clean);
        }
        let quarantined_at = now_secs();
        let outcome = quarantine.replace_rows(
            connection,
            QuarantineRowTarget::PendingMessage,
            corrupt_rows,
            quarantined_at,
        )?;
        Ok(PendingPullQuarantineSummary::replaced(
            quarantined_at,
            outcome,
        ))
    }
}

fn bounded_page_limit(limit: u32) -> usize {
    usize::try_from(limit.clamp(1, 100)).unwrap_or(100)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
