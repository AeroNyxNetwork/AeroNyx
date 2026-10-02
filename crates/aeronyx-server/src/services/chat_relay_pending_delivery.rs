// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_delivery.rs
// ============================================
// Version: 1.2.0-ByteAwareSnapshotPaging
//
// Creation Reason:
//   [CHAT-PENDING-DELIVERY-DOMAIN 2026-08-28 by Codex] Extract complete legacy
//   and snapshot pull use cases from the oversized relay orchestration service.
//
// Modification Reason:
//   [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Select a whole-envelope byte
//   prefix before protecting its cursor; preserve the public count-only path.
//   [CHAT-PENDING-CONTRACT-DOMAIN 2026-08-28 by Codex] Depend directly on the
//   pending delivery contract instead of the central relay orchestrator.
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
//   - `chat_relay_pending_contract.rs` owns public delivery models.
//   - `chat_relay.rs` owns logging, aggregate status, and API re-exports.
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
//   v1.2.0-ByteAwareSnapshotPaging - 2026-10-03, select prefix before cursor
//   v1.1.0-PendingContractDependency - Removed orchestrator dependency
//   v1.0.0-PendingDeliveryDomain - Initial pull use-case composition
// ============================================

use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use rusqlite::Connection;

use aeronyx_core::crypto::transport::ENCRYPTION_OVERHEAD;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::messages::DATA_PACKET_HEADER_SIZE;

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};
use super::chat_relay_pending_contract::{PendingMessage, PendingMessagePageV2};
use super::chat_relay_pending_pull::PendingMessagePullDomain;
use super::chat_relay_pull_cursor::{ChatPullCursorCodec, PullCursorV2, ENCODED_CURSOR_BYTES};
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

// [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Transport policy is internal:
// neither an admission ceiling nor a promise that an oversized first item fits.
#[derive(Clone, Copy)]
pub(crate) enum SnapshotPageBudget {
    CountOnly,
    UdpCoalescing { target_bytes: usize },
}

impl SnapshotPageBudget {
    fn select_prefix(
        self,
        messages: &[(u64, PendingMessage)],
        page_limit: usize,
    ) -> ChatRelayResult<(usize, Option<usize>)> {
        let Self::UdpCoalescing { target_bytes } = self else {
            return Ok((messages.len().min(page_limit), None));
        };
        // Measure the actual empty response, including the frozen opaque cursor
        // and outer transport framing. The final encode below checks composition.
        let mut bytes = snapshot_datagram_bytes(&[], &[0; ENCODED_CURSOR_BYTES])?;
        let mut selected = 0;
        for (_, message) in messages.iter().take(page_limit) {
            let item_bytes = usize::try_from(bincode::serialized_size(&message.envelope)?)
                .map_err(|_| snapshot_size_error())?;
            let next = checked_page_bytes(bytes, item_bytes)?;
            if selected != 0 && next > target_bytes {
                break;
            }
            bytes = next;
            selected += 1;
            if bytes > target_bytes {
                break;
            }
        }
        Ok((selected, Some(bytes)))
    }

    fn validate(
        self,
        page: &PendingMessagePageV2,
        expected_bytes: Option<usize>,
    ) -> ChatRelayResult<()> {
        let Self::UdpCoalescing { target_bytes } = self else {
            return Ok(());
        };
        let actual = snapshot_datagram_bytes(&page.messages, &page.next_cursor)?;
        if Some(actual) != expected_bytes || (actual > target_bytes && page.messages.len() != 1) {
            return Err(snapshot_size_error());
        }
        Ok(())
    }
}

fn snapshot_size_error() -> ChatRelayError {
    ChatRelayError::Serialize(Box::new(bincode::ErrorKind::Custom(
        "snapshot_page_size".into(),
    )))
}

fn checked_page_bytes(current: usize, additional: usize) -> ChatRelayResult<usize> {
    current
        .checked_add(additional)
        .ok_or_else(snapshot_size_error)
}

fn snapshot_datagram_bytes(messages: &[PendingMessage], cursor: &[u8]) -> ChatRelayResult<usize> {
    let response = MemChainMessage::ChatPullResponseV2 {
        envelopes: messages
            .iter()
            .map(|message| message.envelope.clone())
            .collect(),
        next_cursor: cursor.to_vec(),
        // Both canonical bool values occupy one byte.
        has_more: false,
    };
    let clear = encode_memchain(&response).map_err(|_| snapshot_size_error())?;
    checked_page_bytes(
        checked_page_bytes(clear.len(), DATA_PACKET_HEADER_SIZE)?,
        ENCRYPTION_OVERHEAD,
    )
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
        self.pull_snapshot_budgeted(
            connection,
            quarantine,
            receiver,
            after_timestamp,
            encoded_cursor,
            limit,
            SnapshotPageBudget::CountOnly,
        )
    }

    // [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Read the same bounded rows;
    // never perform a second query or recapture the ceiling to shorten a page.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pull_snapshot_budgeted(
        &self,
        connection: &Mutex<Connection>,
        quarantine: &DurableQuarantineDomain,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded_cursor: &[u8],
        limit: u32,
        budget: SnapshotPageBudget,
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
        let (selected, expected_bytes) = budget.select_prefix(&valid_messages, page_limit)?;
        let valid_overflow = valid_messages.len() > selected;
        let has_more = page.raw_has_more || valid_overflow;
        let next_position = if valid_overflow {
            valid_messages
                .get(selected.saturating_sub(1))
                .map(|(sequence, _)| *sequence)
                .unwrap_or(cursor.position)
        } else if has_more {
            page.raw_max_sequence.unwrap_or(cursor.position)
        } else {
            cursor.ceiling
        };
        valid_messages.truncate(selected);
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

        let page = PendingMessagePageV2 {
            messages,
            next_cursor,
            has_more,
        };
        budget.validate(&page, expected_bytes)?;
        Ok(SnapshotPendingDeliveryPage { page, quarantine })
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

// [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Codec-only selection tests;
// repository authentication and durable cursor semantics are tested in the facade.
#[cfg(test)]
mod byte_paging_tests {
    use super::*;
    use aeronyx_core::protocol::chat::{ChatContentType, ChatEnvelope};

    fn message(sequence: u8, bytes: usize) -> (u64, PendingMessage) {
        let envelope = ChatEnvelope {
            message_id: [sequence; 16],
            sender: [0; 32],
            receiver: [0; 32],
            timestamp: 0,
            ciphertext: vec![0; bytes],
            nonce: [0; 24],
            content_type: ChatContentType::Text,
            signature: [0; 64],
        };
        (
            u64::from(sequence),
            PendingMessage {
                message_id: envelope.message_id,
                envelope,
            },
        )
    }

    #[test]
    fn v2_byte_prefix_exact_boundary_plus_one_does_not_skip() {
        let budget = SnapshotPageBudget::UdpCoalescing { target_bytes: 1200 };
        // Full frame: 119 + (188 + 352) + (188 + 353) = 1200.
        let exact = vec![message(1, 352), message(2, 353)];
        assert_eq!(budget.select_prefix(&exact, 50).unwrap(), (2, Some(1200)));
        let over = vec![message(1, 352), message(2, 354), message(3, 1)];
        assert_eq!(budget.select_prefix(&over, 50).unwrap(), (1, Some(659)));
        assert_eq!(
            SnapshotPageBudget::CountOnly
                .select_prefix(&over, 2)
                .unwrap(),
            (2, None)
        );
    }

    #[test]
    fn v2_byte_prefix_single_oversized_and_empty_are_preserved() {
        let budget = SnapshotPageBudget::UdpCoalescing { target_bytes: 1200 };
        let messages = vec![message(1, 2048), message(2, 1)];
        assert_eq!(
            budget.select_prefix(&messages, 50).unwrap(),
            (1, Some(2355))
        );
        assert_eq!(budget.select_prefix(&[], 50).unwrap(), (0, Some(119)));
        let page = PendingMessagePageV2 {
            messages: vec![message(1, 2048).1],
            next_cursor: vec![0; ENCODED_CURSOR_BYTES],
            has_more: true,
        };
        assert!(budget.validate(&page, Some(2355)).is_ok());
        let maximum = vec![message(1, 65_536), message(2, 1)];
        assert_eq!(
            budget.select_prefix(&maximum, 50).unwrap(),
            (1, Some(65_843))
        );
        let maximum_page = PendingMessagePageV2 {
            messages: vec![message(1, 65_536).1],
            next_cursor: vec![0; ENCODED_CURSOR_BYTES],
            has_more: true,
        };
        assert!(budget.validate(&maximum_page, Some(65_843)).is_ok());
    }

    #[test]
    fn v2_byte_prefix_checked_overflow_fails_closed() {
        assert!(checked_page_bytes(usize::MAX, 1).is_err());
        assert_eq!(checked_page_bytes(usize::MAX - 1, 1).unwrap(), usize::MAX);
    }

    #[test]
    fn v2_byte_prefix_actual_encoder_guard_rejects_drift() {
        let budget = SnapshotPageBudget::UdpCoalescing { target_bytes: 1200 };
        let mut page = PendingMessagePageV2 {
            messages: vec![message(1, 352).1, message(2, 353).1],
            next_cursor: vec![0; ENCODED_CURSOR_BYTES],
            has_more: true,
        };
        assert!(budget.validate(&page, Some(1200)).is_ok());
        assert!(budget.validate(&page, Some(1199)).is_err());
        page.next_cursor.push(0);
        assert!(budget.validate(&page, Some(1200)).is_err());
        // Even an accurate oversized estimate cannot admit a multi-item frame.
        assert!(budget.validate(&page, Some(1201)).is_err());
    }
}
