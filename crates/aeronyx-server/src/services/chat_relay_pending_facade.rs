// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_facade.rs
// ============================================
// Version: 1.3.0-HttpBoundedSnapshotPaging
//
// Creation Reason:
//   [CHAT-PENDING-FACADE-DOMAIN 2026-08-28 by Codex] Move offline-message
//   custody, delivery, quarantine telemetry, and acknowledgement APIs out of
//   the relay composition root without widening service field visibility.
//
// Modification Reason:
//   [CHAT-HTTP-V2-BYTE-PAGING 2026-10-03 by Codex] Expose a separate strict
//   HTTP encoded-page budget, keeping existing public and UDP pull semantics.
//   [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Add an internal transport-budgeted
//   snapshot path without changing the public count-based pull contract.
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
//   v1.3.0-HttpBoundedSnapshotPaging - 2026-10-03, HTTP codec-bounded pull API
//   v1.2.0-ByteAwareSnapshotPaging - 2026-10-03, internal budgeted snapshot path
//   v1.1.0-PendingCursorTestSeam - Co-located test-only cursor decoding
//   v1.0.0-PendingMessageFacade - Initial pending-message facade extraction
// ============================================

use aeronyx_core::protocol::chat::ChatEnvelope;
use tracing::{debug, warn};

use crate::services::chat_relay_pending_contract::{PendingMessage, PendingMessagePageV2};
use crate::services::chat_relay_pending_custody::PendingMessageStoreOutcome;
use crate::services::chat_relay_pending_delivery::{
    PendingPullQuarantineSummary, SnapshotPageBudget,
};
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

    // [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Prefix selection and cursor
    // protection stay together in the delivery domain, not in the UDP handler.
    pub(crate) fn pull_pending_v2_coalesced(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded_cursor: &[u8],
        limit: u32,
        target_bytes: usize,
    ) -> ChatRelayResult<PendingMessagePageV2> {
        let delivery = self.pending_delivery.pull_snapshot_budgeted(
            &self.conn,
            &self.durable_quarantine,
            receiver,
            after_timestamp,
            encoded_cursor,
            limit,
            SnapshotPageBudget::UdpCoalescing { target_bytes },
        )?;
        self.record_pending_pull_quarantine(delivery.quarantine);
        Ok(delivery.page)
    }

    /// Retrieves a whole ordered HTTP v2 prefix within the MemChain codec bound.
    ///
    /// [CHAT-HTTP-V2-BYTE-PAGING 2026-10-03 by Codex] Includes response fields
    /// and the protected cursor in the 2 MiB + discriminator raw byte ceiling.
    /// Selection precedes cursor protection; callers must not truncate the page.
    /// This synchronous repository boundary must run off async worker threads.
    ///
    /// # Errors
    ///
    /// Returns the existing cursor/storage errors, or a coarse serialization
    /// error if even the first whole item cannot fit. Valid pending rows remain
    /// stored; no oversized-first exception, ACK, or skip is implied by HTTP.
    pub(crate) fn pull_pending_v2_http(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded_cursor: &[u8],
        limit: u32,
    ) -> ChatRelayResult<PendingMessagePageV2> {
        let delivery = self.pending_delivery.pull_snapshot_budgeted(
            &self.conn,
            &self.durable_quarantine,
            receiver,
            after_timestamp,
            encoded_cursor,
            limit,
            SnapshotPageBudget::HttpEncoded,
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

// [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Real private SQLite fixtures:
// use the existing signing/cursor/custody path, no public test-only API.
#[cfg(test)]
mod byte_paging_tests {
    use super::*;
    use crate::config::ChatRelayConfig;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::ChatContentType;

    fn open(path: &std::path::Path) -> ChatRelayService {
        ChatRelayService::new(
            ChatRelayConfig {
                enabled: true,
                db_path: path.to_string_lossy().into_owned(),
                ..ChatRelayConfig::default()
            },
            [0x91; 32],
        )
        .unwrap()
    }

    fn store(service: &ChatRelayService, id: u8, bytes: usize) -> ChatEnvelope {
        let signer = IdentityKeyPair::generate();
        let mut envelope = ChatEnvelope {
            message_id: [id; 16],
            sender: signer.public_key_bytes(),
            receiver: [0x92; 32],
            timestamp: now_secs(),
            ciphertext: vec![id; bytes],
            nonce: [id; 24],
            content_type: ChatContentType::Text,
            signature: [0; 64],
        };
        envelope.signature = signer.sign(&envelope.sign_data());
        service.store_pending(&envelope).unwrap();
        envelope
    }

    #[test]
    fn v2_byte_snapshot_ack_restart_preserves_ceiling_and_unsent_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custody.sqlite3");
        let service = open(&path);
        for id in 1..=4 {
            store(&service, id, 400);
        }
        let first = service
            .pull_pending_v2_coalesced(&[0x92; 32], 0, &[], 50, 1200)
            .unwrap();
        assert_eq!(first.messages.len(), 1);
        assert!(first.has_more);
        let decoded = service
            .decode_pull_cursor_v2(&[0x92; 32], 0, &first.next_cursor)
            .unwrap();
        assert_eq!((decoded.position, decoded.ceiling), (1, 4));
        store(&service, 5, 400);
        assert_eq!(service.ack_messages(&[[1; 16]], &[0x92; 32]).unwrap(), 1);
        let mut cursor = first.next_cursor;
        drop(service);
        let service = open(&path);
        for id in 2..=4 {
            let page = service
                .pull_pending_v2_coalesced(&[0x92; 32], 0, &cursor, 50, 1200)
                .unwrap();
            assert_eq!(page.messages.len(), 1);
            assert!(page.messages[0].message_id == [id; 16]);
            assert_eq!(page.has_more, id < 4);
            let decoded = service
                .decode_pull_cursor_v2(&[0x92; 32], 0, &page.next_cursor)
                .unwrap();
            assert_eq!((decoded.position, decoded.ceiling), (u64::from(id), 4));
            assert_eq!(service.ack_messages(&[[id; 16]], &[0x92; 32]).unwrap(), 1);
            cursor = page.next_cursor;
        }
        let fresh = service
            .pull_pending_v2_coalesced(&[0x92; 32], 0, &[], 50, 1200)
            .unwrap();
        assert_eq!(fresh.messages.len(), 1);
        assert!(fresh.messages[0].message_id == [5; 16]);
        assert!(!fresh.has_more);
    }

    #[test]
    fn v2_byte_corrupt_only_page_advances_bounded_cursor() {
        let dir = tempfile::tempdir().unwrap();
        let service = open(&dir.path().join("custody.sqlite3"));
        for id in 1..=4 {
            store(&service, id, 400);
        }
        service
            .conn
            .lock()
            .execute(
                "UPDATE pending_messages SET sender = zeroblob(32) WHERE queue_sequence <= 3",
                [],
            )
            .unwrap();
        // limit=2 reads only three raw rows, quarantines them, then continues.
        let first = service
            .pull_pending_v2_coalesced(&[0x92; 32], 0, &[], 2, 1200)
            .unwrap();
        assert!(first.messages.is_empty() && first.has_more);
        let decoded = service
            .decode_pull_cursor_v2(&[0x92; 32], 0, &first.next_cursor)
            .unwrap();
        assert_eq!((decoded.position, decoded.ceiling), (3, 4));
        assert_eq!(service.storage_usage().unwrap().pending_messages, 1);
        let last = service
            .pull_pending_v2_coalesced(&[0x92; 32], 0, &first.next_cursor, 2, 1200)
            .unwrap();
        assert_eq!(last.messages.len(), 1);
        assert!(last.messages[0].message_id == [4; 16]);
        assert!(!last.has_more);
    }

    #[test]
    fn v2_byte_corrupt_tail_never_skips_unselected_valid_row() {
        let dir = tempfile::tempdir().unwrap();
        let service = open(&dir.path().join("custody.sqlite3"));
        for id in 1..=4 {
            store(&service, id, 400);
        }
        service
            .conn
            .lock()
            .execute(
                "UPDATE pending_messages SET sender = zeroblob(32) WHERE queue_sequence IN (2,4)",
                [],
            )
            .unwrap();
        let first = service
            .pull_pending_v2_coalesced(&[0x92; 32], 0, &[], 50, 1200)
            .unwrap();
        assert_eq!(first.messages.len(), 1);
        assert!(first.has_more);
        let decoded = service
            .decode_pull_cursor_v2(&[0x92; 32], 0, &first.next_cursor)
            .unwrap();
        assert_eq!((decoded.position, decoded.ceiling), (1, 4));
        let last = service
            .pull_pending_v2_coalesced(&[0x92; 32], 0, &first.next_cursor, 50, 1200)
            .unwrap();
        assert_eq!(last.messages.len(), 1);
        assert!(last.messages[0].message_id == [3; 16]);
        assert!(!last.has_more);
    }

    #[test]
    fn v2_byte_public_count_only_and_single_oversized_remain_compatible() {
        let dir = tempfile::tempdir().unwrap();
        let service = open(&dir.path().join("custody.sqlite3"));
        let large = store(&service, 1, 2048);
        store(&service, 2, 10);
        let count_page = service.pull_pending_v2(&[0x92; 32], 0, &[], 50).unwrap();
        assert_eq!(count_page.messages.len(), 2);
        assert!(!count_page.has_more);
        let page = service
            .pull_pending_v2_coalesced(&[0x92; 32], 0, &[], 50, 1200)
            .unwrap();
        assert_eq!(page.messages.len(), 1);
        assert!(page.has_more);
        assert!(page.messages[0].envelope.ciphertext == large.ciphertext);
        let decoded = service
            .decode_pull_cursor_v2(&[0x92; 32], 0, &page.next_cursor)
            .unwrap();
        assert_eq!((decoded.position, decoded.ceiling), (1, 2));
        assert_eq!(service.storage_usage().unwrap().pending_messages, 2);
    }

    // [CHAT-HTTP-V2-BYTE-PAGING 2026-10-03 by Codex] Exercise the real codec
    // on the returned cursor and whole signed envelopes, not a size estimate.
    fn http_encoded_len(page: &PendingMessagePageV2) -> Result<usize, bincode::Error> {
        use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
        encode_memchain(&MemChainMessage::ChatPullResponseV2 {
            envelopes: page
                .messages
                .iter()
                .map(|message| message.envelope.clone())
                .collect(),
            next_cursor: page.next_cursor.clone(),
            has_more: page.has_more,
        })
        .map(|encoded| encoded.len())
    }

    #[test]
    fn http_v2_large_count_page_becomes_bounded_prefixes_across_ack_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custody.sqlite3");
        let service = open(&path);
        for id in 1..=100 {
            store(&service, id, 65_536);
        }
        let old = service.pull_pending_v2(&[0x92; 32], 0, &[], 100).unwrap();
        assert_eq!(old.messages.len(), 100);
        // Reproduce the old HTTP failure with an entirely legal retained page.
        assert!(http_encoded_len(&old).is_err());
        drop(old);
        let first = service
            .pull_pending_v2_http(&[0x92; 32], 0, &[], 100)
            .unwrap();
        assert_eq!(first.messages.len(), 31);
        assert_eq!(http_encoded_len(&first).unwrap(), 2_037_523);
        assert!(first.has_more);
        let decoded = service
            .decode_pull_cursor_v2(&[0x92; 32], 0, &first.next_cursor)
            .unwrap();
        assert_eq!((decoded.position, decoded.ceiling), (31, 100));
        for (index, message) in first.messages.iter().enumerate() {
            assert!(message.message_id == [index as u8 + 1; 16]);
            assert_eq!(message.envelope.ciphertext.len(), 65_536);
        }
        // A lost HTTP response has no ACK side effect; the same input returns
        // the same envelope prefix/ceiling (cursor encryption may use a new nonce).
        let retry = service
            .pull_pending_v2_http(&[0x92; 32], 0, &[], 100)
            .unwrap();
        assert!(first
            .messages
            .iter()
            .zip(&retry.messages)
            .all(|(a, b)| a.message_id == b.message_id));
        assert_eq!(retry.messages.len(), 31);
        assert_eq!(service.storage_usage().unwrap().pending_messages, 100);
        store(&service, 101, 1);
        let ids: Vec<_> = first
            .messages
            .iter()
            .map(|message| message.message_id)
            .collect();
        assert_eq!(service.ack_messages(&ids, &[0x92; 32]).unwrap(), 31);
        let mut cursor = first.next_cursor;
        drop(service);
        let service = open(&path);
        for (start, end) in [(32u8, 62u8), (63, 93), (94, 100)] {
            let page = service
                .pull_pending_v2_http(&[0x92; 32], 0, &cursor, 100)
                .unwrap();
            assert_eq!(page.messages.len(), usize::from(end - start + 1));
            assert!(http_encoded_len(&page).unwrap() <= 2_097_153);
            for (id, message) in (start..=end).zip(&page.messages) {
                assert!(message.message_id == [id; 16]);
                assert_eq!(message.envelope.ciphertext.len(), 65_536);
            }
            let decoded = service
                .decode_pull_cursor_v2(&[0x92; 32], 0, &page.next_cursor)
                .unwrap();
            assert_eq!((decoded.position, decoded.ceiling), (u64::from(end), 100));
            assert_eq!(page.has_more, end < 100);
            let ids: Vec<_> = page
                .messages
                .iter()
                .map(|message| message.message_id)
                .collect();
            assert_eq!(service.ack_messages(&ids, &[0x92; 32]).unwrap(), ids.len());
            cursor = page.next_cursor;
        }
        let exhausted = service
            .pull_pending_v2_http(&[0x92; 32], 0, &cursor, 100)
            .unwrap();
        assert!(exhausted.messages.is_empty() && !exhausted.has_more);
        assert_eq!(http_encoded_len(&exhausted).unwrap(), 79);
        let fresh = service
            .pull_pending_v2_http(&[0x92; 32], 0, &[], 100)
            .unwrap();
        assert_eq!(fresh.messages.len(), 1);
        assert!(fresh.messages[0].message_id == [101; 16]);
        assert_eq!(service.storage_usage().unwrap().pending_messages, 1);
    }

    #[test]
    fn http_v2_byte_limited_cursor_never_skips_valid_rows_among_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let service = open(&dir.path().join("custody.sqlite3"));
        for id in 1..=35 {
            store(&service, id, 65_536);
        }
        service
            .conn
            .lock()
            .execute(
                "UPDATE pending_messages SET sender = zeroblob(32) WHERE queue_sequence IN (2,34)",
                [],
            )
            .unwrap();
        let first = service
            .pull_pending_v2_http(&[0x92; 32], 0, &[], 100)
            .unwrap();
        assert_eq!(first.messages.len(), 31);
        assert!(first.has_more);
        let decoded = service
            .decode_pull_cursor_v2(&[0x92; 32], 0, &first.next_cursor)
            .unwrap();
        assert_eq!((decoded.position, decoded.ceiling), (32, 35));
        assert_eq!(service.storage_usage().unwrap().pending_messages, 33);
        let last = service
            .pull_pending_v2_http(&[0x92; 32], 0, &first.next_cursor, 100)
            .unwrap();
        assert_eq!(last.messages.len(), 2);
        assert!(last.messages[0].message_id == [33; 16]);
        assert!(last.messages[1].message_id == [35; 16]);
        assert!(!last.has_more);
        assert!(http_encoded_len(&last).unwrap() <= 2_097_153);
    }

    #[test]
    fn http_v2_corrupt_only_bounded_scan_keeps_forward_progress() {
        let dir = tempfile::tempdir().unwrap();
        let service = open(&dir.path().join("custody.sqlite3"));
        for id in 1..=4 {
            store(&service, id, 1);
        }
        service
            .conn
            .lock()
            .execute(
                "UPDATE pending_messages SET sender = zeroblob(32) WHERE queue_sequence <= 3",
                [],
            )
            .unwrap();
        let first = service
            .pull_pending_v2_http(&[0x92; 32], 0, &[], 2)
            .unwrap();
        assert!(first.messages.is_empty() && first.has_more);
        assert_eq!(http_encoded_len(&first).unwrap(), 79);
        let decoded = service
            .decode_pull_cursor_v2(&[0x92; 32], 0, &first.next_cursor)
            .unwrap();
        assert_eq!((decoded.position, decoded.ceiling), (3, 4));
        let last = service
            .pull_pending_v2_http(&[0x92; 32], 0, &first.next_cursor, 2)
            .unwrap();
        assert_eq!(last.messages.len(), 1);
        assert!(last.messages[0].message_id == [4; 16]);
        assert!(!last.has_more);
    }
}
