// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_message_dedup.rs
// ============================================
// Version: 1.0.0-BoundedOnlineDedupDomain
//
// Creation Reason:
//   [CHAT-ONLINE-DEDUP-DOMAIN 2026-08-28 by Codex] Extract concurrent
//   online-path message deduplication from the relay orchestration service.
//
// Main Functionality:
//   - Defines the online message deduplication capability boundary.
//   - Atomically admits exactly one first observer for a message identifier.
//   - Retains only a bounded, process-local approximation of recent IDs.
//   - Evicts the least-recently inserted observed entry after capacity growth.
//
// Dependencies:
//   - `dashmap` supplies atomic entry admission under concurrent callers.
//   - Standard atomics supply monotonic process-local insertion ordering.
//   - `chat_relay.rs` composes this capability without owning its container.
//
// Main Logical Flow:
//   1. Allocate a process-local insertion sequence for the observation.
//   2. Atomically reject an already occupied message identifier.
//   3. Insert the first observation with its sequence number.
//   4. If capacity was exceeded, remove the oldest retained observation.
//
// Important Note for Next Developer:
//   - This is an online-path optimization, not durable idempotency evidence.
//   - Keep admission atomic; a contains-then-insert sequence is race-prone.
//   - Do not store senders, receivers, payloads, routes, or timestamps here.
//   - Preserve bounded memory and the existing `true means duplicate` contract.
//   - Durable verified-submit and blind-route replay remain separate domains.
//
// Last Modified:
//   v1.0.0-BoundedOnlineDedupDomain - Initial capability extraction
// ============================================

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::{mapref::entry::Entry, DashMap};

/// Capability for process-local online message duplicate detection.
pub(crate) trait OnlineMessageDeduplication {
    /// Returns `true` when the message identifier was already retained.
    fn check_and_insert(&self, message_id: &[u8; 16]) -> bool;
}

/// Fixed-capacity concurrent online message deduplicator.
pub(crate) struct BoundedOnlineMessageDedup {
    observations: DashMap<[u8; 16], u64>,
    capacity: usize,
    sequence: AtomicU64,
}

impl BoundedOnlineMessageDedup {
    /// Creates an empty process-local deduplicator with the supplied capacity.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            observations: DashMap::with_capacity(capacity),
            capacity,
            sequence: AtomicU64::new(0),
        }
    }

    fn evict_oldest_if_over_capacity(&self) {
        if self.observations.len() <= self.capacity {
            return;
        }
        let oldest = self
            .observations
            .iter()
            .min_by_key(|entry| *entry.value())
            .map(|entry| *entry.key());
        if let Some(message_id) = oldest {
            self.observations.remove(&message_id);
        }
    }
}

impl OnlineMessageDeduplication for BoundedOnlineMessageDedup {
    fn check_and_insert(&self, message_id: &[u8; 16]) -> bool {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        match self.observations.entry(*message_id) {
            Entry::Occupied(_) => return true,
            Entry::Vacant(entry) => {
                entry.insert(sequence);
            }
        }

        self.evict_oldest_if_over_capacity();
        false
    }
}
