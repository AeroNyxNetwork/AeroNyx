// ============================================
// File: crates/aeronyx-server/src/services/memchain/mempool.rs
// ============================================
//! # MemPool — In-Memory Fact Buffer
//!
//! ## Creation Reason
//! Holds recently received and verified Facts in memory, providing
//! fast lookups by `fact_id` and an ordered view for the future Miner
//! to batch-process into Checkpoint Blocks.
//!
//! ## Modification Reason
//! - 🌟 v0.4.0: Added insertion-ordered index (`order`) and
//!   `get_facts_after()` method to support P2P SyncRequest/SyncResponse.
//!   The `order` Vec tracks fact_ids in append order, enabling efficient
//!   "give me everything after hash X" queries.
//!
//! ## Main Functionality
//! - `MemPool::add_fact()` — validate hash integrity, dedup, store
//! - `MemPool::get()` — O(1) lookup by `fact_id`
//! - `MemPool::recent()` — return N most recent facts (by timestamp)
//! - `MemPool::get_facts_after()` — 🌟 NEW: return facts inserted after a given hash
//! - `MemPool::last_fact_id()` — 🌟 NEW: return the last inserted fact's id
//! - `MemPool::drain_for_block()` — consume pending facts for block packing
//! - `MemPool::count()` — current pool size
//!
//! ## Thread Safety
//! Uses `dashmap::DashMap` for concurrent lock-free reads and
//! fine-grained shard-level writes. The `order` Vec is protected by
//! `parking_lot::RwLock` for concurrent read access with exclusive writes.
//!
//! ## Dependencies
//! - `dashmap` (workspace)
//! - `parking_lot` (workspace)
//! - `aeronyx_core::ledger::Fact`
//!
//! ## ⚠️ Important Note for Next Developer
//! - Facts with invalid `fact_id` (hash mismatch) are **silently
//!   rejected** — this is intentional to avoid amplifying bad data.
//! - Deduplication is by `fact_id`; re-inserting the same fact is a no-op.
//! - `drain_for_block` takes ownership — facts are removed from the pool
//!   AND from the order index. Ensure they are persisted to AOF **before** draining.
//! - `get_facts_after` uses the insertion order, NOT timestamp order.
//!   This is correct for sync because insertion order reflects network arrival.
//! - `[0u8; 32]` (all-zero hash) is treated as "no known hash" by convention,
//!   causing `get_facts_after` to return everything.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemPool for MemChain integration
//! v0.4.0 - 🌟 Added insertion-order tracking + get_facts_after for P2P sync

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, trace, warn};

use aeronyx_core::ledger::Fact;

// ============================================
// Constants
// ============================================

/// Sentinel hash meaning "I have no history, give me everything".
/// Used by SyncRequest when a node is brand-new.
const ZERO_HASH: [u8; 32] = [0u8; 32];

// ============================================
// MemPool
// ============================================

/// Thread-safe in-memory buffer for recently received Facts.
///
/// Keyed by `fact_id: [u8; 32]`. Duplicate inserts are silently ignored.
///
/// Maintains a separate insertion-ordered index (`order`) so that
/// `get_facts_after()` can efficiently produce the "delta" a peer
/// is missing during P2P synchronisation.
pub struct MemPool {
    /// Primary storage: fact_id → Fact.
    facts: DashMap<[u8; 32], Fact>,
    /// 🌟 Insertion-ordered list of fact_ids.
    /// Protected by RwLock: many concurrent readers, exclusive writer.
    order: RwLock<Vec<[u8; 32]>>,
    /// Monotonic counter of total facts ever accepted (for metrics).
    total_accepted: AtomicU64,
    /// Monotonic counter of total facts rejected (for metrics).
    total_rejected: AtomicU64,
}

impl MemPool {
    /// Creates a new, empty MemPool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            facts: DashMap::new(),
            order: RwLock::new(Vec::new()),
            total_accepted: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
        }
    }

    /// Validates and adds a Fact to the pool.
    ///
    /// # Validation
    /// 1. `fact_id` must match the SHA-256 of canonical content fields.
    /// 2. Duplicate `fact_id` is silently ignored (idempotent).
    ///
    /// # Returns
    /// - `true` if the fact was newly inserted.
    /// - `false` if it was a duplicate or failed validation.
    ///
    /// # Note
    /// Signature verification is NOT done here — it requires access to
    /// the origin node's public key, which is the caller's responsibility.
    /// This method only checks content-hash integrity.
    pub fn add_fact(&self, fact: Fact) -> bool {
        // Validate content hash
        if !fact.verify_id() {
            warn!(
                fact_id = hex::encode(fact.fact_id),
                "[MEMPOOL] ❌ Rejected fact: hash mismatch"
            );
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Dedup check + insert
        let fact_id = fact.fact_id;
        if self.facts.contains_key(&fact_id) {
            trace!(
                fact_id = hex::encode(fact_id),
                "[MEMPOOL] Duplicate fact, skipping"
            );
            return false;
        }

        self.facts.insert(fact_id, fact);

        // 🌟 Track insertion order
        {
            let mut order = self.order.write();
            order.push(fact_id);
        }

        self.total_accepted.fetch_add(1, Ordering::Relaxed);

        debug!(
            fact_id = hex::encode(fact_id),
            pool_size = self.facts.len(),
            "[MEMPOOL] ✅ Fact accepted"
        );

        true
    }

    /// Looks up a Fact by its `fact_id`.
    ///
    /// # Returns
    /// `Some(Fact)` if found, `None` otherwise.
    #[must_use]
    pub fn get(&self, fact_id: &[u8; 32]) -> Option<Fact> {
        self.facts.get(fact_id).map(|entry| entry.value().clone())
    }

    /// Returns up to `n` most recent facts, ordered by timestamp descending.
    ///
    /// This performs a full scan + sort, so avoid calling on very large pools
    /// in hot paths. Suitable for API queries and Miner batches.
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<Fact> {
        let mut all: Vec<Fact> = self.facts.iter().map(|e| e.value().clone()).collect();
        all.sort_unstable_by(|a, b| b.timestamp.cmp(&a.timestamp));
        all.truncate(n);
        all
    }

    /// 🌟 Returns all Facts inserted **after** the given `last_hash`.
    ///
    /// # Arguments
    /// * `last_hash` - The `fact_id` of the last known fact on the requester's
    ///   chain. Facts inserted after this one are returned.
    ///   If `[0u8; 32]` (all-zero sentinel), returns **all** facts.
    ///
    /// # Returns
    /// Vector of Facts in insertion order (oldest first), representing the
    /// delta the requester needs to catch up.
    ///
    /// If `last_hash` is not found in the pool (e.g. it was drained into a
    /// block, or the requester has a corrupted hash), returns **all** facts
    /// as a safe fallback — the requester's dedup will handle duplicates.
    #[must_use]
    pub fn get_facts_after(&self, last_hash: [u8; 32]) -> Vec<Fact> {
        let order = self.order.read();

        // All-zero sentinel or empty pool → return everything
        if last_hash == ZERO_HASH || order.is_empty() {
            return order
                .iter()
                .filter_map(|id| self.facts.get(id).map(|e| e.value().clone()))
                .collect();
        }

        // Find the position of last_hash
        let pos = order.iter().position(|id| *id == last_hash);

        match pos {
            Some(idx) => {
                // Return everything AFTER this position
                order[idx + 1..]
                    .iter()
                    .filter_map(|id| self.facts.get(id).map(|e| e.value().clone()))
                    .collect()
            }
            None => {
                // Hash not found — return everything as safe fallback
                debug!(
                    last_hash = hex::encode(last_hash),
                    "[MEMPOOL] last_hash not found in order index, returning all facts"
                );
                order
                    .iter()
                    .filter_map(|id| self.facts.get(id).map(|e| e.value().clone()))
                    .collect()
            }
        }
    }

    /// 🌟 Returns the `fact_id` of the most recently inserted Fact.
    ///
    /// Returns `None` if the pool is empty, or `Some([0u8; 32])` would
    /// never be returned (zero-hash is a sentinel, not a real fact).
    #[must_use]
    pub fn last_fact_id(&self) -> Option<[u8; 32]> {
        let order = self.order.read();
        order.last().copied()
    }

    /// Drains **all** facts from the pool and returns them.
    ///
    /// This is intended for the Miner to consume pending facts and
    /// pack them into a Checkpoint Block.
    ///
    /// # Warning
    /// Facts are **removed** from the pool AND the order index.
    /// Ensure they have been persisted to the AOF **before** calling this.
    pub fn drain_for_block(&self) -> Vec<Fact> {
        // 🌟 Clear order index first
        let keys: Vec<[u8; 32]> = {
            let mut order = self.order.write();
            let keys = order.clone();
            order.clear();
            keys
        };

        let mut drained = Vec::with_capacity(keys.len());
        for key in &keys {
            if let Some((_k, fact)) = self.facts.remove(key) {
                drained.push(fact);
            }
        }
        // Maintain timestamp order for block packing
        drained.sort_unstable_by_key(|f| f.timestamp);
        debug!(
            count = drained.len(),
            "[MEMPOOL] Drained facts for block packing"
        );
        drained
    }

    /// Returns the current number of facts in the pool.
    #[must_use]
    pub fn count(&self) -> usize {
        self.facts.len()
    }

    /// Returns total facts ever accepted (monotonic counter).
    #[must_use]
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted.load(Ordering::Relaxed)
    }

    /// Returns total facts ever rejected (monotonic counter).
    #[must_use]
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected.load(Ordering::Relaxed)
    }
}

impl Default for MemPool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemPool")
            .field("count", &self.count())
            .field("total_accepted", &self.total_accepted())
            .field("total_rejected", &self.total_rejected())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fact(ts: u64, subject: &str) -> Fact {
        Fact::new(ts, subject.into(), "pred".into(), "obj".into())
    }

    #[test]
    fn test_add_and_get() {
        let pool = MemPool::new();
        let fact = make_fact(100, "test");
        let id = fact.fact_id;

        assert!(pool.add_fact(fact.clone()), "First insert should succeed");
        assert!(!pool.add_fact(fact), "Duplicate should return false");

        let retrieved = pool.get(&id).expect("Should find fact");
        assert_eq!(retrieved.subject, "test");
        assert_eq!(pool.count(), 1);
    }

    #[test]
    fn test_reject_invalid_hash() {
        let pool = MemPool::new();
        let mut fact = make_fact(100, "test");
        fact.fact_id = [0xFF; 32]; // Invalid hash

        assert!(!pool.add_fact(fact), "Invalid hash should be rejected");
        assert_eq!(pool.count(), 0);
        assert_eq!(pool.total_rejected(), 1);
    }

    #[test]
    fn test_recent_ordering() {
        let pool = MemPool::new();
        pool.add_fact(make_fact(100, "old"));
        pool.add_fact(make_fact(300, "new"));
        pool.add_fact(make_fact(200, "mid"));

        let recent = pool.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].subject, "new");
        assert_eq!(recent[1].subject, "mid");
    }

    #[test]
    fn test_drain_for_block() {
        let pool = MemPool::new();
        pool.add_fact(make_fact(200, "b"));
        pool.add_fact(make_fact(100, "a"));

        let drained = pool.drain_for_block();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].subject, "a", "Should be sorted by timestamp asc");
        assert_eq!(drained[1].subject, "b");
        assert_eq!(pool.count(), 0, "Pool should be empty after drain");
    }

    // ========================================
    // 🌟 New tests for v0.4.0
    // ========================================

    #[test]
    fn test_get_facts_after_zero_hash() {
        let pool = MemPool::new();
        pool.add_fact(make_fact(100, "a"));
        pool.add_fact(make_fact(200, "b"));

        // Zero hash = give me everything
        let all = pool.get_facts_after([0u8; 32]);
        assert_eq!(all.len(), 2);
        // Insertion order: a first, b second
        assert_eq!(all[0].subject, "a");
        assert_eq!(all[1].subject, "b");
    }

    #[test]
    fn test_get_facts_after_known_hash() {
        let pool = MemPool::new();
        let fact_a = make_fact(100, "a");
        let hash_a = fact_a.fact_id;
        pool.add_fact(fact_a);
        pool.add_fact(make_fact(200, "b"));
        pool.add_fact(make_fact(300, "c"));

        // "I have up to A" → should get B and C
        let delta = pool.get_facts_after(hash_a);
        assert_eq!(delta.len(), 2);
        assert_eq!(delta[0].subject, "b");
        assert_eq!(delta[1].subject, "c");
    }

    #[test]
    fn test_get_facts_after_last_hash() {
        let pool = MemPool::new();
        let fact_c = make_fact(300, "c");
        let hash_c = fact_c.fact_id;
        pool.add_fact(make_fact(100, "a"));
        pool.add_fact(make_fact(200, "b"));
        pool.add_fact(fact_c);

        // "I already have C" (the latest) → empty delta
        let delta = pool.get_facts_after(hash_c);
        assert!(delta.is_empty());
    }

    #[test]
    fn test_get_facts_after_unknown_hash() {
        let pool = MemPool::new();
        pool.add_fact(make_fact(100, "a"));
        pool.add_fact(make_fact(200, "b"));

        // Unknown hash → return everything (safe fallback)
        let all = pool.get_facts_after([0xFF; 32]);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_last_fact_id() {
        let pool = MemPool::new();
        assert!(pool.last_fact_id().is_none());

        let fact_a = make_fact(100, "a");
        let hash_a = fact_a.fact_id;
        pool.add_fact(fact_a);
        assert_eq!(pool.last_fact_id(), Some(hash_a));

        let fact_b = make_fact(200, "b");
        let hash_b = fact_b.fact_id;
        pool.add_fact(fact_b);
        assert_eq!(pool.last_fact_id(), Some(hash_b));
    }

    #[test]
    fn test_drain_clears_order() {
        let pool = MemPool::new();
        pool.add_fact(make_fact(100, "a"));
        pool.add_fact(make_fact(200, "b"));

        let _ = pool.drain_for_block();

        assert!(pool.last_fact_id().is_none());
        assert!(pool.get_facts_after([0u8; 32]).is_empty());
    }
}
