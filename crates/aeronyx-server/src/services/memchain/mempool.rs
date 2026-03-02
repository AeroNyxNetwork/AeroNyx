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
//! ## Main Functionality
//! - `MemPool::add_fact()` — validate hash integrity, dedup, store
//! - `MemPool::get()` — O(1) lookup by `fact_id`
//! - `MemPool::recent()` — return N most recent facts (by timestamp)
//! - `MemPool::drain_for_block()` — consume pending facts for block packing
//! - `MemPool::count()` — current pool size
//!
//! ## Thread Safety
//! Uses `dashmap::DashMap` for concurrent lock-free reads and
//! fine-grained shard-level writes. Safe to call from multiple
//! tokio tasks simultaneously.
//!
//! ## Dependencies
//! - `dashmap` (workspace)
//! - `aeronyx_core::ledger::Fact`
//!
//! ## ⚠️ Important Note for Next Developer
//! - Facts with invalid `fact_id` (hash mismatch) are **silently
//!   rejected** — this is intentional to avoid amplifying bad data.
//! - Deduplication is by `fact_id`; re-inserting the same fact is a no-op.
//! - `drain_for_block` takes ownership — facts are removed from the pool.
//!   Ensure they are persisted to AOF **before** draining.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemPool for MemChain integration

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use tracing::{debug, trace, warn};

use aeronyx_core::ledger::Fact;

// ============================================
// MemPool
// ============================================

/// Thread-safe in-memory buffer for recently received Facts.
///
/// Keyed by `fact_id: [u8; 32]`. Duplicate inserts are silently ignored.
pub struct MemPool {
    /// Primary storage: fact_id → Fact.
    facts: DashMap<[u8; 32], Fact>,
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

    /// Drains **all** facts from the pool and returns them.
    ///
    /// This is intended for the Miner to consume pending facts and
    /// pack them into a Checkpoint Block.
    ///
    /// # Warning
    /// Facts are **removed** from the pool. Ensure they have been
    /// persisted to the AOF **before** calling this.
    pub fn drain_for_block(&self) -> Vec<Fact> {
        let keys: Vec<[u8; 32]> = self.facts.iter().map(|e| *e.key()).collect();
        let mut drained = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some((_k, fact)) = self.facts.remove(&key) {
                drained.push(fact);
            }
        }
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
}
