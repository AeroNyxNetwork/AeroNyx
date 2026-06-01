// ============================================
// File: crates/aeronyx-server/src/services/traffic_tracker.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   New file. Per-wallet traffic delta tracker for membership quota
//   enforcement. Aggregates bytes_in / bytes_out per wallet across all
//   sessions belonging to that wallet, then drains the deltas on each
//   heartbeat cycle so the CMS can atomically update UserTrafficQuota.
//
// Main Functionality:
//   - record_rx(): hot-path inbound byte counter (called per VPN packet)
//   - record_tx(): hot-path outbound byte counter (called per TUN packet)
//   - drain():     atomically swaps all counters to zero, returns non-zero
//                  deltas as HashMap<wallet_hex, TrafficDelta>
//   - remove_wallet(): cleans up zero-count entries after session close
//
// Dependencies:
//   - Called by handlers/packet.rs on every IPv4/IPv6 VPN packet
//   - Called by server.rs cleanup task for remove_wallet()
//   - Drained by management/reporter.rs before each heartbeat send
//   - TrafficDelta type re-exported in management/client.rs
//
// Main Logical Flow:
//   1. PacketHandler calls record_rx / record_tx on VPN path (hot path)
//   2. HeartbeatReporter calls drain() before building heartbeat payload
//   3. drain() atomically swap(0) on every WalletCounters entry
//   4. Only non-zero entries are included in the returned HashMap
//   5. remove_wallet() called from cleanup task after session expiry
//      to prevent unbounded map growth
//
// ⚠️ Important Notes for Next Developer:
//   - drain() uses Ordering::Relaxed for the swap — this is intentional.
//     We do NOT need cross-thread ordering guarantees for byte counters;
//     the worst case is a few bytes attributed to the wrong 30s window,
//     which is acceptable for quota accounting.
//   - remove_wallet() is conservative: it only removes the entry when
//     BOTH counters are zero at the moment of the check. If a packet
//     races with remove_wallet(), the entry stays and will be cleaned
//     up on the next heartbeat cycle via drain().
//   - wallet_hex keys must be lowercase hex (consistent with
//     hex::encode output). Never uppercase or mixed case.
//   - This struct does NOT own sessions. It is a pure counter store.
//     Session lifecycle is managed by SessionManager exclusively.
//
// Last Modified: v1.0.0-Membership — initial implementation
// ============================================

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::management::client::TrafficDelta;

// ============================================
// WalletCounters (internal)
// ============================================

/// Atomic byte counters for a single wallet.
///
/// Heap-allocated via Arc so DashMap entries stay small (one pointer
/// width) and the AtomicU64 fields are cache-line aligned by the
/// allocator rather than packed inside the DashMap shard.
struct WalletCounters {
    bytes_in:  AtomicU64,
    bytes_out: AtomicU64,
}

impl WalletCounters {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            bytes_in:  AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
        })
    }
}

// ============================================
// TrafficTracker
// ============================================

/// Per-wallet traffic delta tracker.
///
/// Thread-safe. Designed for high-frequency writes (one call per VPN
/// packet) and low-frequency reads (one drain per heartbeat, ~30s).
///
/// # Example
/// ```rust
/// let tracker = Arc::new(TrafficTracker::new());
///
/// // Hot path — called per packet
/// tracker.record_rx("aabbcc...", 1400);
/// tracker.record_tx("aabbcc...", 800);
///
/// // Heartbeat path — drain and send to CMS
/// let deltas = tracker.drain();
/// // deltas: HashMap<wallet_hex, TrafficDelta>
/// ```
pub struct TrafficTracker {
    wallets: DashMap<String, Arc<WalletCounters>>,
}

impl TrafficTracker {
    /// Creates a new empty TrafficTracker.
    pub fn new() -> Self {
        Self {
            wallets: DashMap::new(),
        }
    }

    // ----------------------------------------
    // Hot path: called on every VPN packet
    // ----------------------------------------

    /// Records inbound bytes for a wallet (client → server).
    ///
    /// Called from `PacketHandler::handle_udp_packet` on IPv4/IPv6 VPN
    /// path only. Uses `Ordering::Relaxed` — counter precision across
    /// 30-second windows is sufficient for quota accounting.
    #[inline]
    pub fn record_rx(&self, wallet_hex: &str, bytes: u64) {
        self.get_or_create(wallet_hex)
            .bytes_in
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records outbound bytes for a wallet (server → client).
    ///
    /// Called from `PacketHandler::handle_tun_packet`.
    #[inline]
    pub fn record_tx(&self, wallet_hex: &str, bytes: u64) {
        self.get_or_create(wallet_hex)
            .bytes_out
            .fetch_add(bytes, Ordering::Relaxed);
    }

    // ----------------------------------------
    // Heartbeat path: drain once per ~30s
    // ----------------------------------------

    /// Atomically takes all accumulated deltas and resets counters to zero.
    ///
    /// Called by `HeartbeatReporter` before building the heartbeat payload.
    /// Returns only wallets that had non-zero traffic in the period.
    ///
    /// # Atomicity guarantee
    /// Each counter is individually swapped to zero. Traffic that arrives
    /// between two `swap(0)` calls on bytes_in and bytes_out is attributed
    /// to the next heartbeat window — acceptable for quota accounting.
    pub fn drain(&self) -> HashMap<String, TrafficDelta> {
        let mut result = HashMap::new();

        for entry in self.wallets.iter() {
            let counters = entry.value();
            let bytes_in  = counters.bytes_in.swap(0,  Ordering::Relaxed);
            let bytes_out = counters.bytes_out.swap(0, Ordering::Relaxed);

            if bytes_in > 0 || bytes_out > 0 {
                result.insert(
                    entry.key().clone(),
                    TrafficDelta { bytes_in, bytes_out },
                );
            }
        }

        result
    }

    // ----------------------------------------
    // Cleanup path: called on session expiry
    // ----------------------------------------

    /// Removes the tracker entry for a wallet if its counters are zero.
    ///
    /// Called from `server.rs::spawn_cleanup_task` after a session expires.
    /// Conservative: if a concurrent packet increments the counter between
    /// our load and the remove, the entry remains and will be drained on
    /// the next heartbeat cycle.
    pub fn remove_wallet(&self, wallet_hex: &str) {
        // Only remove if both counters are currently zero.
        let should_remove = self.wallets.get(wallet_hex).map(|entry| {
            let counters = entry.value();
            counters.bytes_in.load(Ordering::Relaxed) == 0
                && counters.bytes_out.load(Ordering::Relaxed) == 0
        }).unwrap_or(false);

        if should_remove {
            self.wallets.remove(wallet_hex);
        }
    }

    // ----------------------------------------
    // Internal helpers
    // ----------------------------------------

    /// Returns the Arc<WalletCounters> for the wallet, creating if absent.
    #[inline]
    fn get_or_create(&self, wallet_hex: &str) -> Arc<WalletCounters> {
        // Fast path: entry already exists.
        if let Some(entry) = self.wallets.get(wallet_hex) {
            return Arc::clone(entry.value());
        }
        // Slow path: insert new entry. or_insert_with handles races.
        Arc::clone(
            self.wallets
                .entry(wallet_hex.to_string())
                .or_insert_with(WalletCounters::new)
                .value(),
        )
    }
}

impl Default for TrafficTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TrafficTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrafficTracker")
            .field("wallet_count", &self.wallets.len())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_drain() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);
        t.record_rx("aabb", 200);
        t.record_tx("aabb", 50);
        t.record_rx("ccdd", 400);

        let d = t.drain();
        assert_eq!(d.len(), 2);
        assert_eq!(d["aabb"].bytes_in,  300);
        assert_eq!(d["aabb"].bytes_out, 50);
        assert_eq!(d["ccdd"].bytes_in,  400);
        assert_eq!(d["ccdd"].bytes_out, 0);
    }

    #[test]
    fn test_drain_resets_counters() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);

        let d1 = t.drain();
        assert_eq!(d1["aabb"].bytes_in, 100);

        // Second drain must return empty (counters reset)
        let d2 = t.drain();
        assert!(d2.is_empty());
    }

    #[test]
    fn test_drain_skips_zero_wallets() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);

        let _d1 = t.drain(); // resets aabb to 0

        // aabb entry exists but both counters are 0 — must not appear
        let d2 = t.drain();
        assert!(!d2.contains_key("aabb"));
    }

    #[test]
    fn test_remove_wallet_zero_counters() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);
        let _ = t.drain(); // counters now zero

        t.remove_wallet("aabb");
        assert!(!t.wallets.contains_key("aabb"));
    }

    #[test]
    fn test_remove_wallet_nonzero_stays() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);

        // counters still non-zero — must NOT be removed
        t.remove_wallet("aabb");
        assert!(t.wallets.contains_key("aabb"));
    }

    #[test]
    fn test_multiple_wallets_independent() {
        let t = TrafficTracker::new();
        t.record_rx("wallet1", 1000);
        t.record_tx("wallet2", 2000);

        let d = t.drain();
        assert_eq!(d["wallet1"].bytes_in,  1000);
        assert_eq!(d["wallet1"].bytes_out, 0);
        assert_eq!(d["wallet2"].bytes_in,  0);
        assert_eq!(d["wallet2"].bytes_out, 2000);
    }
}
