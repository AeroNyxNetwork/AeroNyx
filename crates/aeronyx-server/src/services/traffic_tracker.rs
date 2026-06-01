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
//   - TrafficDelta type defined in management/client.rs
//
// Main Logical Flow:
//   1. PacketHandler calls record_rx / record_tx on VPN path (hot path)
//   2. HeartbeatReporter calls drain() before building heartbeat payload
//   3. drain() atomically swap(0) on every WalletCounters entry
//   4. Only non-zero entries are included in the returned HashMap
//   5. remove_wallet() called from cleanup task after session expiry
//
// ⚠️ Important Notes for Next Developer:
//   - drain() uses Ordering::Relaxed — intentional. We do not need
//     cross-thread ordering for byte counters; a few bytes attributed
//     to the wrong 30s window is acceptable for quota accounting.
//   - remove_wallet() is conservative: only removes when BOTH counters
//     are zero. If a packet races with remove_wallet(), the entry stays
//     and will be cleaned up on the next heartbeat cycle via drain().
//   - wallet_hex keys must be lowercase hex (hex::encode output).
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
pub struct TrafficTracker {
    wallets: DashMap<String, Arc<WalletCounters>>,
}

impl TrafficTracker {
    pub fn new() -> Self {
        Self { wallets: DashMap::new() }
    }

    /// Records inbound bytes for a wallet (client → server).
    /// Called from PacketHandler::handle_udp_packet on IPv4/IPv6 VPN path.
    #[inline]
    pub fn record_rx(&self, wallet_hex: &str, bytes: u64) {
        self.get_or_create(wallet_hex)
            .bytes_in
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records outbound bytes for a wallet (server → client).
    /// Called from PacketHandler::handle_tun_packet.
    #[inline]
    pub fn record_tx(&self, wallet_hex: &str, bytes: u64) {
        self.get_or_create(wallet_hex)
            .bytes_out
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Atomically takes all accumulated deltas and resets counters to zero.
    /// Called by HeartbeatReporter before building the heartbeat payload.
    /// Returns only wallets that had non-zero traffic in the period.
    pub fn drain(&self) -> HashMap<String, TrafficDelta> {
        let mut result = HashMap::new();
        for entry in self.wallets.iter() {
            let counters  = entry.value();
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

    /// Removes the tracker entry for a wallet if its counters are zero.
    /// Called from server.rs spawn_cleanup_task after a session expires.
    pub fn remove_wallet(&self, wallet_hex: &str) {
        let should_remove = self.wallets.get(wallet_hex).map(|entry| {
            let c = entry.value();
            c.bytes_in.load(Ordering::Relaxed) == 0
                && c.bytes_out.load(Ordering::Relaxed) == 0
        }).unwrap_or(false);

        if should_remove {
            self.wallets.remove(wallet_hex);
        }
    }

    #[inline]
    fn get_or_create(&self, wallet_hex: &str) -> Arc<WalletCounters> {
        if let Some(entry) = self.wallets.get(wallet_hex) {
            return Arc::clone(entry.value());
        }
        Arc::clone(
            self.wallets
                .entry(wallet_hex.to_string())
                .or_insert_with(WalletCounters::new)
                .value(),
        )
    }
}

impl Default for TrafficTracker {
    fn default() -> Self { Self::new() }
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
        let d2 = t.drain();
        assert!(d2.is_empty());
    }

    #[test]
    fn test_drain_skips_zero_wallets() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);
        let _ = t.drain();
        let d2 = t.drain();
        assert!(!d2.contains_key("aabb"));
    }

    #[test]
    fn test_remove_wallet_zero_counters() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);
        let _ = t.drain();
        t.remove_wallet("aabb");
        assert!(!t.wallets.contains_key("aabb"));
    }

    #[test]
    fn test_remove_wallet_nonzero_stays() {
        let t = TrafficTracker::new();
        t.record_rx("aabb", 100);
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
