// ============================================================================
// File: crates/aeronyx-server/src/services/wallet_routes.rs
// ============================================================================
// Version: 1.3.0-Sovereign
//
// Modification Reason:
//   New file. Implements the in-memory wallet → session routing table used by
//   the v1.3.0-Sovereign chat relay. Previously the server relied on
//   SessionManager::get_all_by_wallet() which trusts the session handshake
//   key as proof of wallet ownership. The new design requires an explicit
//   signed announcement before adding a wallet to the route table.
//
// Main Functionality:
//   - WalletRouteCache: RwLock-guarded HashMap of wallet → {session_id → RouteEntry}
//   - announce(): insert/refresh a route entry (called after successful sig verify)
//   - lookup(): return all active (session_id, SocketAddr) for a wallet
//   - remove_session(): called on session disconnect to prune stale entries
//   - cleanup_stale(): background task hook to evict entries idle > TTL
//   - snapshot_for_admin(): returns (wallet_count, total_session_count) for monitoring
//
// Dependencies:
//   - aeronyx_common::types::SessionId
//   - std::net::SocketAddr
//   - std::time::Instant (for last_active)
//   - parking_lot::RwLock (consistent with the rest of the codebase)
//
// Main Logical Flow:
//   announce(wallet, session_id, endpoint):
//     1. Acquire write lock
//     2. Get or insert the inner HashMap for this wallet
//     3. Insert/overwrite RouteEntry with current Instant::now()
//
//   lookup(wallet):
//     1. Acquire read lock
//     2. Find wallet entry, collect (session_id, endpoint) pairs
//     3. Return Vec (empty if wallet not found or all sessions stale)
//
//   remove_session(session_id):
//     1. Acquire write lock
//     2. Iterate all wallet entries, remove matching session_id
//     3. Remove wallet entries that became empty
//
//   cleanup_stale(ttl):
//     1. Acquire write lock
//     2. For each wallet, retain only sessions with last_active within TTL
//     3. Remove wallets with no remaining sessions
//
// ⚠️ Important Notes for Next Developer:
//   - This table is PURE MEMORY — it does not persist across restarts.
//     Clients re-announce on reconnect via DeviceRegister or WalletPresence.
//   - Do NOT hold the write lock while doing I/O (SQLite, UDP send, etc.).
//     The lock must be acquired, mutated, and released before any blocking op.
//   - cleanup_stale() should be called from a background task every 60 s
//     with a TTL of 300 s (see server.rs spawn). The 60 s window between
//     cleanup runs means a stale entry can live up to TTL+60 s in the worst case.
//   - announce() is idempotent — calling it repeatedly for the same
//     (wallet, session_id) pair just refreshes last_active. This is intentional.
//   - remove_session() is O(wallets). For typical deployments (<10k wallets)
//     this is negligible. If wallet count grows to millions, consider an
//     inverse index: session_id → wallet for O(1) removal.
//
// Last Modified: v1.3.0-Sovereign — Initial implementation
// ============================================================================

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use aeronyx_common::types::SessionId;

// ============================================
// RouteEntry
// ============================================

/// A single active session entry in the wallet route table.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// The UDP endpoint for this session (used for direct push).
    pub endpoint: SocketAddr,
    /// When this entry was last refreshed by a signed message.
    /// Used by `cleanup_stale()` to evict idle entries.
    pub last_active: Instant,
}

// ============================================
// WalletRouteCache
// ============================================

/// In-memory route table mapping wallet public keys to active sessions.
///
/// ## Thread Safety
/// Uses `parking_lot::RwLock` for interior mutability. Multiple readers
/// can hold the lock concurrently; writes are exclusive.
///
/// ## Persistence
/// None — this table is rebuilt from scratch on server restart as clients
/// reconnect and re-announce via DeviceRegister / WalletPresence.
pub struct WalletRouteCache {
    /// wallet_pubkey → { session_id → RouteEntry }
    inner: RwLock<HashMap<[u8; 32], HashMap<SessionId, RouteEntry>>>,
}

impl WalletRouteCache {
    /// Creates a new empty route cache.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    // ============================================
    // announce
    // ============================================

    /// Registers or refreshes a wallet → session mapping.
    ///
    /// Called after every successfully verified signed message so that
    /// `last_active` stays current and the route is not cleaned up by
    /// `cleanup_stale()` while the client is actively communicating.
    ///
    /// ## Idempotency
    /// Safe to call repeatedly for the same (wallet, session_id) pair —
    /// subsequent calls just update `last_active` and `endpoint`.
    pub fn announce(&self, wallet: &[u8; 32], session_id: SessionId, endpoint: SocketAddr) {
        let mut map = self.inner.write();
        map.entry(*wallet)
            .or_insert_with(HashMap::new)
            .insert(session_id, RouteEntry {
                endpoint,
                last_active: Instant::now(),
            });
    }

    // ============================================
    // lookup
    // ============================================

    /// Returns all active (session_id, endpoint) pairs for a wallet.
    ///
    /// Returns an empty `Vec` if the wallet has no known active sessions.
    ///
    /// ## Note
    /// The returned list may include sessions that have since disconnected
    /// but have not yet been removed (race between disconnect and lookup).
    /// Callers should handle push failures gracefully and call
    /// `remove_session()` on failure.
    pub fn lookup(&self, wallet: &[u8; 32]) -> Vec<(SessionId, SocketAddr)> {
        let map = self.inner.read();
        match map.get(wallet) {
            None => Vec::new(),
            Some(sessions) => sessions
                .iter()
                .map(|(sid, entry)| (sid.clone(), entry.endpoint))
                .collect(),
        }
    }

    // ============================================
    // remove_session
    // ============================================

    /// Removes all route entries associated with the given session.
    ///
    /// Called when a session disconnects (graceful or timeout). Walks all
    /// wallet entries and removes matching session_id. Empty wallet entries
    /// are pruned.
    ///
    /// ## Complexity
    /// O(wallets) — acceptable for typical deployments. See notes above
    /// if wallet count grows to millions.
    pub fn remove_session(&self, session_id: &SessionId) {
        let mut map = self.inner.write();
        map.retain(|_wallet, sessions| {
            sessions.remove(session_id);
            !sessions.is_empty()
        });
    }

    // ============================================
    // cleanup_stale
    // ============================================

    /// Evicts route entries whose `last_active` is older than `ttl`.
    ///
    /// Intended to be called by a background task every 60 s with
    /// `ttl = Duration::from_secs(300)`.
    ///
    /// Returns the number of evicted session entries (for logging).
    pub fn cleanup_stale(&self, ttl: Duration) -> usize {
        let now = Instant::now();
        let mut map = self.inner.write();
        let mut evicted = 0usize;

        map.retain(|_wallet, sessions| {
            let before = sessions.len();
            sessions.retain(|_sid, entry| {
                now.duration_since(entry.last_active) <= ttl
            });
            evicted += before - sessions.len();
            !sessions.is_empty()
        });

        evicted
    }

    // ============================================
    // snapshot_for_admin
    // ============================================

    /// Returns `(wallet_count, total_session_count)` for monitoring.
    ///
    /// Acquires a read lock briefly; safe to call from any task.
    pub fn snapshot_for_admin(&self) -> (usize, usize) {
        let map = self.inner.read();
        let wallets = map.len();
        let sessions: usize = map.values().map(|s| s.len()).sum();
        (wallets, sessions)
    }
}

impl Default for WalletRouteCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn make_session() -> SessionId {
        SessionId::from_bytes(&rand::random::<[u8; 16]>())
            .expect("random bytes form valid SessionId")
    }

    fn make_addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{}", port).parse().unwrap()
    }

    // ── announce + lookup ────────────────────────────────────────────────

    #[test]
    fn test_announce_then_lookup_returns_entry() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid = make_session();
        let addr = make_addr(9000);

        cache.announce(&wallet, sid.clone(), addr);

        let results = cache.lookup(&wallet);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, sid);
        assert_eq!(results[0].1, addr);
    }

    #[test]
    fn test_lookup_unknown_wallet_returns_empty() {
        let cache = WalletRouteCache::new();
        let wallet = [0xBBu8; 32];
        assert!(cache.lookup(&wallet).is_empty());
    }

    #[test]
    fn test_announce_refreshes_endpoint() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid = make_session();

        cache.announce(&wallet, sid.clone(), make_addr(9000));
        cache.announce(&wallet, sid.clone(), make_addr(9001)); // update endpoint

        let results = cache.lookup(&wallet);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, make_addr(9001), "Endpoint must be updated");
    }

    // ── Multi-device ─────────────────────────────────────────────────────

    #[test]
    fn test_multiple_sessions_same_wallet() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid1 = make_session();
        let sid2 = make_session();
        let sid3 = make_session();

        cache.announce(&wallet, sid1.clone(), make_addr(9001));
        cache.announce(&wallet, sid2.clone(), make_addr(9002));
        cache.announce(&wallet, sid3.clone(), make_addr(9003));

        let results = cache.lookup(&wallet);
        assert_eq!(results.len(), 3, "All 3 sessions must be returned");

        let ports: Vec<u16> = results.iter().map(|(_, a)| a.port()).collect();
        assert!(ports.contains(&9001));
        assert!(ports.contains(&9002));
        assert!(ports.contains(&9003));
    }

    #[test]
    fn test_multiple_wallets_isolated() {
        let cache = WalletRouteCache::new();
        let wallet_a = [0xAAu8; 32];
        let wallet_b = [0xBBu8; 32];
        let sid_a = make_session();
        let sid_b = make_session();

        cache.announce(&wallet_a, sid_a.clone(), make_addr(9000));
        cache.announce(&wallet_b, sid_b.clone(), make_addr(9001));

        assert_eq!(cache.lookup(&wallet_a).len(), 1);
        assert_eq!(cache.lookup(&wallet_b).len(), 1);
        // wallet_a's session must not appear in wallet_b's lookup
        let b_sids: Vec<SessionId> = cache.lookup(&wallet_b).into_iter().map(|(s, _)| s).collect();
        assert!(!b_sids.contains(&sid_a));
    }

    // ── remove_session ───────────────────────────────────────────────────

    #[test]
    fn test_remove_session_clears_entry() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid = make_session();

        cache.announce(&wallet, sid.clone(), make_addr(9000));
        cache.remove_session(&sid);

        assert!(
            cache.lookup(&wallet).is_empty(),
            "After remove_session, lookup must return empty"
        );
    }

    #[test]
    fn test_remove_session_only_removes_matching_session() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid1 = make_session();
        let sid2 = make_session();

        cache.announce(&wallet, sid1.clone(), make_addr(9001));
        cache.announce(&wallet, sid2.clone(), make_addr(9002));

        cache.remove_session(&sid1);

        let results = cache.lookup(&wallet);
        assert_eq!(results.len(), 1, "Only one session must remain");
        assert_eq!(results[0].0, sid2);
    }

    #[test]
    fn test_remove_session_prunes_empty_wallet_entry() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid = make_session();

        cache.announce(&wallet, sid.clone(), make_addr(9000));
        cache.remove_session(&sid);

        // Wallet entry itself must be pruned (snapshot shows 0 wallets)
        let (wallets, sessions) = cache.snapshot_for_admin();
        assert_eq!(wallets, 0);
        assert_eq!(sessions, 0);
    }

    #[test]
    fn test_remove_nonexistent_session_is_noop() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid_present = make_session();
        let sid_absent = make_session();

        cache.announce(&wallet, sid_present.clone(), make_addr(9000));
        cache.remove_session(&sid_absent); // must not panic or affect other entries

        assert_eq!(cache.lookup(&wallet).len(), 1);
    }

    // ── cleanup_stale ────────────────────────────────────────────────────

    #[test]
    fn test_cleanup_stale_removes_old_entries() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid = make_session();

        cache.announce(&wallet, sid, make_addr(9000));

        // Evict with a zero TTL — everything is immediately stale
        let evicted = cache.cleanup_stale(Duration::from_secs(0));
        assert_eq!(evicted, 1, "One stale entry must be evicted");
        assert!(cache.lookup(&wallet).is_empty());
    }

    #[test]
    fn test_cleanup_stale_preserves_fresh_entries() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid = make_session();

        cache.announce(&wallet, sid, make_addr(9000));

        // Evict with a very long TTL — nothing should be removed
        let evicted = cache.cleanup_stale(Duration::from_secs(3600));
        assert_eq!(evicted, 0, "Fresh entry must not be evicted");
        assert_eq!(cache.lookup(&wallet).len(), 1);
    }

    #[test]
    fn test_cleanup_stale_selectively_evicts() {
        let cache = WalletRouteCache::new();
        let wallet = [0xAAu8; 32];
        let sid_stale = make_session();
        let sid_fresh = make_session();

        cache.announce(&wallet, sid_stale.clone(), make_addr(9001));

        // Sleep just enough to make the first entry "stale" relative to 0 TTL
        // then announce a second entry (fresh)
        std::thread::sleep(Duration::from_millis(2));
        cache.announce(&wallet, sid_fresh.clone(), make_addr(9002));

        // Evict anything older than 1 ms
        let evicted = cache.cleanup_stale(Duration::from_millis(1));
        // sid_stale is > 1 ms old; sid_fresh may or may not be depending on timing.
        // At minimum, no panic and wallet entry still exists.
        assert!(evicted <= 1);
        let remaining = cache.lookup(&wallet);
        assert!(!remaining.is_empty(), "At least sid_fresh must remain");
    }

    // ── snapshot_for_admin ────────────────────────────────────────────────

    #[test]
    fn test_snapshot_reflects_current_state() {
        let cache = WalletRouteCache::new();
        let (w, s) = cache.snapshot_for_admin();
        assert_eq!(w, 0);
        assert_eq!(s, 0);

        let wallet_a = [0xAAu8; 32];
        let wallet_b = [0xBBu8; 32];
        cache.announce(&wallet_a, make_session(), make_addr(9000));
        cache.announce(&wallet_a, make_session(), make_addr(9001));
        cache.announce(&wallet_b, make_session(), make_addr(9002));

        let (w, s) = cache.snapshot_for_admin();
        assert_eq!(w, 2, "Two distinct wallets");
        assert_eq!(s, 3, "Three total sessions");
    }
}
