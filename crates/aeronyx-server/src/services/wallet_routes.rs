// ============================================================================
// File: crates/aeronyx-server/src/services/wallet_routes.rs
// ============================================================================
// Version: 1.4.0-BoundedRouteIndex
//
// Modification Reason:
//   New file. Implements the in-memory wallet → session routing table used by
//   the v1.3.0-Sovereign chat relay. Previously the server relied on
//   SessionManager::get_all_by_wallet() which trusts the session handshake
//   key as proof of wallet ownership. The new design requires an explicit
//   signed announcement before adding a wallet to the route table.
//
// Main Functionality:
//   - WalletRouteCache: bounded bidirectional wallet ↔ session route index
//   - announce(): insert/refresh a route entry (called after successful sig verify)
//   - lookup(): return all active (session_id, SocketAddr) for a wallet
//   - remove_route(): rollback one wallet/session association atomically
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
//     2. Enforce the per-session wallet-identity ceiling
//     3. Update both forward and reverse indexes atomically
//
//   lookup(wallet):
//     1. Acquire read lock
//     2. Find wallet entry, collect (session_id, endpoint) pairs
//     3. Return Vec (empty if wallet not found or all sessions stale)
//
//   remove_session(session_id):
//     1. Acquire write lock
//     2. Read the session's wallet set from the reverse index
//     3. Remove only those forward entries and prune empty wallets
//
//   cleanup_stale(ttl):
//     1. Acquire write lock
//     2. For each wallet, retain only sessions with last_active within TTL
//     3. Remove matching reverse-index entries and empty wallets
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
//   - [WALLET-ROUTE-BOUND 2026-07-29 by Codex] A live session may announce
//     at most MAX_WALLETS_PER_SESSION distinct signed wallet identities.
//     Keep this bound aligned with SessionManager's device-registration bound.
//   - Both indexes live under one RwLock. Never split their mutation across
//     independent locks or a disconnect can leave a route visible.
//
// Last Modified: v1.4.0-BoundedRouteIndex — Bounded bidirectional index
// ============================================================================

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use aeronyx_common::types::SessionId;

/// Maximum distinct signed wallet identities one live transport session may
/// publish. This keeps memory proportional to the configured session capacity
/// while still allowing a small, intentional multi-account client.
const MAX_WALLETS_PER_SESSION: usize = 8;

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

/// Forward and reverse route indexes guarded by one atomic lock.
#[derive(Default)]
struct WalletRouteState {
    /// wallet_pubkey → { session_id → RouteEntry }
    routes_by_wallet: HashMap<[u8; 32], HashMap<SessionId, RouteEntry>>,
    /// session_id → wallet_pubkeys announced by that live session
    wallets_by_session: HashMap<SessionId, HashSet<[u8; 32]>>,
}

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
    /// Forward and reverse indexes share one lock so mutations are atomic.
    inner: RwLock<WalletRouteState>,
}

impl WalletRouteCache {
    /// Creates a new empty route cache.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(WalletRouteState::default()),
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
    ///
    /// Returns `false` without mutating either index if this session already
    /// reached `MAX_WALLETS_PER_SESSION`.
    pub fn announce(&self, wallet: &[u8; 32], session_id: SessionId, endpoint: SocketAddr) -> bool {
        let mut state = self.inner.write();
        let is_existing = state
            .wallets_by_session
            .get(&session_id)
            .is_some_and(|wallets| wallets.contains(wallet));
        if !is_existing
            && state
                .wallets_by_session
                .get(&session_id)
                .is_some_and(|wallets| wallets.len() >= MAX_WALLETS_PER_SESSION)
        {
            return false;
        }

        state
            .wallets_by_session
            .entry(session_id.clone())
            .or_default()
            .insert(*wallet);
        state.routes_by_wallet.entry(*wallet).or_default().insert(
            session_id,
            RouteEntry {
                endpoint,
                last_active: Instant::now(),
            },
        );
        true
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
        let state = self.inner.read();
        match state.routes_by_wallet.get(wallet) {
            None => Vec::new(),
            Some(sessions) => sessions
                .iter()
                .map(|(sid, entry)| (sid.clone(), entry.endpoint))
                .collect(),
        }
    }

    // ============================================
    // remove_route
    // ============================================

    /// Removes one wallet → session association from both indexes.
    ///
    /// Used to roll back `DeviceRegister` when the session-side device index
    /// rejects the registration after the signed route was admitted.
    pub fn remove_route(&self, wallet: &[u8; 32], session_id: &SessionId) -> bool {
        let mut state = self.inner.write();
        let removed = state
            .routes_by_wallet
            .get_mut(wallet)
            .is_some_and(|sessions| sessions.remove(session_id).is_some());
        if !removed {
            return false;
        }

        let prune_wallet = state
            .routes_by_wallet
            .get(wallet)
            .is_some_and(HashMap::is_empty);
        if prune_wallet {
            state.routes_by_wallet.remove(wallet);
        }

        let prune_session = if let Some(wallets) = state.wallets_by_session.get_mut(session_id) {
            wallets.remove(wallet);
            wallets.is_empty()
        } else {
            false
        };
        if prune_session {
            state.wallets_by_session.remove(session_id);
        }

        true
    }

    // ============================================
    // remove_session
    // ============================================

    /// Removes all route entries associated with the given session.
    ///
    /// Called when a session disconnects (graceful or timeout). Uses the
    /// reverse index to visit only wallets announced by this session.
    ///
    /// ## Complexity
    /// O(wallets announced by this session), bounded by
    /// `MAX_WALLETS_PER_SESSION`.
    pub fn remove_session(&self, session_id: &SessionId) {
        let mut state = self.inner.write();
        let Some(wallets) = state.wallets_by_session.remove(session_id) else {
            return;
        };

        for wallet in wallets {
            let prune_wallet = if let Some(sessions) = state.routes_by_wallet.get_mut(&wallet) {
                sessions.remove(session_id);
                sessions.is_empty()
            } else {
                false
            };
            if prune_wallet {
                state.routes_by_wallet.remove(&wallet);
            }
        }
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
        let mut state = self.inner.write();
        let mut evicted = 0usize;
        let mut stale_pairs = Vec::new();

        state.routes_by_wallet.retain(|wallet, sessions| {
            let before = sessions.len();
            sessions.retain(|session_id, entry| {
                let fresh = now.duration_since(entry.last_active) <= ttl;
                if !fresh {
                    stale_pairs.push((session_id.clone(), *wallet));
                }
                fresh
            });
            evicted += before - sessions.len();
            !sessions.is_empty()
        });

        for (session_id, wallet) in stale_pairs {
            let prune_session = if let Some(wallets) = state.wallets_by_session.get_mut(&session_id)
            {
                wallets.remove(&wallet);
                wallets.is_empty()
            } else {
                false
            };
            if prune_session {
                state.wallets_by_session.remove(&session_id);
            }
        }

        evicted
    }

    // ============================================
    // snapshot_for_admin
    // ============================================

    /// Returns `(wallet_count, total_session_count)` for monitoring.
    ///
    /// Acquires a read lock briefly; safe to call from any task.
    pub fn snapshot_for_admin(&self) -> (usize, usize) {
        let state = self.inner.read();
        let wallets = state.routes_by_wallet.len();
        let sessions: usize = state.routes_by_wallet.values().map(|s| s.len()).sum();
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

    #[test]
    fn test_single_session_wallet_count_is_bounded() {
        let cache = WalletRouteCache::new();
        let sid = make_session();

        for index in 0..MAX_WALLETS_PER_SESSION {
            let wallet = [index as u8; 32];
            assert!(cache.announce(&wallet, sid.clone(), make_addr(9000 + index as u16)));
        }

        let rejected_wallet = [0xFE; 32];
        assert!(!cache.announce(&rejected_wallet, sid.clone(), make_addr(9999)));
        assert!(cache.lookup(&rejected_wallet).is_empty());
        assert_eq!(
            cache.snapshot_for_admin(),
            (MAX_WALLETS_PER_SESSION, MAX_WALLETS_PER_SESSION)
        );

        let existing_wallet = [0u8; 32];
        assert!(cache.announce(&existing_wallet, sid, make_addr(9100)));
        assert_eq!(cache.lookup(&existing_wallet)[0].1, make_addr(9100));
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
        let b_sids: Vec<SessionId> = cache
            .lookup(&wallet_b)
            .into_iter()
            .map(|(s, _)| s)
            .collect();
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
    fn test_remove_route_updates_both_indexes() {
        let cache = WalletRouteCache::new();
        let sid = make_session();
        let removed_wallet = [0xAAu8; 32];
        let retained_wallet = [0xBBu8; 32];
        assert!(cache.announce(&removed_wallet, sid.clone(), make_addr(9001)));
        assert!(cache.announce(&retained_wallet, sid.clone(), make_addr(9002)));

        assert!(cache.remove_route(&removed_wallet, &sid));
        assert!(!cache.remove_route(&removed_wallet, &sid));
        assert!(cache.lookup(&removed_wallet).is_empty());
        assert_eq!(cache.lookup(&retained_wallet).len(), 1);

        for index in 0..(MAX_WALLETS_PER_SESSION - 2) {
            assert!(cache.announce(
                &[0x10 + index as u8; 32],
                sid.clone(),
                make_addr(9100 + index as u16),
            ));
        }
        assert!(cache.announce(&[0xFE; 32], sid, make_addr(9999)));
    }

    #[test]
    fn test_remove_session_prunes_every_announced_wallet() {
        let cache = WalletRouteCache::new();
        let sid = make_session();
        let wallets = [[0xAAu8; 32], [0xBBu8; 32], [0xCCu8; 32]];

        for (index, wallet) in wallets.iter().enumerate() {
            assert!(cache.announce(wallet, sid.clone(), make_addr(9000 + index as u16)));
        }

        cache.remove_session(&sid);

        assert_eq!(cache.snapshot_for_admin(), (0, 0));
        for wallet in wallets {
            assert!(cache.lookup(&wallet).is_empty());
        }
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
    fn test_cleanup_stale_releases_session_wallet_capacity() {
        let cache = WalletRouteCache::new();
        let sid = make_session();

        for index in 0..MAX_WALLETS_PER_SESSION {
            assert!(cache.announce(
                &[index as u8; 32],
                sid.clone(),
                make_addr(9000 + index as u16),
            ));
        }
        assert_eq!(
            cache.cleanup_stale(Duration::from_secs(0)),
            MAX_WALLETS_PER_SESSION
        );

        let replacement = [0xFE; 32];
        assert!(cache.announce(&replacement, sid, make_addr(9999)));
        assert_eq!(cache.lookup(&replacement).len(), 1);
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
