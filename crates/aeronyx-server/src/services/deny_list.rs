// ============================================
// File: crates/aeronyx-server/src/services/deny_list.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   New file. Prevents the 30-second reconnect loop that occurs when a
//   wallet is disconnected by the heartbeat membership check but
//   immediately reconnects before the next heartbeat fires.
//
//   Without this, the flow is:
//     heartbeat → disconnect wallet_B → client retries → session created →
//     next heartbeat → disconnect again → infinite loop every 30s
//
//   With this, the flow is:
//     heartbeat → disconnect wallet_B + add to deny list →
//     client retries → handshake rejects immediately →
//     client receives RESET, backs off
//
// Main Functionality:
//   - DenyList: thread-safe in-memory wallet block list
//   - add(): record a wallet denial with reason and TTL
//   - is_denied(): check before allowing handshake
//   - remove(): explicitly unblock (e.g. on subscription upgrade)
//   - cleanup(): evict expired entries (called periodically)
//
// Deny Reasons and TTLs:
//   - NoPremiumAccess: wallet's tier cannot access premium nodes.
//     TTL = permanent (u64::MAX) until explicitly removed.
//     Cleared when CMS confirms tier upgrade via heartbeat response.
//   - QuotaExceeded: Free tier monthly quota exhausted.
//     TTL = until end of current calendar month (UTC).
//     Automatically expires; no manual removal needed.
//
// Dependencies:
//   - services/handshake.rs: checks is_denied() before creating session
//   - management/reporter.rs: calls add() after disconnect decision
//   - server.rs: passes Arc<DenyList> to HandshakeService and reporter
//
// Main Logical Flow:
//   1. reporter calls add(wallet, reason) after disconnect
//   2. HandshakeService::process() calls is_denied() at entry
//   3. If denied: return Err(ServerError::WalletDenied) → send RESET
//   4. Periodic cleanup() removes expired QuotaExceeded entries
//   5. On heartbeat where user_permissions shows access restored:
//      reporter calls remove(wallet)
//
// ⚠️ Important Notes for Next Developer:
//   - DenyList is in-memory only. Server restart clears all entries.
//     This is intentional: CMS is the source of truth. After restart,
//     the first heartbeat will re-populate the deny list if needed.
//   - wallet_hex keys must be lowercase hex (consistent with TrafficTracker).
//   - NoPremiumAccess TTL is u64::MAX unix seconds (~year 292 billion).
//     Treat it as "permanent until explicitly removed".
//   - cleanup() should be called every 60s from the cleanup task in server.rs.
//     It only removes QuotaExceeded entries whose month has rolled over.
//   - is_denied() returns false for unknown wallets (fail-open for handshake).
//
// Last Modified: v1.0.0-Membership — initial implementation
// ============================================

use dashmap::DashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

// ============================================
// DenyReason
// ============================================

/// The reason a wallet is on the deny list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenyReason {
    /// Wallet's subscription tier cannot access premium nodes.
    /// Permanent until explicitly removed (e.g. on tier upgrade).
    NoPremiumAccess,
    /// Free tier monthly traffic quota exhausted.
    /// Expires automatically at the start of the next calendar month (UTC).
    QuotaExceeded,
    /// Operator-blocked wallet from nodeboard.
    /// Permanent until an explicit operator unban command removes it.
    OperatorBan,
}

impl std::fmt::Display for DenyReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPremiumAccess => write!(f, "no_premium_access"),
            Self::QuotaExceeded => write!(f, "quota_exceeded"),
            Self::OperatorBan => write!(f, "operator_ban"),
        }
    }
}

// ============================================
// DenyEntry (internal)
// ============================================

#[derive(Debug, Clone)]
struct DenyEntry {
    reason: DenyReason,
    denied_at: Instant,
    /// Unix timestamp (seconds) after which this entry expires.
    /// u64::MAX = permanent.
    expires_at_unix: u64,
}

impl DenyEntry {
    /// Returns true if this entry is still active (not expired).
    fn is_active(&self) -> bool {
        if self.expires_at_unix == u64::MAX {
            return true; // permanent
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now < self.expires_at_unix
    }
}

// ============================================
// DenyList
// ============================================

/// Thread-safe in-memory wallet deny list.
///
/// Prevents disconnected wallets from immediately reconnecting before
/// the next heartbeat cycle. Checked at handshake time.
pub struct DenyList {
    entries: DashMap<String, DenyEntry>,
}

impl DenyList {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    /// Adds a wallet to the deny list.
    ///
    /// - `NoPremiumAccess`: permanent entry (TTL = u64::MAX).
    /// - `QuotaExceeded`:   expires at start of next UTC calendar month.
    /// - `OperatorBan`:     permanent entry until operator unban.
    pub fn add(&self, wallet_hex: &str, reason: DenyReason) {
        let expires_at_unix = match reason {
            DenyReason::NoPremiumAccess => u64::MAX,
            DenyReason::QuotaExceeded => next_month_unix(),
            DenyReason::OperatorBan => u64::MAX,
        };

        info!(
            wallet   = %&wallet_hex[..8.min(wallet_hex.len())],
            reason   = %reason,
            expires  = if expires_at_unix == u64::MAX { "permanent".to_string() }
                       else { format!("unix:{}", expires_at_unix) },
            "[DENY_LIST] Wallet added"
        );

        self.entries.insert(
            wallet_hex.to_string(),
            DenyEntry {
                reason,
                denied_at: Instant::now(),
                expires_at_unix,
            },
        );
    }

    /// Returns true if the wallet is currently on the deny list and
    /// the entry has not expired.
    ///
    /// Returns false for unknown wallets (fail-open).
    #[must_use]
    pub fn is_denied(&self, wallet_hex: &str) -> bool {
        self.entries
            .get(wallet_hex)
            .map(|e| e.is_active())
            .unwrap_or(false)
    }

    /// Returns the deny reason for a wallet, if present and active.
    #[must_use]
    pub fn deny_reason(&self, wallet_hex: &str) -> Option<DenyReason> {
        self.entries
            .get(wallet_hex)
            .filter(|e| e.is_active())
            .map(|e| e.reason.clone())
    }

    /// Returns all active wallets currently denied for a specific reason.
    #[must_use]
    pub fn wallets_for_reason(&self, reason: DenyReason) -> Vec<String> {
        self.entries
            .iter()
            .filter(|entry| entry.is_active() && entry.reason == reason)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Explicitly removes a wallet from the deny list.
    ///
    /// Called when CMS heartbeat response confirms the wallet's access
    /// has been restored (tier upgrade or quota reset).
    pub fn remove(&self, wallet_hex: &str) {
        if self.entries.remove(wallet_hex).is_some() {
            info!(
                wallet = %&wallet_hex[..8.min(wallet_hex.len())],
                "[DENY_LIST] Wallet removed (access restored)"
            );
        }
    }

    /// Removes all expired entries from the list.
    ///
    /// Should be called periodically (every 60s) from the cleanup task.
    /// Only QuotaExceeded entries with elapsed TTL are removed;
    /// NoPremiumAccess entries are permanent and never auto-removed.
    ///
    /// Returns the number of entries evicted.
    pub fn cleanup(&self) -> usize {
        let mut evicted = 0usize;
        self.entries.retain(|wallet, entry| {
            if entry.is_active() {
                true
            } else {
                debug!(
                    wallet = %&wallet[..8.min(wallet.len())],
                    reason = %entry.reason,
                    "[DENY_LIST] Expired entry evicted"
                );
                evicted += 1;
                false
            }
        });
        if evicted > 0 {
            info!(
                evicted,
                "[DENY_LIST] Cleanup removed {} expired entries", evicted
            );
        }
        evicted
    }

    /// Returns the total number of entries (active + expired not yet cleaned).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for DenyList {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DenyList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenyList")
            .field("entries", &self.entries.len())
            .finish()
    }
}

// ============================================
// Helper: next UTC calendar month as Unix timestamp
// ============================================

/// Returns the Unix timestamp (seconds) of 00:00:00 UTC on the first day
/// of the next calendar month.
///
/// Example: called on 2026-05-15 → returns Unix timestamp of 2026-06-01 00:00:00 UTC.
///
/// Uses only std — no chrono dependency.
fn next_month_unix() -> u64 {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Approximate: days elapsed since epoch → current year/month
    // We use a simple calculation: add enough seconds to reach the 1st of
    // next month. Worst case off by one day (DST-free UTC, so exact).
    let days_since_epoch = now_secs / 86400;
    // Rata Die to Gregorian (algorithm by Howard Hinnant, public domain)
    let z = days_since_epoch as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    // Next month
    let (next_y, next_m) = if m == 12 { (y + 1, 1i64) } else { (y, m + 1) };

    // Convert next_y/next_m/1 back to Unix timestamp
    // Days from epoch to year (Gregorian proleptic)
    let y0 = if next_m <= 2 { next_y - 1 } else { next_y };
    let era = if y0 >= 0 { y0 } else { y0 - 399 } / 400;
    let yoe = y0 - era * 400;
    let doy = (153 * (if next_m > 2 { next_m - 3 } else { next_m + 9 }) + 2) / 5;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;

    (days as u64) * 86400
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_is_denied() {
        let dl = DenyList::new();
        assert!(!dl.is_denied("aabbcc"));

        dl.add("aabbcc", DenyReason::NoPremiumAccess);
        assert!(dl.is_denied("aabbcc"));
    }

    #[test]
    fn test_remove_clears_entry() {
        let dl = DenyList::new();
        dl.add("aabbcc", DenyReason::NoPremiumAccess);
        assert!(dl.is_denied("aabbcc"));

        dl.remove("aabbcc");
        assert!(!dl.is_denied("aabbcc"));
    }

    #[test]
    fn test_unknown_wallet_not_denied() {
        let dl = DenyList::new();
        assert!(!dl.is_denied("unknown_wallet"));
    }

    #[test]
    fn test_quota_exceeded_has_expiry() {
        let dl = DenyList::new();
        dl.add("wallet1", DenyReason::QuotaExceeded);

        // Entry must be active right after insertion.
        assert!(dl.is_denied("wallet1"));

        // expires_at_unix must be > now (not permanent).
        let entry = dl.entries.get("wallet1").unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        assert!(entry.expires_at_unix > now);
        assert!(entry.expires_at_unix != u64::MAX);
    }

    #[test]
    fn test_no_premium_access_is_permanent() {
        let dl = DenyList::new();
        dl.add("wallet2", DenyReason::NoPremiumAccess);

        let entry = dl.entries.get("wallet2").unwrap();
        assert_eq!(entry.expires_at_unix, u64::MAX);
    }

    #[test]
    fn test_cleanup_removes_expired() {
        let dl = DenyList::new();

        // Insert an already-expired entry by manipulating expires_at_unix.
        dl.entries.insert(
            "expired_wallet".to_string(),
            DenyEntry {
                reason: DenyReason::QuotaExceeded,
                denied_at: Instant::now(),
                expires_at_unix: 1, // Unix timestamp 1 = long expired
            },
        );

        dl.add("active_wallet", DenyReason::NoPremiumAccess);

        assert_eq!(dl.len(), 2);
        let evicted = dl.cleanup();
        assert_eq!(evicted, 1);
        assert_eq!(dl.len(), 1);
        assert!(!dl.is_denied("expired_wallet"));
        assert!(dl.is_denied("active_wallet"));
    }

    #[test]
    fn test_deny_reason_returned() {
        let dl = DenyList::new();
        dl.add("w1", DenyReason::QuotaExceeded);
        dl.add("w2", DenyReason::NoPremiumAccess);

        assert_eq!(dl.deny_reason("w1"), Some(DenyReason::QuotaExceeded));
        assert_eq!(dl.deny_reason("w2"), Some(DenyReason::NoPremiumAccess));
        assert_eq!(dl.deny_reason("w3"), None);
    }

    #[test]
    fn test_next_month_unix_is_in_future() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let next = next_month_unix();
        assert!(next > now, "next_month_unix must be in the future");
        // Must be within ~32 days
        assert!(
            next - now <= 32 * 86400,
            "next_month_unix must be within 32 days"
        );
    }
}
