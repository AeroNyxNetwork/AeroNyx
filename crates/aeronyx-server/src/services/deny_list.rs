// ============================================
// File: crates/aeronyx-server/src/services/deny_list.rs
// ============================================
// Version: 1.1.0-Membership
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
//   - add(): record an independent wallet denial reason with its own TTL
//   - is_denied(): check before allowing handshake
//   - remove_reason(): clear only the control-plane reason being restored
//   - remove(): explicitly clear all reasons for backwards compatibility
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
//   - [DENY-REASON-ISOLATION 2026-07-29 by Codex] Membership, quota, and
//     operator-ban reasons are independent. Never clear all reasons when only
//     one control plane restores access; use remove_reason().
//   - wallet_hex keys must be lowercase hex (consistent with TrafficTracker).
//   - NoPremiumAccess TTL is u64::MAX unix seconds (~year 292 billion).
//     Treat it as "permanent until explicitly removed".
//   - cleanup() should be called every 60s from the cleanup task in server.rs.
//     It only removes QuotaExceeded entries whose month has rolled over.
//   - is_denied() returns false for unknown wallets (fail-open for handshake).
//
// Last Modified: v1.1.0-Membership — isolate concurrent deny reasons
// ============================================

use dashmap::DashMap;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

// ============================================
// DenyReason
// ============================================

/// The reason a wallet is on the deny list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
struct DenyReasonEntry {
    /// Unix timestamp (seconds) after which this entry expires.
    /// u64::MAX = permanent.
    expires_at_unix: u64,
}

impl DenyReasonEntry {
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

#[derive(Debug, Clone, Default)]
struct DenyEntry {
    /// Independent reasons keyed by the authority that imposed them.
    reasons: HashMap<DenyReason, DenyReasonEntry>,
}

impl DenyEntry {
    /// Returns the highest-priority active reason for legacy diagnostics.
    ///
    /// Operator bans must always win so membership state cannot hide an
    /// explicit operator action. Callers that restore one authority should
    /// use `remove_reason()` instead of relying on this aggregate view.
    fn primary_active_reason(&self) -> Option<DenyReason> {
        [
            DenyReason::OperatorBan,
            DenyReason::QuotaExceeded,
            DenyReason::NoPremiumAccess,
        ]
        .into_iter()
        .find(|reason| self.has_active_reason(*reason))
    }

    fn has_active_reason(&self, reason: DenyReason) -> bool {
        self.reasons
            .get(&reason)
            .map(DenyReasonEntry::is_active)
            .unwrap_or(false)
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

        // [DENY-REASON-ISOLATION 2026-07-29 by Codex] A wallet can be
        // independently blocked by operator policy and membership/quota
        // policy. Replacing the whole wallet entry here allowed a quota update
        // to erase an operator ban.
        self.entries
            .entry(wallet_hex.to_string())
            .or_default()
            .reasons
            .insert(reason, DenyReasonEntry { expires_at_unix });
    }

    /// Returns true if the wallet is currently on the deny list and
    /// the entry has not expired.
    ///
    /// Returns false for unknown wallets (fail-open).
    #[must_use]
    pub fn is_denied(&self, wallet_hex: &str) -> bool {
        self.entries
            .get(wallet_hex)
            .and_then(|entry| entry.primary_active_reason())
            .is_some()
    }

    /// Returns true when a specific deny reason is present and active.
    #[must_use]
    pub fn has_reason(&self, wallet_hex: &str, reason: DenyReason) -> bool {
        self.entries
            .get(wallet_hex)
            .map(|entry| entry.has_active_reason(reason))
            .unwrap_or(false)
    }

    /// Returns the deny reason for a wallet, if present and active.
    #[must_use]
    pub fn deny_reason(&self, wallet_hex: &str) -> Option<DenyReason> {
        self.entries
            .get(wallet_hex)
            .and_then(|entry| entry.primary_active_reason())
    }

    /// Returns all active wallets currently denied for a specific reason.
    #[must_use]
    pub fn wallets_for_reason(&self, reason: DenyReason) -> Vec<String> {
        self.entries
            .iter()
            .filter(|entry| entry.has_active_reason(reason))
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Removes one authority-specific deny reason without affecting others.
    ///
    /// Returns true when the requested active or expired reason existed.
    pub fn remove_reason(&self, wallet_hex: &str, reason: DenyReason) -> bool {
        let (removed, became_empty) = match self.entries.get_mut(wallet_hex) {
            Some(mut entry) => {
                let removed = entry.reasons.remove(&reason).is_some();
                (removed, entry.reasons.is_empty())
            }
            None => (false, false),
        };

        if became_empty {
            // The entry may have gained a reason after the mutable guard was
            // dropped. `remove_if` keeps that concurrent update intact.
            self.entries
                .remove_if(wallet_hex, |_wallet, entry| entry.reasons.is_empty());
        }

        if removed {
            info!(
                wallet = %&wallet_hex[..8.min(wallet_hex.len())],
                reason = %reason,
                "[DENY_LIST] Wallet deny reason removed"
            );
        }

        removed
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
            let before = entry.reasons.len();
            entry.reasons.retain(|reason, state| {
                if state.is_active() {
                    return true;
                }
                debug!(
                    wallet = %&wallet[..8.min(wallet.len())],
                    reason = %reason,
                    "[DENY_LIST] Expired deny reason evicted"
                );
                false
            });
            evicted += before - entry.reasons.len();
            !entry.reasons.is_empty()
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
        let reason = entry.reasons.get(&DenyReason::QuotaExceeded).unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        assert!(reason.expires_at_unix > now);
        assert!(reason.expires_at_unix != u64::MAX);
    }

    #[test]
    fn test_no_premium_access_is_permanent() {
        let dl = DenyList::new();
        dl.add("wallet2", DenyReason::NoPremiumAccess);

        let entry = dl.entries.get("wallet2").unwrap();
        assert_eq!(
            entry
                .reasons
                .get(&DenyReason::NoPremiumAccess)
                .unwrap()
                .expires_at_unix,
            u64::MAX
        );
    }

    #[test]
    fn test_cleanup_removes_expired() {
        let dl = DenyList::new();

        // Insert an already-expired entry by manipulating expires_at_unix.
        dl.entries.insert(
            "expired_wallet".to_string(),
            DenyEntry {
                reasons: HashMap::from([(
                    DenyReason::QuotaExceeded,
                    DenyReasonEntry {
                        expires_at_unix: 1, // Unix timestamp 1 = long expired
                    },
                )]),
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
    fn test_operator_ban_and_membership_reason_coexist() {
        let dl = DenyList::new();
        dl.add("wallet", DenyReason::OperatorBan);
        dl.add("wallet", DenyReason::QuotaExceeded);

        assert!(dl.has_reason("wallet", DenyReason::OperatorBan));
        assert!(dl.has_reason("wallet", DenyReason::QuotaExceeded));
        assert_eq!(
            dl.deny_reason("wallet"),
            Some(DenyReason::OperatorBan),
            "operator action must remain the primary diagnostic reason"
        );
        assert_eq!(dl.len(), 1, "len counts denied wallets, not reasons");
    }

    #[test]
    fn test_remove_reason_preserves_other_authorities() {
        let dl = DenyList::new();
        dl.add("wallet", DenyReason::OperatorBan);
        dl.add("wallet", DenyReason::QuotaExceeded);

        assert!(dl.remove_reason("wallet", DenyReason::QuotaExceeded));
        assert!(dl.is_denied("wallet"));
        assert!(dl.has_reason("wallet", DenyReason::OperatorBan));
        assert!(!dl.has_reason("wallet", DenyReason::QuotaExceeded));

        assert!(dl.remove_reason("wallet", DenyReason::OperatorBan));
        assert!(!dl.is_denied("wallet"));
        assert!(dl.is_empty());
    }

    #[test]
    fn test_cleanup_expired_reason_preserves_permanent_reason() {
        let dl = DenyList::new();
        dl.add("wallet", DenyReason::OperatorBan);
        dl.entries.get_mut("wallet").unwrap().reasons.insert(
            DenyReason::QuotaExceeded,
            DenyReasonEntry { expires_at_unix: 1 },
        );

        assert_eq!(dl.cleanup(), 1);
        assert!(dl.has_reason("wallet", DenyReason::OperatorBan));
        assert!(!dl.has_reason("wallet", DenyReason::QuotaExceeded));
        assert_eq!(dl.len(), 1);
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
