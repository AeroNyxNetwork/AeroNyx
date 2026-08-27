// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_custody_anchor_guard.rs
// ============================================
// Version: 1.0.0-CustodyAnchorGuardContract
//
// Creation Reason:
//   [CHAT-CUSTODY-ANCHOR-GUARD-DOMAIN 2026-08-28 by Codex] Extract the
//   custody-anchor RAII contract from the relay orchestration service.
//
// Main Functionality:
//   - Binds one signed custody anchor to an acquired maintenance lock.
//   - Exposes only the signed aggregate anchor to callers.
//   - Releases the host-local lock automatically when the guard is dropped.
//   - Allows service-owned compatibility APIs to consume the guarded anchor.
//
// Dependencies:
//   - `aeronyx-core` owns the signed custody audit anchor wire contract.
//   - `rusqlite::Connection` owns the cross-process maintenance lock lifetime.
//   - `chat_relay.rs` verifies and signs an anchor before constructing a guard.
//
// Main Logical Flow:
//   1. The service acquires the cross-process maintenance lock.
//   2. It verifies the complete private audit chain and signs one anchor.
//   3. The private constructor binds both resources into this guard.
//   4. Consumers inspect the anchor while RAII preserves the lock boundary.
//
// Important Note for Next Developer:
//   - Do not expose the lock connection or its operator-local filesystem path.
//   - Never construct a guard before chain verification and anchor signing.
//   - Preserve RAII release; explicit unlock paths can create TOCTOU windows.
//   - The public anchor contains aggregate commitments, never private audit MACs.
//
// Last Modified:
//   v1.0.0-CustodyAnchorGuardContract - Initial RAII contract extraction
// ============================================

use aeronyx_core::protocol::chat::CustodyAuditAnchorV1;
use rusqlite::Connection;

/// Cross-process guard binding one signed custody anchor to the current
/// immutable maintenance checkpoint for the complete lifetime of the value.
///
/// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] The private `SQLite`
/// connection owns an exclusive maintenance transaction and is released by
/// RAII. Callers may inspect only the signed aggregate anchor; no lock path,
/// private audit state, HMAC, or custody metadata crosses this boundary.
pub struct ChatRelayCustodyAuditAnchorGuard {
    _filesystem_lock: Connection,
    anchor: CustodyAuditAnchorV1,
}

impl ChatRelayCustodyAuditAnchorGuard {
    /// Binds a verified signed anchor to its still-held maintenance lock.
    pub(crate) fn new(filesystem_lock: Connection, anchor: CustodyAuditAnchorV1) -> Self {
        Self {
            _filesystem_lock: filesystem_lock,
            anchor,
        }
    }

    /// Returns the exact current producer-signed anchor protected by the guard.
    #[must_use]
    pub const fn anchor(&self) -> &CustodyAuditAnchorV1 {
        &self.anchor
    }

    /// Consumes the guard and returns its anchor, releasing the lock afterward.
    pub(crate) fn into_anchor(self) -> CustodyAuditAnchorV1 {
        self.anchor
    }
}
