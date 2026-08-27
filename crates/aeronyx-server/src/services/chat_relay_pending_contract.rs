// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_contract.rs
// ============================================
// Version: 1.0.0-PendingDeliveryContract
//
// Creation Reason:
//   [CHAT-PENDING-CONTRACT-DOMAIN 2026-08-28 by Codex] Move pending delivery
//   models out of the relay service and remove domain-to-orchestrator coupling.
//
// Main Functionality:
//   - Defines one validated pending encrypted message returned to a client.
//   - Defines the stable v2 snapshot page and opaque continuation state.
//   - Preserves the existing public type shape through service re-exports.
//
// Dependencies:
//   - `aeronyx-core` owns the signed end-to-end encrypted chat envelope.
//   - Pending pull and delivery domains consume these contracts directly.
//   - `chat_relay.rs` re-exports them for backward-compatible public paths.
//
// Main Logical Flow:
//   1. Durable readers validate and decode an encrypted envelope.
//   2. Delivery composition collects validated messages into one page.
//   3. The service returns the same public contract without inspecting content.
//
// Important Note for Next Developer:
//   - These contracts contain encrypted payloads; never add plaintext fields.
//   - Keep message ID and cursor wire semantics backward compatible.
//   - Pagination policy belongs in `chat_relay_pending_delivery.rs`.
//   - Durable row validation belongs in `chat_relay_pending_pull.rs`.
//
// Last Modified:
//   v1.0.0-PendingDeliveryContract - Initial contract extraction
// ============================================

use aeronyx_core::protocol::chat::ChatEnvelope;

/// A pending offline message retrieved from the store.
#[derive(Debug)]
pub struct PendingMessage {
    /// Opaque client-generated message identifier used for ACK pagination.
    pub message_id: [u8; 16],
    /// Signed end-to-end encrypted envelope; relay code must not inspect its ciphertext.
    pub envelope: ChatEnvelope,
}

/// One stable ChatPullV2 page and the opaque continuation state for it.
#[derive(Debug)]
pub struct PendingMessagePageV2 {
    /// Valid signed envelopes returned to the authenticated receiver.
    pub messages: Vec<PendingMessage>,
    /// AEAD-protected continuation cursor. An empty request cursor starts a new snapshot.
    pub next_cursor: Vec<u8>,
    /// Whether the caller should continue the current snapshot with `next_cursor`.
    pub has_more: bool,
}
