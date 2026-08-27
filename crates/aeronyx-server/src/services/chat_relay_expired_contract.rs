// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_expired_contract.rs
// ============================================
// Version: 1.0.0-ExpiredNotificationContract
//
// Creation Reason:
//   [CHAT-EXPIRED-CONTRACT-DOMAIN 2026-08-28 by Codex] Move the public expiry
//   notification model and its bounded decoding invariants out of the
//   oversized relay orchestration service.
//
// Main Functionality:
//   - Defines the stable queued `ChatExpired` notification contract.
//   - Bounds persisted bincode bytes before deserialization.
//   - Rejects empty or oversized message-id batches after deserialization.
//   - Keeps durable row fields explicit for authenticated delivery lookup.
//
// Dependencies:
//   - `chat_relay_error.rs` supplies the stable typed corruption boundary.
//   - `chat_relay_expired_delivery.rs` validates durable rows into this model.
//   - `chat_relay_cleanup.rs` uses the same exported encoding limits.
//   - `chat_relay.rs` re-exports all established public and crate-local paths.
//
// Main Logical Flow:
//   1. Reject encoded payloads above the fixed defensive byte ceiling.
//   2. Decode the exact fixed-width message-id vector.
//   3. Reject empty or over-batched decoded collections.
//   4. Return validated message ids without exposing storage implementation.
//
// Important Note for Next Developer:
//   - Preserve the existing field layout and `chat_relay` re-export paths.
//   - Bounds must be checked both before and after deserialization.
//   - Do not log sender, receiver, notification id, or message ids here.
//   - Changing either limit changes cleanup and delivery admission together.
//
// Last Modified:
//   v1.0.0-ExpiredNotificationContract - Initial contract extraction
// ============================================

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Maximum IDs encoded into one authenticated `ChatExpired` frame.
pub(crate) const MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION: usize = 32;
/// Defensive ceiling for one persisted bincode notification payload.
pub(crate) const MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES: usize = 1024;

/// A queued `ChatExpired` notification for an offline sender.
#[derive(Debug)]
pub struct ExpiredNotification {
    /// Local notification row identifier.
    pub id: i64,
    /// Original sender public key used only for authenticated delivery lookup.
    pub sender: [u8; 32],
    /// Original receiver public key returned inside the encrypted client flow.
    pub receiver: [u8; 32],
    /// Bincode-serialised `Vec<[u8; 16]>`.
    pub message_ids_raw: Vec<u8>,
}

impl ExpiredNotification {
    /// Deserialises the bounded stored message-id collection.
    pub fn message_ids(&self) -> ChatRelayResult<Vec<[u8; 16]>> {
        if self.message_ids_raw.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_payload_size",
            });
        }
        let message_ids: Vec<[u8; 16]> = bincode::deserialize(&self.message_ids_raw)?;
        if message_ids.is_empty() || message_ids.len() > MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_message_count",
            });
        }
        Ok(message_ids)
    }
}
