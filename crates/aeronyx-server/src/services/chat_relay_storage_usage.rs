// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_storage_usage.rs
// ============================================
// Version: 1.0.0-StorageUsageReadDomain
//
// Creation Reason:
//   [CHAT-STORAGE-USAGE-DOMAIN 2026-08-28 by Codex] Extract aggregate relay
//   storage accounting reads from the relay orchestration service.
//
// Main Functionality:
//   - Defines the public privacy-safe aggregate storage usage snapshot.
//   - Defines a replaceable read-only storage usage repository capability.
//   - Reads the authoritative singleton counters from SQLite.
//   - Rejects corrupt negative counters instead of coercing them to zero.
//
// Dependencies:
//   - `rusqlite` supplies the production aggregate counter repository.
//   - `serde` preserves the existing API and heartbeat serialization contract.
//   - `chat_relay_error.rs` owns stable path-free corruption failures.
//
// Main Logical Flow:
//   1. Read all four counters from the singleton accounting row.
//   2. Convert every signed SQLite integer through a fail-closed boundary.
//   3. Return only aggregate counts and bytes without any routing identity.
//
// Important Note for Next Developer:
//   - Never add wallet, sender, receiver, route, or payload labels here.
//   - Do not silently clamp negative values; they indicate durable corruption.
//   - Keep this query aligned with storage-schema accounting transactions.
//   - Canonical row reconciliation remains recovery-image certification work.
//
// Last Modified:
//   v1.0.0-StorageUsageReadDomain - Initial repository extraction
// ============================================

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Aggregate durable relay usage with no user or routing identifiers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayStorageUsage {
    /// Active pending message rows.
    pub pending_messages: u64,
    /// Encoded bytes held by active pending messages.
    pub pending_message_bytes: u64,
    /// Pending encrypted blob rows.
    pub pending_blobs: u64,
    /// Encrypted blob bytes retained by the node.
    pub pending_blob_bytes: u64,
}

/// Read-only aggregate storage accounting capability.
pub(crate) trait RelayStorageUsageRepository {
    /// Reads the current privacy-safe aggregate accounting snapshot.
    fn read(&self, connection: &Connection) -> ChatRelayResult<ChatRelayStorageUsage>;
}

/// SQLite implementation backed by the authoritative singleton counter row.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SqliteRelayStorageUsageRepository;

impl RelayStorageUsageRepository for SqliteRelayStorageUsageRepository {
    fn read(&self, connection: &Connection) -> ChatRelayResult<ChatRelayStorageUsage> {
        let counters = connection.query_row(
            "SELECT
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             FROM relay_storage_usage
             WHERE singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )?;
        Ok(ChatRelayStorageUsage {
            pending_messages: nonnegative(counters.0, "pending_message_count")?,
            pending_message_bytes: nonnegative(counters.1, "pending_message_bytes")?,
            pending_blobs: nonnegative(counters.2, "pending_blob_count")?,
            pending_blob_bytes: nonnegative(counters.3, "pending_blob_bytes")?,
        })
    }
}

fn nonnegative(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}
