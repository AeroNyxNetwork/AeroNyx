// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_blob_facade.rs
// ============================================
// Version: 1.0.0-EncryptedBlobFacade
//
// Creation Reason:
//   [CHAT-BLOB-FACADE-DOMAIN 2026-08-28 by Codex] Move opaque encrypted-blob
//   storage, retrieval, and sender-authorized deletion APIs out of the relay
//   composition root without changing identifiers, quotas, or durable format.
//
// Main Functionality:
//   - Stores bounded opaque ciphertext under node and receiver quotas.
//   - Retrieves ciphertext by its node-private HMAC-derived identifier.
//   - Deletes ciphertext only for the sender bound at creation time.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns the composed service and database lock.
//   - `EncryptedBlobCustodyDomain` owns validation, identity, and persistence.
//
// Main Logical Flow:
//   1. Prepare and validate the blob command before taking the DB lock.
//   2. Execute one bounded custody operation through the domain component.
//   3. Emit aggregate size/action telemetry without content or identities.
//   4. Return only the opaque blob identifier or ciphertext requested.
//
// Important Note for Next Developer:
//   - The relay must never parse, decrypt, classify, or inspect blob contents.
//   - Keep identifiers node-private and sender authorization fail-closed.
//   - Do not log blob IDs, sender/receiver keys, hashes, ciphertext, or paths.
//   - Quota and durable-write behavior belongs to the custody domain.
//
// Last Modified:
//   v1.0.0-EncryptedBlobFacade - Initial encrypted-blob facade extraction
// ============================================

use tracing::{debug, info};

use crate::services::chat_relay_blob_custody::EncryptedBlobStoreOutcome;

use super::{now_secs, ChatRelayResult, ChatRelayService};

impl ChatRelayService {
    /// Stores one opaque encrypted blob under node-wide and receiver quotas.
    ///
    /// # Errors
    ///
    /// Returns an item-size, capacity, serialization, or `SQLite` error.
    pub fn put_blob(
        &self,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        data: &[u8],
        file_hash: &[u8; 32],
    ) -> ChatRelayResult<String> {
        let write = self
            .blob_custody
            .prepare_put(sender, receiver, data, file_hash, now_secs())?;
        let mut conn = self.conn.lock();
        let outcome = self.blob_custody.put(&mut conn, write)?;
        drop(conn);

        if let EncryptedBlobStoreOutcome::Stored { size, .. } = &outcome {
            info!(size = *size, "[CHAT_RELAY] Encrypted blob stored");
        }
        Ok(outcome.blob_id().to_owned())
    }

    /// Retrieves an opaque encrypted blob by its HMAC-derived identifier.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error or [`super::ChatRelayError::BlobNotFound`].
    pub fn get_blob(&self, blob_id: &str) -> ChatRelayResult<Vec<u8>> {
        let conn = self.conn.lock();
        let data = self.blob_custody.get(&conn, blob_id)?;
        drop(conn);
        debug!(size = data.len(), "[CHAT_RELAY] Encrypted blob retrieved");
        Ok(data)
    }

    /// Deletes an encrypted blob when requested by its original sender.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite`, not-found, or authorization error.
    pub fn delete_blob(&self, blob_id: &str, requester: &[u8; 32]) -> ChatRelayResult<()> {
        let conn = self.conn.lock();
        self.blob_custody.delete(&conn, blob_id, requester)?;
        drop(conn);
        info!("[CHAT_RELAY] Encrypted blob deleted by authorized sender");
        Ok(())
    }
}
