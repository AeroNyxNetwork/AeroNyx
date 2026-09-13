// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_verified_submit_coordinator.rs
// ============================================
// Version: 1.0.0-VerifiedSubmitCoordinator
//
// Creation Reason:
//   [VERIFIED-SUBMIT-COORDINATOR-DOMAIN 2026-08-28 by Codex] Compose private
//   verified-submit replay protection and durable ownership outside the relay
//   orchestration service.
//
// Main Functionality:
//   [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Separates read-only
//   completed replay from fresh effect admission and unbounded-age local cache.
//   - Serializes equal authenticated submissions through bounded lock lanes.
//   - Resolves process-local and durable replay state in one use-case boundary.
//   - Authenticates recovered sealed responses against their exact request.
//   - Reserves owner-fenced durable custody before external side effects.
//   - Seals and persists exact responses while retaining same-process replay.
//
// Dependencies:
//   - `chat_relay_verified_submit.rs` owns fingerprints, cache, and AEAD.
//   - `chat_relay_verified_submit_store.rs` owns SQLite state transitions.
//   - `aeronyx-core` owns verified-submit request and response wire contracts.
//   - The relay service retains aggregate telemetry and public API wrappers.
//
// Main Logical Flow:
//   1. Derive private cache and envelope fingerprints after authentication.
//   2. Prefer an exact process-local replay before reading durable state.
//   3. Recover and validate a durable exact response, then warm local replay.
//   4. Reserve one owner-fenced slot before the caller performs side effects.
//   5. Validate, seal, durably complete, and remember the exact response.
//
// Important Note for Next Developer:
//   - Never expose or log private fingerprints, process epochs, or ciphertext.
//   - Hold the returned single-flight guard through all external side effects.
//   - The caller supplies time so retry and recovery semantics remain explicit.
//   - Preserve local replay insertion even when durable completion returns an
//     error; this prevents duplicate same-process side effects after failure.
//   - Entry recovery telemetry remains service-owned and aggregate-only.
//
// Last Modified:
//   v1.0.0-VerifiedSubmitCoordinator - Initial use-case composition
// ============================================

use aeronyx_core::protocol::memchain::{
    ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1,
};
use parking_lot::Mutex;
use rusqlite::Connection;

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};
use super::chat_relay_verified_submit::{
    VerifiedSubmitAdmission, VerifiedSubmitCacheLookup, VerifiedSubmitReplay,
};
use super::chat_relay_verified_submit_store::{
    DurableVerifiedSubmitLookup, SqliteVerifiedSubmitDurableStore, VerifiedSubmitDurableRepository,
};

/// Complete verified-submit replay and durable ownership use-case coordinator.
pub(crate) struct VerifiedSubmitCoordinator {
    replay: VerifiedSubmitReplay,
    store: SqliteVerifiedSubmitDurableStore,
}

impl VerifiedSubmitCoordinator {
    /// Composes bounded process-local replay with restart-safe durable storage.
    pub(crate) fn new(
        node_secret: [u8; 32],
        capacity: usize,
        response_ttl_secs: u64,
        owner_takeover_grace_secs: u64,
    ) -> ChatRelayResult<Self> {
        Ok(Self {
            replay: VerifiedSubmitReplay::new(node_secret, capacity)?,
            store: SqliteVerifiedSubmitDurableStore::new(
                response_ttl_secs,
                capacity,
                owner_takeover_grace_secs,
            ),
        })
    }

    /// Serializes requests sharing one private sender/request-id cache key.
    pub(crate) async fn lock(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> tokio::sync::MutexGuard<'_, ()> {
        self.replay.lock(request).await
    }

    /// Resolves one authenticated request against local and durable replay.
    pub(crate) fn lookup(
        &self,
        connection: &Mutex<Connection>,
        request: &ChatRelayVerifiedSubmitRequestV1,
        now: u64,
    ) -> ChatRelayResult<VerifiedSubmitCacheLookup> {
        let cache_key = self.replay.cache_key(request);
        let envelope_fingerprint = self.replay.envelope_fingerprint(request);
        let memory_lookup = self.replay.lookup_cached(&cache_key, &envelope_fingerprint);
        if !matches!(memory_lookup, VerifiedSubmitCacheLookup::Miss) {
            return Ok(memory_lookup);
        }

        match self
            .store
            .lookup(connection, &cache_key, &envelope_fingerprint, now)?
        {
            DurableVerifiedSubmitLookup::Miss => Ok(VerifiedSubmitCacheLookup::Miss),
            DurableVerifiedSubmitLookup::Conflict => Ok(VerifiedSubmitCacheLookup::Conflict),
            DurableVerifiedSubmitLookup::Pending => Ok(VerifiedSubmitCacheLookup::Pending),
            DurableVerifiedSubmitLookup::Completed(durable) => {
                let response = self.replay.recover_response(
                    &cache_key,
                    &envelope_fingerprint,
                    &durable.nonce,
                    &durable.ciphertext,
                )?;
                response.validate_for_request(request).map_err(|_| {
                    ChatRelayError::CorruptStoredData {
                        field: "verified_submit_response_request_binding",
                    }
                })?;
                self.replay
                    .remember_cached(cache_key, envelope_fingerprint, response.clone());
                Ok(VerifiedSubmitCacheLookup::Exact(response))
            }
        }
    }

    /// Reads a retained completion without using or populating the memory cache.
    ///
    /// [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Equality remains the
    /// existing sender/request-id plus signed-envelope fingerprint, NOT equality
    /// of every request byte: a newly signed timestamp does not change the key.
    /// The handler must validate session ownership and both signatures first.
    pub(crate) fn lookup_completed_readonly(
        &self,
        connection: &Mutex<Connection>,
        request: &ChatRelayVerifiedSubmitRequestV1,
        now: u64,
    ) -> ChatRelayResult<Option<ChatRelayVerifiedSubmitResponseV1>> {
        let cache_key = self.replay.cache_key(request);
        let fingerprint = self.replay.envelope_fingerprint(request);
        let Some(durable) =
            self.store
                .lookup_completed_readonly(connection, &cache_key, &fingerprint, now)?
        else {
            return Ok(None);
        };
        let response = self.replay.recover_response(
            &cache_key,
            &fingerprint,
            &durable.nonce,
            &durable.ciphertext,
        )?;
        response
            .validate_for_request(request)
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_request_binding",
            })?;
        Ok(Some(response))
    }

    /// Atomically reserves one private replay slot before external effects.
    pub(crate) fn reserve(
        &self,
        connection: &Mutex<Connection>,
        request: &ChatRelayVerifiedSubmitRequestV1,
        process_epoch: &[u8],
        now: u64,
    ) -> ChatRelayResult<VerifiedSubmitAdmission> {
        let cache_key = self.replay.cache_key(request);
        let envelope_fingerprint = self.replay.envelope_fingerprint(request);
        self.store.reserve(
            connection,
            &cache_key,
            &envelope_fingerprint,
            process_epoch,
            now,
        )
    }

    /// Seals and persists one exact response for restart-safe retry replay.
    pub(crate) fn remember_response(
        &self,
        connection: &Mutex<Connection>,
        request: &ChatRelayVerifiedSubmitRequestV1,
        response: &ChatRelayVerifiedSubmitResponseV1,
        process_epoch: &[u8],
        now: u64,
    ) -> ChatRelayResult<()> {
        let cache_key = self.replay.cache_key(request);
        let envelope_fingerprint = self.replay.envelope_fingerprint(request);
        response
            .validate_for_request(request)
            .map_err(|_| ChatRelayError::VerifiedSubmitProtectionFailed)?;
        let protected =
            self.replay
                .protect_response(&cache_key, &envelope_fingerprint, response)?;
        let durable_result = self.store.complete(
            connection,
            &cache_key,
            &envelope_fingerprint,
            process_epoch,
            protected,
            now,
        );

        // [CRASH-SAFE-VERIFIED-SUBMIT 2026-08-24 by Codex] Preserve
        // same-process retry safety even if durable completion fails. The
        // caller receives only the typed storage failure and records no
        // request-derived values in logs or aggregate health.
        self.replay
            .remember_cached(cache_key, envelope_fingerprint, response.clone());
        durable_result
    }
}

#[cfg(test)]
mod stale_replay_tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::{ChatContentType, ChatEnvelope};
    use aeronyx_core::protocol::memchain::CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1;

    // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Real SQLite and
    // production AEAD; no unbounded-age cache hit or cache fill is permitted.
    #[test]
    fn verified_submit_stale_completed_replay_authenticates_without_memory_cache() {
        let connection = Mutex::new(Connection::open_in_memory().unwrap());
        connection
            .lock()
            .execute_batch(
                "CREATE TABLE relay_verified_submit_responses (
                cache_key BLOB PRIMARY KEY, envelope_fingerprint BLOB,
                response_nonce BLOB, response_ciphertext BLOB, completed_at INTEGER);
             CREATE TABLE relay_verified_submit_reservations (
                cache_key BLOB PRIMARY KEY, envelope_fingerprint BLOB,
                reserved_at INTEGER, owner_epoch BLOB, owner_acquired_at INTEGER);",
            )
            .unwrap();
        let sender = IdentityKeyPair::generate();
        let mut envelope = ChatEnvelope {
            message_id: [2; 16],
            sender: sender.public_key_bytes(),
            receiver: [3; 32],
            timestamp: 1000,
            ciphertext: vec![4; 16],
            nonce: [5; 24],
            content_type: ChatContentType::Text,
            signature: [0; 64],
        };
        envelope.signature = sender.sign(&envelope.sign_data());
        let request =
            ChatRelayVerifiedSubmitRequestV1::signed([6; 16], envelope, 1000, &sender).unwrap();
        let response = ChatRelayVerifiedSubmitResponseV1 {
            request_id: request.request_id,
            message_id: request.envelope.message_id,
            result: CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1,
            terminal_receipt: None,
        };
        let coordinator = VerifiedSubmitCoordinator::new([1; 32], 8, 121, 5).unwrap();
        assert_eq!(
            coordinator
                .reserve(&connection, &request, &[7; 16], 1000)
                .unwrap(),
            VerifiedSubmitAdmission::Reserved
        );
        coordinator
            .remember_response(&connection, &request, &response, &[7; 16], 1000)
            .unwrap();
        // Memory has a result, but its expired durable counterpart must not replay.
        assert!(coordinator
            .lookup_completed_readonly(&connection, &request, 1122)
            .unwrap()
            .is_none());
        let recovered = VerifiedSubmitCoordinator::new([1; 32], 8, 121, 5).unwrap();
        let key = recovered.replay.cache_key(&request);
        let fingerprint = recovered.replay.envelope_fingerprint(&request);
        let before: i64 = connection
            .lock()
            .query_row("SELECT total_changes()", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            recovered
                .lookup_completed_readonly(&connection, &request, 1121)
                .unwrap(),
            Some(response.clone())
        );
        assert!(matches!(
            recovered.replay.lookup_cached(&key, &fingerprint),
            VerifiedSubmitCacheLookup::Miss
        ));
        assert_eq!(
            connection
                .lock()
                .query_row("SELECT total_changes()", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            before
        );
        // Correct AEAD but wrong response message id still fails binding.
        let mut wrong_response = response;
        wrong_response.message_id = [9; 16];
        let protected = recovered
            .replay
            .protect_response(&key, &fingerprint, &wrong_response)
            .unwrap();
        connection.lock().execute(
            "UPDATE relay_verified_submit_responses SET response_nonce=?1, response_ciphertext=?2",
            rusqlite::params![protected.nonce.as_slice(), protected.ciphertext],
        ).unwrap();
        let before: i64 = connection
            .lock()
            .query_row("SELECT total_changes()", [], |r| r.get(0))
            .unwrap();
        assert!(recovered
            .lookup_completed_readonly(&connection, &request, 1061)
            .is_err());
        assert_eq!(
            connection
                .lock()
                .query_row("SELECT total_changes()", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            before
        );
        // Same legal length with unauthentic ciphertext must not leak plaintext.
        connection.lock().execute("UPDATE relay_verified_submit_responses SET response_ciphertext=zeroblob(length(response_ciphertext))", []).unwrap();
        let before: i64 = connection
            .lock()
            .query_row("SELECT total_changes()", [], |r| r.get(0))
            .unwrap();
        assert!(recovered
            .lookup_completed_readonly(&connection, &request, 1061)
            .is_err());
        assert_eq!(
            connection
                .lock()
                .query_row("SELECT total_changes()", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            before
        );
    }
}
