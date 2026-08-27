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
