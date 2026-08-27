// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_blind_route_coordinator.rs
// ============================================
// Version: 1.0.0-BlindRouteCoordinator
//
// Creation Reason:
//   [BLIND-ROUTE-COORDINATOR-DOMAIN 2026-08-28 by Codex] Compose private
//   blind-route identity, exact-response protection, and durable owner-fenced
//   state outside the oversized relay orchestration service.
//
// Main Functionality:
//   - Derives node-private route keys and request fingerprints.
//   - Reserves bounded durable route ownership before external effects.
//   - Arms an owned claim immediately before peel, forwarding, or custody.
//   - Releases only the current process's claim when no effect was armed.
//   - Protects and atomically persists an exact opaque response for retries.
//   - Authenticates completed responses recovered after process restart.
//
// Dependencies:
//   - `chat_relay_blind_route.rs` owns private identity and response AEAD.
//   - `chat_relay_blind_route_store.rs` owns SQLite state transitions.
//   - The relay service supplies connection, process epoch, time, and telemetry.
//
// Main Logical Flow:
//   1. Derive private identifiers only after the request is authenticated.
//   2. Reserve one owner-fenced durable claim before any external effect.
//   3. Return a recovered exact response only after AEAD authentication.
//   4. Arm the claim at the final boundary before an external effect begins.
//   5. Seal the exact response and atomically replace the owned reservation.
//
// Important Note for Next Developer:
//   - Never expose or log route ids, commitments, derived identifiers, epochs,
//     protected responses, or plaintext responses from this boundary.
//   - Only an armed foreign-process claim may become recovery work; preserve
//     the durable repository's fail-closed takeover and owner-CAS semantics.
//   - The response remains opaque. Do not parse it or add protocol coupling.
//   - Recovery telemetry remains service-owned and aggregate-only.
//
// Last Modified:
//   v1.0.0-BlindRouteCoordinator - Initial use-case composition
// ============================================

use parking_lot::Mutex;
use rusqlite::Connection;

use super::chat_relay_blind_route::{BlindRelayRouteAdmission, BlindRouteReplay};
use super::chat_relay_blind_route_store::{
    BlindRouteDurableRepository, DurableBlindRouteAdmission, SqliteBlindRouteDurableStore,
};
use super::chat_relay_error::ChatRelayResult;

/// Complete blind-route replay and durable ownership use-case coordinator.
pub(crate) struct BlindRouteCoordinator {
    replay: BlindRouteReplay,
    store: SqliteBlindRouteDurableStore,
}

impl BlindRouteCoordinator {
    /// Composes private replay identity with restart-safe durable ownership.
    pub(crate) fn new(
        node_secret: [u8; 32],
        replay_ttl_secs: u64,
        capacity: usize,
        owner_takeover_grace_secs: u64,
    ) -> ChatRelayResult<Self> {
        Ok(Self {
            replay: BlindRouteReplay::new(node_secret)?,
            store: SqliteBlindRouteDurableStore::new(
                replay_ttl_secs,
                capacity,
                owner_takeover_grace_secs,
            ),
        })
    }

    /// Reserves one authenticated route or recovers its sealed exact response.
    pub(crate) fn reserve(
        &self,
        connection: &Mutex<Connection>,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        process_epoch: &[u8],
        now: u64,
    ) -> ChatRelayResult<BlindRelayRouteAdmission> {
        let cache_key = self.replay.cache_key(route_id);
        let request_fingerprint = self.replay.request_fingerprint(request_commitment);

        match self.store.reserve(
            connection,
            &cache_key,
            &request_fingerprint,
            process_epoch,
            now,
        )? {
            DurableBlindRouteAdmission::Reserved => Ok(BlindRelayRouteAdmission::Reserved),
            DurableBlindRouteAdmission::ReservedForRecovery => {
                Ok(BlindRelayRouteAdmission::ReservedForRecovery)
            }
            DurableBlindRouteAdmission::Pending => Ok(BlindRelayRouteAdmission::Pending),
            DurableBlindRouteAdmission::Conflict => Ok(BlindRelayRouteAdmission::Conflict),
            DurableBlindRouteAdmission::CapacityExhausted => {
                Ok(BlindRelayRouteAdmission::CapacityExhausted)
            }
            DurableBlindRouteAdmission::Completed(durable) => {
                let response = self.replay.recover_response(
                    &cache_key,
                    &request_fingerprint,
                    &durable.nonce,
                    &durable.ciphertext,
                )?;
                Ok(BlindRelayRouteAdmission::Completed {
                    response,
                    completed_at: durable.completed_at,
                })
            }
        }
    }

    /// Arms this process's exact claim before its first external effect.
    pub(crate) fn arm_effect(
        &self,
        connection: &Mutex<Connection>,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        process_epoch: &[u8],
        started_at: u64,
    ) -> ChatRelayResult<()> {
        let cache_key = self.replay.cache_key(route_id);
        let request_fingerprint = self.replay.request_fingerprint(request_commitment);
        self.store.arm_effect(
            connection,
            &cache_key,
            &request_fingerprint,
            process_epoch,
            started_at,
        )
    }

    /// Releases only an unarmed claim owned by this exact process epoch.
    pub(crate) fn release_unarmed(
        &self,
        connection: &Mutex<Connection>,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        process_epoch: &[u8],
    ) -> ChatRelayResult<bool> {
        let cache_key = self.replay.cache_key(route_id);
        let request_fingerprint = self.replay.request_fingerprint(request_commitment);
        self.store
            .release_unarmed(connection, &cache_key, &request_fingerprint, process_epoch)
    }

    /// Seals and persists one exact opaque response for restart-safe replay.
    pub(crate) fn remember_response(
        &self,
        connection: &Mutex<Connection>,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        process_epoch: &[u8],
        response: &[u8],
        completed_at: u64,
    ) -> ChatRelayResult<()> {
        let cache_key = self.replay.cache_key(route_id);
        let request_fingerprint = self.replay.request_fingerprint(request_commitment);
        let protected = self
            .replay
            .protect_response(&cache_key, &request_fingerprint, response)?;
        self.store.complete(
            connection,
            &cache_key,
            &request_fingerprint,
            process_epoch,
            protected,
            completed_at,
        )
    }
}
