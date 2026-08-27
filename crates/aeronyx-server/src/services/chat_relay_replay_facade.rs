// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_replay_facade.rs
// ============================================
// Version: 1.0.0-DurableReplayFacade
//
// Creation Reason:
//   [CHAT-REPLAY-FACADE-DOMAIN 2026-08-28 by Codex] Move verified-submit and
//   blind-route durable replay workflows out of the relay composition root
//   while preserving owner fencing, effect arming, and exact-response replay.
//
// Main Functionality:
//   - Serializes equal verified-submit requests through fixed lock lanes.
//   - Looks up, reserves, recovers, and completes verified-submit replay state.
//   - Reserves, arms, releases, and completes blind-route replay state.
//   - Bridges recovery admissions to aggregate-only peer telemetry methods.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns the composed service and stable re-exports.
//   - Verified-submit coordinator owns private request binding and replay state.
//   - Blind-route coordinator owns private route binding and replay state.
//
// Main Logical Flow:
//   1. Derive private replay identity inside the selected coordinator.
//   2. Reserve durable owner-fenced state before any external side effect.
//   3. Arm blind-route claims immediately before the first external effect.
//   4. Persist an exact sealed response or retain fail-closed recovery evidence.
//
// Important Note for Next Developer:
//   - Never persist or log raw request IDs, route IDs, commitments, or payloads.
//   - Hold verified-submit lane guards through lookup, effects, and completion.
//   - Never release an armed blind-route claim as if no effect occurred.
//   - Reservation and response persistence failures must remain fail-closed.
//
// Last Modified:
//   v1.0.0-DurableReplayFacade - Initial durable replay facade extraction
// ============================================

use aeronyx_core::protocol::memchain::{
    ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1,
};

use super::{
    now_secs, BlindRelayRouteAdmission, ChatRelayResult, ChatRelayService, VerifiedSubmitAdmission,
    VerifiedSubmitCacheLookup,
};

impl ChatRelayService {
    /// Serializes requests sharing one private sender/request-id cache key.
    ///
    /// Unrelated submissions remain concurrent across fixed lock lanes. The
    /// caller must hold the returned guard through lookup, relay/custody, and
    /// response insertion so duplicate requests cannot both become leaders.
    pub(crate) async fn lock_verified_submit(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> tokio::sync::MutexGuard<'_, ()> {
        self.verified_submit.lock(request).await
    }

    /// Looks up a completed response after request authentication.
    pub(crate) fn verified_submit_cache_lookup(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> ChatRelayResult<VerifiedSubmitCacheLookup> {
        self.verified_submit.lookup(&self.conn, request, now_secs())
    }

    /// Atomically reserves one private replay slot before any external effect.
    pub(crate) fn reserve_verified_submit(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> ChatRelayResult<VerifiedSubmitAdmission> {
        let now = now_secs();
        let outcome = self.verified_submit.reserve(
            &self.conn,
            request,
            self.replay_process_epoch.as_slice(),
            now,
        )?;
        if matches!(outcome, VerifiedSubmitAdmission::ReservedForEntryRecovery) {
            // [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex]
            // Admission remains the authoritative attempted transition.
            self.record_verified_submit_recovery_attempted(now);
        }
        Ok(outcome)
    }

    /// Retains one completed response for exact retry replay across restarts.
    pub(crate) fn remember_verified_submit_response(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
        response: &ChatRelayVerifiedSubmitResponseV1,
    ) -> ChatRelayResult<()> {
        self.verified_submit.remember_response(
            &self.conn,
            request,
            response,
            self.replay_process_epoch.as_slice(),
            now_secs(),
        )
    }

    /// Reserves one authenticated blind route before peel, forward, or store.
    pub(crate) fn reserve_blind_relay_route(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
    ) -> ChatRelayResult<BlindRelayRouteAdmission> {
        let now = now_secs();
        let admission = self.blind_route.reserve(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
            now,
        )?;
        if matches!(admission, BlindRelayRouteAdmission::ReservedForRecovery) {
            self.record_blind_route_recovery_attempted(now);
        }
        Ok(admission)
    }

    /// Arms an owned route claim immediately before its first external effect.
    pub(crate) fn arm_blind_relay_route_effect(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        started_at: u64,
    ) -> ChatRelayResult<()> {
        self.blind_route.arm_effect(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
            started_at,
        )
    }

    /// Releases only this process's claim when no external effect was armed.
    pub(crate) fn release_unarmed_blind_relay_route(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
    ) -> ChatRelayResult<bool> {
        self.blind_route.release_unarmed(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
        )
    }

    /// Atomically replaces one route reservation with its sealed exact ACK.
    pub(crate) fn remember_blind_relay_route_response(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        response: &[u8],
        completed_at: u64,
    ) -> ChatRelayResult<()> {
        self.blind_route.remember_response(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
            response,
            completed_at,
        )
    }
}
