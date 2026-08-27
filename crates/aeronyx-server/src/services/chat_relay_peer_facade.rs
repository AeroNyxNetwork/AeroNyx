// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_peer_facade.rs
// ============================================
// Version: 1.1.0-OnlineAdmissionFacade
//
// Creation Reason:
//   [CHAT-PEER-FACADE-DOMAIN 2026-08-28 by Codex] Move privacy-safe relay
//   observations, direct-peer circuit permits, verified-submit outcomes, and
//   peer status snapshots out of the relay composition root.
//
// Modification Reason:
//   [CHAT-ONLINE-ADMISSION-FACADE-DOMAIN 2026-08-28 by Codex] Co-locate the
//   bounded live-path duplicate admission API with online relay operations.
//
// Main Functionality:
//   - Admits each live-path message identifier at most once per process window.
//   - Records typed direct-peer and authenticated-onion relay rounds.
//   - Owns the begin/cancel/complete lifecycle for direct-peer circuit permits.
//   - Records aggregate verified-submit, inbound, and recovery outcomes.
//   - Returns one privacy-safe peer relay and circuit status snapshot.
//
// Dependencies:
//   - Parent `chat_relay.rs` owns the composed service and stable re-exports.
//   - Direct-peer circuit owns durable restart-safe admission state.
//   - Peer telemetry owns process-local aggregate counters and SLO windows.
//
// Main Logical Flow:
//   1. Reject a duplicate live-path identifier through bounded in-memory state.
//   2. Sanitize compatibility failure strings into closed typed buckets.
//   3. Admit direct delivery through the durable circuit when required.
//   4. Return only aggregate relay and circuit status to public consumers.
//
// Important Note for Next Developer:
//   - Never add peer, endpoint, wallet, message, route, receipt, or payload IDs.
//   - Every acquired circuit permit must be completed or explicitly cancelled.
//   - A circuit persistence failure must continue to fail closed.
//   - Compatibility strings must remain sanitized into the closed reason enum.
//
// Last Modified:
//   v1.1.0-OnlineAdmissionFacade - Co-located live-path duplicate admission
//   v1.0.0-PeerRelayFacade - Initial peer relay facade extraction
// ============================================

use crate::services::chat_relay_peer_telemetry::{
    BlindRouteRecoveryEvent, OutboundRouteClass, PeerRelayTelemetrySink, VerifiedSubmitEvent,
};

use super::{
    now_secs, ChatRelayDirectPeerPermit, ChatRelayInboundFailureReason,
    ChatRelayOutboundFailureReason, ChatRelayPeerStatus, ChatRelayService,
    VerifiedSubmitRecoveryOutcome,
};

impl ChatRelayService {
    /// Returns `true` when this message was already admitted on the live path.
    pub fn is_online_duplicate(&self, message_id: &[u8; 16]) -> bool {
        self.dedup.check_and_insert(message_id)
    }

    /// Records a compatibility direct node-to-node encrypted relay round.
    ///
    /// The backward-compatible aggregate fields are also advanced.
    /// The failure reason must be a stable bucket such as
    /// `peer_relay_request_timeout` or `peer_relay_http_503`; do not pass peer
    /// URLs, message IDs, wallet IDs, client IPs, or payload-derived data.
    /// Unrecognized legacy input is intentionally reported as `unknown`.
    pub fn record_peer_relay_outbound(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
    ) {
        self.record_peer_relay_outbound_typed(
            now,
            attempted,
            accepted,
            failure_reason
                .as_deref()
                .map(ChatRelayOutboundFailureReason::from_bucket),
        );
    }

    /// Records direct relay health through the compiler-checked reason type.
    pub(crate) fn record_peer_relay_outbound_typed(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
    ) {
        self.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            OutboundRouteClass::DirectPeer,
        );
    }

    /// Begins one target-bound direct relay delivery when the circuit permits.
    ///
    /// A returned permit must be completed after a network outcome or cancelled
    /// if local request construction fails before network I/O. `None` means the
    /// caller must fail closed without falling back to an older relay protocol.
    pub(crate) fn begin_direct_peer_delivery(&self, now: u64) -> Option<ChatRelayDirectPeerPermit> {
        // [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Admission is intentionally
        // process-global and source-blind; per-peer quarantine remains owned by
        // PeerStore and must not be duplicated with identity-labelled state here.
        self.direct_peer_relay_circuit.begin(&self.conn, now)
    }

    /// Releases an unused half-open permit after a local preflight failure.
    pub(crate) fn cancel_direct_peer_delivery(&self, now: u64, permit: ChatRelayDirectPeerPermit) {
        self.direct_peer_relay_circuit
            .cancel(&self.conn, now, permit);
    }

    /// Completes one aggregate target-bound direct relay delivery observation.
    ///
    /// `retry_triggered` means a second exact request was sent. A triggered
    /// retry is classified as either recovered or exhausted. Independently,
    /// `final_failure_deterministic` records that the final typed failure was
    /// not eligible for another ambiguity retry. Every delivery updates the
    /// recent SLO window, while successful first attempts leave the lifetime
    /// exception counters unchanged.
    ///
    /// [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] Parameters are
    /// deliberately aggregate booleans. Do not extend this API with peer,
    /// message, commitment, endpoint, wallet, or payload identifiers.
    pub(crate) fn complete_direct_peer_delivery(
        &self,
        now: u64,
        permit: ChatRelayDirectPeerPermit,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) -> bool {
        debug_assert!(!(delivery_succeeded && final_failure_deterministic));
        // [DIRECT-RELAY-SLO 2026-08-15 by Codex] Every v3 delivery contributes
        // exactly one aggregate sample, including successful first attempts.
        // The fixed ring retains no event identity or peer dimension.
        let observe_slo_failed = || {
            self.peer_telemetry.record_direct_peer_delivery(
                now,
                retry_triggered,
                delivery_succeeded,
                final_failure_deterministic,
            )
        };
        self.direct_peer_relay_circuit.complete(
            &self.conn,
            now,
            permit,
            delivery_succeeded,
            observe_slo_failed,
        )
    }

    /// Records one authenticated receipt-verified onion relay round.
    ///
    /// [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] This advances both the
    /// backward-compatible aggregate and the authenticated-onion snapshot.
    /// A later compatibility fallback may update the aggregate but cannot
    /// erase the authenticated path's latest evidence.
    pub fn record_authenticated_onion_outbound(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
    ) {
        self.record_authenticated_onion_outbound_typed(
            now,
            attempted,
            accepted,
            failure_reason
                .as_deref()
                .map(ChatRelayOutboundFailureReason::from_bucket),
        );
    }

    /// Records authenticated onion health through the validated reason type.
    pub(crate) fn record_authenticated_onion_outbound_typed(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
    ) {
        self.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            OutboundRouteClass::AuthenticatedOnion,
        );
    }

    fn record_outbound_round(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
        route_class: OutboundRouteClass,
    ) {
        self.peer_telemetry.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            route_class,
        );
    }

    /// Records the closed aggregate result of one explicit verified submit.
    pub(crate) fn record_verified_submit_result(&self, now: u64, result: u8) {
        // [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] This is a
        // single node-wide counter update. Do not attach request, message,
        // route, endpoint, wallet, receipt, or payload dimensions here.
        self.peer_telemetry
            .record_verified_submit(now, result, VerifiedSubmitEvent::Closed);
    }

    /// Records one exact retry served without repeating route or custody work.
    pub(crate) fn record_verified_submit_replay(&self, now: u64, result: u8) {
        self.peer_telemetry
            .record_verified_submit(now, result, VerifiedSubmitEvent::Replay);
    }

    /// Records fail-closed reuse of a request id for a different envelope.
    pub(crate) fn record_verified_submit_conflict(&self, now: u64, result: u8) {
        self.peer_telemetry
            .record_verified_submit(now, result, VerifiedSubmitEvent::Conflict);
    }

    /// Records a crash-left exact request rejected before repeating effects.
    pub(crate) fn record_verified_submit_pending_rejection(&self, now: u64, result: u8) {
        self.peer_telemetry.record_verified_submit(
            now,
            result,
            VerifiedSubmitEvent::PendingRejection,
        );
    }

    /// Records admission saturation without exposing retained request metadata.
    pub(crate) fn record_verified_submit_capacity_rejection(&self, now: u64, result: u8) {
        self.peer_telemetry.record_verified_submit(
            now,
            result,
            VerifiedSubmitEvent::CapacityRejection,
        );
    }

    /// Records one foreign-owner takeover after durable admission commits.
    pub(crate) fn record_verified_submit_recovery_attempted(&self, now: u64) {
        self.peer_telemetry
            .record_verified_submit_recovery_attempted(now);
    }

    /// Records one closed restart-recovery transition without request data.
    pub(crate) fn record_verified_submit_recovery_outcome(
        &self,
        now: u64,
        outcome: VerifiedSubmitRecoveryOutcome,
    ) {
        // [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] Keep this
        // process-local aggregate independent from the normal protocol result
        // counters; one recovered request must not be counted as two submits.
        self.peer_telemetry
            .record_verified_submit_recovery_outcome(now, outcome);
    }

    /// Records an accepted inbound peer relay request.
    pub fn record_peer_relay_inbound_accepted(
        &self,
        now: u64,
        duplicate: bool,
        delivered_online: usize,
        stored_pending: bool,
    ) {
        self.peer_telemetry.record_inbound_accepted(
            now,
            duplicate,
            delivered_online,
            stored_pending,
        );
    }

    /// Records a rejected inbound peer relay request with a stable reason bucket.
    ///
    /// This compatibility entry point sanitizes unrecognized text to `unknown`.
    pub fn record_peer_relay_inbound_rejected(&self, now: u64, reason: impl Into<String>) {
        let reason = reason.into();
        self.record_peer_relay_inbound_rejected_typed(
            now,
            ChatRelayInboundFailureReason::from_bucket(&reason),
        );
    }

    /// Records an inbound rejection through the validated reason type.
    pub(crate) fn record_peer_relay_inbound_rejected_typed(
        &self,
        now: u64,
        reason: ChatRelayInboundFailureReason,
    ) {
        self.peer_telemetry.record_inbound_rejected(now, reason);
    }

    /// Returns a privacy-safe node-to-node relay health snapshot.
    #[must_use]
    pub fn peer_status(&self) -> ChatRelayPeerStatus {
        // [CHAT-PEER-TELEMETRY-DOMAIN 2026-08-26 by Codex] Circuit state stays
        // independently durable; all process telemetry is copied atomically.
        let now = now_secs();
        let circuit = self.direct_peer_relay_circuit.snapshot(now);
        self.peer_telemetry.snapshot(now, circuit)
    }

    /// Records one foreign-owner armed-route recovery attempt.
    pub(crate) fn record_blind_route_recovery_attempted(&self, now: u64) {
        // [CHAT-PEER-FACADE-DOMAIN 2026-08-28 by Codex] Keep all closed
        // recovery-event mapping inside the aggregate telemetry facade.
        self.peer_telemetry
            .record_blind_route_recovery(now, BlindRouteRecoveryEvent::Attempted);
    }

    /// Records a successfully sealed ACK for an armed route takeover.
    pub(crate) fn record_blind_route_recovery_completed(&self, now: u64) {
        self.peer_telemetry
            .record_blind_route_recovery(now, BlindRouteRecoveryEvent::Completed);
    }

    /// Records an armed takeover retained for a later exact retry.
    pub(crate) fn record_blind_route_recovery_deferred(&self, now: u64) {
        self.peer_telemetry
            .record_blind_route_recovery(now, BlindRouteRecoveryEvent::Deferred);
    }
}
