// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_peer_telemetry.rs
// ============================================
// Version: 1.2.0-OnionRouteBuildReasons
//
// Creation Reason:
//   [CHAT-PEER-TELEMETRY-DOMAIN 2026-08-26 by Codex] Extract privacy-safe
//   relay health classification and process-local aggregation from the
//   oversized chat relay orchestration service.
//
// Modification Reason:
//   [CHAT-RELAY-STATUS-CONTRACT-DOMAIN 2026-08-27 by Codex] Depend directly
//   on the focused status contract and its single-source policy defaults.
//   [ONION-ROUTE-BUILD-TELEMETRY 2026-08-29 by Codex] Accept bounded onion
//   route-construction dispositions without exposing route topology.
//
// Main Functionality:
//   - Defines validated inbound and outbound failure reason value objects.
//   - Models route classes and verified-submit recovery outcomes as enums.
//   - Maintains a bounded fixed-memory direct-peer SLO window.
//   - Atomically updates SLO and lifetime retry counters.
//   - Exposes a replaceable telemetry sink capability to the relay service.
//
// Dependencies:
//   - `chat_relay_status.rs` owns stable status contracts and policy defaults.
//   - `chat_relay_direct_peer_circuit.rs` owns durable admission state.
//   - `aeronyx_core::protocol::memchain` owns verified-submit result codes.
//
// Main Logical Flow:
//   1. Sanitize diagnostics into a closed operational vocabulary.
//   2. Record one aggregate observation under a single state lock.
//   3. Classify the bounded direct-peer SLO window without identities.
//   4. Compose the independent durable circuit snapshot at read time.
//
// Important Note for Next Developer:
//   - Never add peer, route, endpoint, wallet, message, or payload dimensions.
//   - Keep the failure vocabularies closed; unrecognized values are `unknown`.
//   - Preserve one-lock updates for SLO and lifetime direct-peer counters.
//   - Keep serialized field names and defaults in `chat_relay_status.rs` compatible.
//
// Last Modified:
//   v1.2.0-OnionRouteBuildReasons - Added closed onion route build outcomes
//   v1.1.0-StatusContractDependency - Consumed shared status contracts directly
//   v1.0.0-PeerRelayTelemetryDomain - Initial trait-based composition
// ============================================

use parking_lot::RwLock;

use aeronyx_core::protocol::memchain::{
    chat_verified_submit_result_label, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1,
    CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1, CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1,
    CHAT_VERIFIED_SUBMIT_REJECTED_V1,
};

#[cfg(test)]
use super::chat_relay_status::DIRECT_PEER_RETRY_SLO_WINDOW_SECS;
use super::chat_relay_status::{
    ChatRelayDirectPeerCircuitStatus, ChatRelayDirectPeerSloStatus, ChatRelayPeerStatus,
    DIRECT_PEER_RETRY_SLO_BUCKET_COUNT, DIRECT_PEER_RETRY_SLO_BUCKET_SECS,
    DIRECT_PEER_RETRY_SLO_FAILED_MIN_FAILURES, DIRECT_PEER_RETRY_SLO_FAILED_SUCCESS_BPS,
    DIRECT_PEER_RETRY_SLO_TARGET_BPS,
};

/// Closed aggregate outcomes after one entry-recovery admission.
// [CHAT-PEER-TELEMETRY-DOMAIN 2026-08-26 by Codex] The containing module is
// private, while this type is re-exported through the stable chat_relay path.
#[allow(clippy::redundant_pub_crate)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerifiedSubmitRecoveryOutcome {
    /// Entry custody and exact response persistence both completed.
    Completed,
    /// The attempt durably closed without entry custody.
    Failed,
    /// Response persistence failed, leaving durable retry evidence recoverable.
    Deferred,
}

impl VerifiedSubmitRecoveryOutcome {
    /// Derives the one closed recovery bucket from custody and replay state.
    pub(crate) const fn from_results(entry_custody: bool, response_persisted: bool) -> Self {
        // [CHAT-PEER-TELEMETRY-DOMAIN 2026-08-26 by Codex] Persistence failure
        // dominates because the owner-fenced reservation remains authoritative.
        if !response_persisted {
            Self::Deferred
        } else if entry_custody {
            Self::Completed
        } else {
            Self::Failed
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutboundRouteClass {
    AuthenticatedOnion,
    DirectPeer,
}

/// Validated outbound relay-health reason accepted by heartbeat telemetry.
#[allow(clippy::redundant_pub_crate)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatRelayOutboundFailureReason(String);

impl ChatRelayOutboundFailureReason {
    /// Sanitizes one internal diagnostic bucket for aggregate health export.
    pub(crate) fn from_bucket(reason: &str) -> Self {
        let safe = matches!(
            reason,
            "peer_http_client_unavailable"
                | "no_receipt_capable_terminal"
                | "no_network_diverse_receipt_path"
                | "no_receipt_capable_middle"
                | "onion_terminal_selection_changed"
                | "onion_terminal_diversity_exhausted"
                | "onion_middle_candidate_unavailable"
                | "onion_middle_endpoint_missing"
                | "onion_middle_endpoint_invalid"
                | "onion_request_build_failed"
                | "onion_payload_encoding_failed"
                | "onion_route_refresh_required"
                | "onion_route_policy_rejected"
                | "onion_route_local_construction_failed"
                | "onion_delivery_receipt_rejected"
                | "onion_delivery_route_surface_changed"
                | "onion_delivery_ack_response_too_large"
                | "onion_delivery_ack_response_body_read_failed"
                | "onion_delivery_ack_response_json_decode_failed"
                | "onion_delivery_request_timeout"
                | "onion_delivery_request_connect"
                | "onion_delivery_request_http_status"
                | "onion_delivery_request_decode"
                | "onion_delivery_request_body"
                | "onion_delivery_request_request"
                | "onion_delivery_request_unknown"
                | "peer_relay_circuit_open"
                | "peer_relay_half_open_probe_unavailable"
                | "peer_relay_target_auth_encode_failed"
                | "peer_relay_auth_encode_failed"
                | "peer_relay_request_timeout"
                | "peer_relay_request_connect"
                | "peer_relay_request_http_status"
                | "peer_relay_request_decode"
                | "peer_relay_request_body"
                | "peer_relay_request_request"
                | "peer_relay_request_unknown"
                | "peer_relay_ack_response_too_large"
                | "peer_relay_ack_response_body_read_failed"
                | "peer_relay_ack_response_json_decode_failed"
                | "peer_relay_ack_rejected"
                | "peer_relay_receipt_request_missing"
                | "peer_relay_receipt_missing"
                | "peer_relay_receipt_version_invalid"
                | "peer_relay_receipt_binding_invalid"
                | "peer_relay_receipt_signature_invalid"
                | "peer_relay_receipt_timestamp_in_future"
                | "peer_relay_receipt_timestamp_expired"
        ) || relay_http_status_bucket(reason, "onion_delivery_http_")
            || relay_http_status_bucket(reason, "onion_delivery_request_http_")
            || relay_http_status_bucket(reason, "peer_relay_http_")
            || relay_http_status_bucket(reason, "peer_relay_request_http_");

        Self(if safe { reason } else { "unknown" }.to_string())
    }

    fn into_bucket(self) -> String {
        self.0
    }
}

/// Validated inbound relay-health reason accepted by heartbeat telemetry.
#[allow(clippy::redundant_pub_crate)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatRelayInboundFailureReason(String);

impl ChatRelayInboundFailureReason {
    /// Sanitizes one inbound rejection bucket for aggregate health export.
    pub(crate) fn from_bucket(reason: &str) -> Self {
        let safe = matches!(
            reason,
            "rate_limited"
                | "backpressure"
                | "peer_auth_invalid"
                | "peer_target_mismatch"
                | "peer_auth_retry_in_flight"
                | "peer_auth_retry_cache_saturated"
                | "peer_auth_rate_limited"
                | "relay_unavailable"
                | "invalid_signature"
                | "envelope_too_large"
                | "envelope_serialization_failed"
                | "timestamp_expired"
                | "timestamp_in_future"
                | "pending_capacity_exhausted"
                | "store_pending_failed"
                | "sqlite_error"
                | "serialization_error"
                | "corrupt_stored_data"
                | "timestamp_out_of_range"
                | "message_id_conflict"
                | "queue_sequence_exhausted"
                | "message_too_large"
                | "mailbox_full"
                | "pending_message_count_quota"
                | "pending_message_byte_quota"
        );

        Self(if safe { reason } else { "unknown" }.to_string())
    }

    fn into_bucket(self) -> String {
        self.0
    }
}

fn relay_http_status_bucket(reason: &str, prefix: &str) -> bool {
    let Some(status) = reason.strip_prefix(prefix) else {
        return false;
    };
    status.len() == 3
        && status.bytes().all(|byte| byte.is_ascii_digit())
        && status
            .parse::<u16>()
            .is_ok_and(|status| (100..=599).contains(&status))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DirectPeerRetrySloBucket {
    initialized: bool,
    epoch_minute: u64,
    deliveries: u64,
    delivered: u64,
    retry_triggered: u64,
    retry_recovered: u64,
    retry_exhausted: u64,
    deterministic_failure: u64,
}

#[derive(Debug, Default)]
struct DirectPeerRetrySloWindow {
    buckets: [DirectPeerRetrySloBucket; DIRECT_PEER_RETRY_SLO_BUCKET_COUNT],
    latest_epoch_minute: u64,
}

impl DirectPeerRetrySloWindow {
    fn record(
        &mut self,
        now: u64,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) {
        let observed_epoch = now / DIRECT_PEER_RETRY_SLO_BUCKET_SECS;
        let epoch_minute = observed_epoch.max(self.latest_epoch_minute);
        self.latest_epoch_minute = epoch_minute;
        let index = (epoch_minute % DIRECT_PEER_RETRY_SLO_BUCKET_COUNT as u64) as usize;
        let bucket = &mut self.buckets[index];
        if !bucket.initialized || bucket.epoch_minute != epoch_minute {
            *bucket = DirectPeerRetrySloBucket {
                initialized: true,
                epoch_minute,
                ..DirectPeerRetrySloBucket::default()
            };
        }

        bucket.deliveries = bucket.deliveries.saturating_add(1);
        if delivery_succeeded {
            bucket.delivered = bucket.delivered.saturating_add(1);
        }
        if retry_triggered {
            bucket.retry_triggered = bucket.retry_triggered.saturating_add(1);
            if delivery_succeeded {
                bucket.retry_recovered = bucket.retry_recovered.saturating_add(1);
            } else {
                bucket.retry_exhausted = bucket.retry_exhausted.saturating_add(1);
            }
        }
        if final_failure_deterministic {
            bucket.deterministic_failure = bucket.deterministic_failure.saturating_add(1);
        }
    }

    fn snapshot(&self, now: u64) -> ChatRelayDirectPeerSloStatus {
        let current_epoch = (now / DIRECT_PEER_RETRY_SLO_BUCKET_SECS).max(self.latest_epoch_minute);
        let mut snapshot = ChatRelayDirectPeerSloStatus {
            evaluated_at: now,
            ..ChatRelayDirectPeerSloStatus::default()
        };
        for bucket in &self.buckets {
            if !bucket.initialized
                || bucket.epoch_minute > current_epoch
                || current_epoch.saturating_sub(bucket.epoch_minute)
                    >= DIRECT_PEER_RETRY_SLO_BUCKET_COUNT as u64
            {
                continue;
            }
            snapshot.deliveries_total = snapshot.deliveries_total.saturating_add(bucket.deliveries);
            snapshot.delivered_total = snapshot.delivered_total.saturating_add(bucket.delivered);
            snapshot.retry_triggered_total = snapshot
                .retry_triggered_total
                .saturating_add(bucket.retry_triggered);
            snapshot.retry_recovered_total = snapshot
                .retry_recovered_total
                .saturating_add(bucket.retry_recovered);
            snapshot.retry_exhausted_total = snapshot
                .retry_exhausted_total
                .saturating_add(bucket.retry_exhausted);
            snapshot.deterministic_failure_total = snapshot
                .deterministic_failure_total
                .saturating_add(bucket.deterministic_failure);
        }
        snapshot.failed_total = snapshot
            .deliveries_total
            .saturating_sub(snapshot.delivered_total);
        snapshot.delivery_success_bps =
            ratio_basis_points(snapshot.delivered_total, snapshot.deliveries_total);
        snapshot.retry_recovery_bps = ratio_basis_points(
            snapshot.retry_recovered_total,
            snapshot.retry_triggered_total,
        );
        snapshot.meets_slo = snapshot
            .delivery_success_bps
            .map(|ratio| ratio >= DIRECT_PEER_RETRY_SLO_TARGET_BPS);
        snapshot.status = if snapshot.deliveries_total == 0 {
            "idle"
        } else if snapshot.failed_total >= DIRECT_PEER_RETRY_SLO_FAILED_MIN_FAILURES
            && snapshot.delivery_success_bps.unwrap_or(0)
                <= DIRECT_PEER_RETRY_SLO_FAILED_SUCCESS_BPS
        {
            "failed"
        } else if snapshot.meets_slo == Some(true) {
            "healthy"
        } else {
            "degraded"
        }
        .to_string();
        snapshot
    }
}

fn ratio_basis_points(numerator: u64, denominator: u64) -> Option<u16> {
    if denominator == 0 {
        return None;
    }
    let basis_points = (u128::from(numerator).saturating_mul(10_000)) / u128::from(denominator);
    Some(basis_points.min(10_000) as u16)
}

#[derive(Debug)]
struct PeerRelayTelemetryState {
    status: ChatRelayPeerStatus,
    direct_peer_retry_slo: DirectPeerRetrySloWindow,
}

/// Replaceable aggregate telemetry capability consumed by relay orchestration.
pub(super) trait PeerRelayTelemetrySink: Send + Sync {
    fn record_outbound_round(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
        route_class: OutboundRouteClass,
    );
    fn record_direct_peer_delivery(
        &self,
        now: u64,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) -> bool;
    fn record_verified_submit(&self, now: u64, result: u8, event: VerifiedSubmitEvent);
    fn record_verified_submit_recovery_attempted(&self, now: u64);
    fn record_verified_submit_recovery_outcome(
        &self,
        now: u64,
        outcome: VerifiedSubmitRecoveryOutcome,
    );
    fn record_inbound_accepted(
        &self,
        now: u64,
        duplicate: bool,
        delivered_online: usize,
        stored_pending: bool,
    );
    fn record_inbound_rejected(&self, now: u64, reason: ChatRelayInboundFailureReason);
    fn record_blind_route_recovery(&self, now: u64, event: BlindRouteRecoveryEvent);
    fn snapshot(&self, now: u64, circuit: ChatRelayDirectPeerCircuitStatus) -> ChatRelayPeerStatus;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerifiedSubmitEvent {
    Closed,
    Replay,
    Conflict,
    PendingRejection,
    CapacityRejection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRouteRecoveryEvent {
    Attempted,
    Completed,
    Deferred,
}

/// In-memory process telemetry composed by `ChatRelayService`.
#[derive(Debug)]
pub(super) struct PeerRelayTelemetryDomain {
    state: RwLock<PeerRelayTelemetryState>,
}

impl PeerRelayTelemetryDomain {
    pub(super) fn new(status: ChatRelayPeerStatus) -> Self {
        Self {
            state: RwLock::new(PeerRelayTelemetryState {
                status,
                direct_peer_retry_slo: DirectPeerRetrySloWindow::default(),
            }),
        }
    }

    #[cfg(test)]
    pub(super) fn direct_peer_slo_snapshot(&self, now: u64) -> ChatRelayDirectPeerSloStatus {
        self.state.read().direct_peer_retry_slo.snapshot(now)
    }
}

impl PeerRelayTelemetrySink for PeerRelayTelemetryDomain {
    fn record_outbound_round(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
        route_class: OutboundRouteClass,
    ) {
        let failed = attempted.saturating_sub(accepted);
        let status_bucket = if attempted == 0 && failure_reason.is_some() {
            "failed"
        } else if attempted == 0 {
            "idle"
        } else if accepted == attempted {
            "healthy"
        } else if accepted > 0 {
            "degraded"
        } else {
            "failed"
        };
        let failure_reason = if failed > 0 || (attempted == 0 && status_bucket == "failed") {
            Some(failure_reason.map_or_else(
                || "unknown".to_string(),
                ChatRelayOutboundFailureReason::into_bucket,
            ))
        } else {
            None
        };

        let mut state = self.state.write();
        let status = &mut state.status;
        let route_status = match route_class {
            OutboundRouteClass::AuthenticatedOnion => &mut status.authenticated_onion_outbound,
            OutboundRouteClass::DirectPeer => &mut status.direct_peer_outbound,
        };
        route_status.attempted_total = route_status
            .attempted_total
            .saturating_add(attempted as u64);
        route_status.accepted_total = route_status.accepted_total.saturating_add(accepted as u64);
        route_status.failed_total = route_status.failed_total.saturating_add(failed as u64);
        route_status.rounds = route_status.rounds.saturating_add(1);
        route_status.last_attempted = attempted as u64;
        route_status.last_accepted = accepted as u64;
        route_status.last_failed = failed as u64;
        route_status.last_status = Some(status_bucket.to_string());
        route_status.last_failure_reason.clone_from(&failure_reason);
        route_status.last_at = Some(now);
        if accepted > 0 {
            route_status.consecutive_failures = 0;
            route_status.last_success_at = Some(now);
        } else if attempted > 0 || route_status.last_failure_reason.is_some() {
            route_status.consecutive_failures = route_status.consecutive_failures.saturating_add(1);
        }

        status.outbound_attempted_total = status
            .outbound_attempted_total
            .saturating_add(attempted as u64);
        status.outbound_accepted_total = status
            .outbound_accepted_total
            .saturating_add(accepted as u64);
        status.outbound_failed_total = status.outbound_failed_total.saturating_add(failed as u64);
        status.outbound_rounds = status.outbound_rounds.saturating_add(1);
        status.last_outbound_attempted = attempted as u64;
        status.last_outbound_accepted = accepted as u64;
        status.last_outbound_failed = failed as u64;
        status.last_outbound_status = Some(status_bucket.to_string());
        status.last_outbound_failure_reason = failure_reason;
        status.last_outbound_at = Some(now);
        if accepted > 0 {
            status.consecutive_outbound_failures = 0;
            status.last_outbound_success_at = Some(now);
        } else if attempted > 0 || status.last_outbound_failure_reason.is_some() {
            status.consecutive_outbound_failures =
                status.consecutive_outbound_failures.saturating_add(1);
        }
        drop(state);
    }

    fn record_direct_peer_delivery(
        &self,
        now: u64,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) -> bool {
        let mut state = self.state.write();
        state.direct_peer_retry_slo.record(
            now,
            retry_triggered,
            delivery_succeeded,
            final_failure_deterministic,
        );
        let slo_failed = state.direct_peer_retry_slo.snapshot(now).status == "failed";

        if retry_triggered || final_failure_deterministic {
            let retry = &mut state.status.direct_peer_retry;
            if retry_triggered {
                retry.retry_triggered_total = retry.retry_triggered_total.saturating_add(1);
                if delivery_succeeded {
                    retry.retry_recovered_total = retry.retry_recovered_total.saturating_add(1);
                    retry.last_outcome = Some("recovered".to_string());
                } else {
                    retry.retry_exhausted_total = retry.retry_exhausted_total.saturating_add(1);
                    retry.last_outcome = Some("exhausted".to_string());
                }
            } else {
                retry.last_outcome = Some("deterministic_failure".to_string());
            }
            if final_failure_deterministic {
                retry.deterministic_failure_total =
                    retry.deterministic_failure_total.saturating_add(1);
            }
            retry.last_at = Some(now);
        }
        drop(state);
        slo_failed
    }

    fn record_verified_submit(&self, now: u64, result: u8, event: VerifiedSubmitEvent) {
        let mut state = self.state.write();
        let status = &mut state.status.verified_submit;
        match event {
            VerifiedSubmitEvent::Replay => {
                status.replayed_total = status.replayed_total.saturating_add(1);
            }
            VerifiedSubmitEvent::Conflict => {
                status.request_conflict_total = status.request_conflict_total.saturating_add(1);
            }
            VerifiedSubmitEvent::PendingRejection => {
                status.pending_rejected_total = status.pending_rejected_total.saturating_add(1);
            }
            VerifiedSubmitEvent::CapacityRejection => {
                status.capacity_rejected_total = status.capacity_rejected_total.saturating_add(1);
            }
            VerifiedSubmitEvent::Closed => {}
        }
        status.total = status.total.saturating_add(1);
        match result {
            CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1 => {
                status.onion_and_entry_total = status.onion_and_entry_total.saturating_add(1);
            }
            CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1 => {
                status.onion_only_total = status.onion_only_total.saturating_add(1);
            }
            CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1 => {
                status.entry_retry_total = status.entry_retry_total.saturating_add(1);
            }
            CHAT_VERIFIED_SUBMIT_REJECTED_V1 => {
                status.rejected_total = status.rejected_total.saturating_add(1);
            }
            _ => status.unknown_result_total = status.unknown_result_total.saturating_add(1),
        }
        status.last_result = Some(
            chat_verified_submit_result_label(result)
                .unwrap_or("unknown")
                .to_string(),
        );
        status.last_at = Some(now);
        drop(state);
    }

    fn record_verified_submit_recovery_attempted(&self, now: u64) {
        let mut state = self.state.write();
        let recovery = &mut state.status.verified_submit.entry_recovery;
        recovery.attempted_total = recovery.attempted_total.saturating_add(1);
        recovery.last_outcome = Some("attempted".to_string());
        recovery.last_event_at = Some(now);
        drop(state);
    }

    fn record_verified_submit_recovery_outcome(
        &self,
        now: u64,
        outcome: VerifiedSubmitRecoveryOutcome,
    ) {
        let mut state = self.state.write();
        let recovery = &mut state.status.verified_submit.entry_recovery;
        let bucket = match outcome {
            VerifiedSubmitRecoveryOutcome::Completed => {
                recovery.completed_total = recovery.completed_total.saturating_add(1);
                "completed"
            }
            VerifiedSubmitRecoveryOutcome::Failed => {
                recovery.failed_total = recovery.failed_total.saturating_add(1);
                "failed"
            }
            VerifiedSubmitRecoveryOutcome::Deferred => {
                recovery.deferred_total = recovery.deferred_total.saturating_add(1);
                "deferred"
            }
        };
        recovery.last_outcome = Some(bucket.to_string());
        recovery.last_event_at = Some(now);
        drop(state);
    }

    fn record_inbound_accepted(
        &self,
        now: u64,
        duplicate: bool,
        delivered_online: usize,
        stored_pending: bool,
    ) {
        let mut state = self.state.write();
        let status = &mut state.status;
        status.inbound_accepted_total = status.inbound_accepted_total.saturating_add(1);
        if duplicate {
            status.inbound_duplicate_total = status.inbound_duplicate_total.saturating_add(1);
        }
        status.inbound_delivered_online_total = status
            .inbound_delivered_online_total
            .saturating_add(delivered_online as u64);
        if stored_pending {
            status.inbound_stored_pending_total =
                status.inbound_stored_pending_total.saturating_add(1);
        }
        status.last_inbound_status = Some(if duplicate { "duplicate" } else { "accepted" }.into());
        status.last_inbound_failure_reason = None;
        status.last_inbound_at = Some(now);
        drop(state);
    }

    fn record_inbound_rejected(&self, now: u64, reason: ChatRelayInboundFailureReason) {
        let mut state = self.state.write();
        let status = &mut state.status;
        status.inbound_rejected_total = status.inbound_rejected_total.saturating_add(1);
        status.last_inbound_status = Some("rejected".to_string());
        status.last_inbound_failure_reason = Some(reason.into_bucket());
        status.last_inbound_at = Some(now);
        drop(state);
    }

    fn record_blind_route_recovery(&self, now: u64, event: BlindRouteRecoveryEvent) {
        let mut state = self.state.write();
        let recovery = &mut state.status.blind_route_recovery;
        let bucket = match event {
            BlindRouteRecoveryEvent::Attempted => {
                recovery.attempted_total = recovery.attempted_total.saturating_add(1);
                "attempted"
            }
            BlindRouteRecoveryEvent::Completed => {
                recovery.completed_total = recovery.completed_total.saturating_add(1);
                "completed"
            }
            BlindRouteRecoveryEvent::Deferred => {
                recovery.deferred_total = recovery.deferred_total.saturating_add(1);
                "deferred"
            }
        };
        recovery.last_outcome = Some(bucket.to_string());
        recovery.last_event_at = Some(now);
        drop(state);
    }

    fn snapshot(&self, now: u64, circuit: ChatRelayDirectPeerCircuitStatus) -> ChatRelayPeerStatus {
        let state = self.state.read();
        let mut status = state.status.clone();
        status.direct_peer_retry.recent_window = state.direct_peer_retry_slo.snapshot(now);
        status.direct_peer_retry.circuit = circuit;
        drop(state);
        status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_peer_window_expires_and_requires_repeated_failure() {
        let mut window = DirectPeerRetrySloWindow::default();
        let base = 1_800_000_000;

        window.record(base, false, true, false);
        window.record(base + 1, true, true, false);
        let healthy = window.snapshot(base + 2);
        assert_eq!(healthy.delivery_success_bps, Some(10_000));
        assert_eq!(healthy.retry_recovery_bps, Some(10_000));
        assert_eq!(healthy.status, "healthy");

        window.record(base + 61, false, false, true);
        window.record(base + 62, true, false, false);
        window.record(base + 63, true, false, true);
        let failed = window.snapshot(base + 64);
        assert_eq!(failed.deliveries_total, 5);
        assert_eq!(failed.failed_total, 3);
        assert_eq!(failed.delivery_success_bps, Some(4_000));
        assert_eq!(failed.status, "failed");

        let expired = window.snapshot(base + DIRECT_PEER_RETRY_SLO_WINDOW_SECS + 61);
        assert_eq!(expired.deliveries_total, 0);
        assert_eq!(expired.status, "idle");
    }

    #[test]
    fn unknown_failure_details_cannot_escape_as_health_labels() {
        assert_eq!(
            ChatRelayOutboundFailureReason::from_bucket("receiver=private").into_bucket(),
            "unknown"
        );
        assert_eq!(
            ChatRelayInboundFailureReason::from_bucket("wallet-secret").into_bucket(),
            "unknown"
        );
        assert_eq!(
            ChatRelayOutboundFailureReason::from_bucket("peer_relay_http_503").into_bucket(),
            "peer_relay_http_503"
        );
        assert_eq!(
            ChatRelayOutboundFailureReason::from_bucket("peer_relay_http_999").into_bucket(),
            "unknown"
        );
    }

    #[test]
    fn direct_peer_delivery_updates_window_and_lifetime_under_one_domain() {
        let domain = PeerRelayTelemetryDomain::new(ChatRelayPeerStatus::new(true));
        assert!(!domain.record_direct_peer_delivery(100, true, true, false));
        let status = domain.snapshot(100, ChatRelayDirectPeerCircuitStatus::default());
        assert_eq!(status.direct_peer_retry.retry_triggered_total, 1);
        assert_eq!(status.direct_peer_retry.retry_recovered_total, 1);
        assert_eq!(status.direct_peer_retry.recent_window.deliveries_total, 1);
        assert_eq!(
            status.direct_peer_retry.recent_window.retry_recovered_total,
            1
        );
    }
}
