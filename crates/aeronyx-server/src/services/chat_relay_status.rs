// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_status.rs
// ============================================
// Version: 1.0.0-RelayStatusContractDomain
//
// Creation Reason:
//   [CHAT-RELAY-STATUS-CONTRACT-DOMAIN 2026-08-27 by Codex] Extract the
//   privacy-safe serialized relay status models and their shared policy
//   defaults from the oversized chat relay orchestration service.
//
// Main Functionality:
//   - Defines aggregate-only peer relay, route, retry, and recovery status.
//   - Defines the stable direct-peer SLO and circuit status contracts.
//   - Owns one source of truth for SLO/circuit policy values exposed in status.
//   - Provides compatibility-safe defaults for older serialized snapshots.
//
// Dependencies:
//   - `serde` preserves the existing heartbeat and nodeboard JSON contracts.
//   - Behavior modules consume the policy constants and populate these models.
//
// Main Logical Flow:
//   1. Runtime domains record only aggregate, privacy-safe transitions.
//   2. Runtime domains compose one typed status snapshot.
//   3. Serde emits the unchanged public field names and defaults.
//   4. `chat_relay.rs` re-exports every public type on its legacy path.
//
// Important Note for Next Developer:
//   - Never add identities, routes, endpoints, message IDs, or payload data.
//   - Keep serialized field names and defaults backward compatible.
//   - Change SLO/circuit defaults only together with the behavior and tests.
//   - Keep this module free of filesystem, SQLite, network, and clock I/O.
//
// Last Modified:
//   v1.0.0-RelayStatusContractDomain - Initial status contract extraction
// ============================================

use serde::{Deserialize, Serialize};

/// Recent target-bound delivery health uses five fixed one-minute buckets.
pub(super) const DIRECT_PEER_RETRY_SLO_BUCKET_SECS: u64 = 60;
pub(super) const DIRECT_PEER_RETRY_SLO_BUCKET_COUNT: usize = 5;
pub(super) const DIRECT_PEER_RETRY_SLO_WINDOW_SECS: u64 =
    DIRECT_PEER_RETRY_SLO_BUCKET_SECS * DIRECT_PEER_RETRY_SLO_BUCKET_COUNT as u64;
/// 99.00% target-bound delivery success target, represented in basis points.
pub(super) const DIRECT_PEER_RETRY_SLO_TARGET_BPS: u16 = 9_900;
/// Require repeated failures before declaring a short-window outage.
pub(super) const DIRECT_PEER_RETRY_SLO_FAILED_MIN_FAILURES: u64 = 3;
/// At or below 50% delivery success with enough failures is a failed window.
pub(super) const DIRECT_PEER_RETRY_SLO_FAILED_SUCCESS_BPS: u16 = 5_000;

/// Cooldown before one source-blind target-bound relay recovery probe.
pub(super) const DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS: u64 = 30;
/// Maximum time reserved for one half-open delivery before fail-closed reopen.
pub(super) const DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS: u64 = 15;
/// Consecutive half-open delivery successes required to close the circuit.
pub(super) const DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES: u8 = 2;

/// Aggregate-only health for one outbound encrypted relay route class.
///
/// [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] Authenticated onion relay
/// and compatibility direct relay can run sequentially for the same opaque
/// envelope. Keeping independent snapshots prevents the fallback result from
/// overwriting the route class an operator or live proof is actually testing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayOutboundRouteStatus {
    /// Total requests attempted through this route class.
    pub attempted_total: u64,
    /// Total requests accepted through this route class.
    pub accepted_total: u64,
    /// Total requests that failed or were rejected through this route class.
    pub failed_total: u64,
    /// Total route-class rounds observed, including failed preflight rounds.
    pub rounds: u64,
    /// Requests attempted in the latest route-class round.
    pub last_attempted: u64,
    /// Requests accepted in the latest route-class round.
    pub last_accepted: u64,
    /// Requests failed in the latest route-class round.
    pub last_failed: u64,
    /// Latest health bucket: healthy, degraded, failed, idle.
    pub last_status: Option<String>,
    /// Privacy-safe reason bucket for the latest failure.
    pub last_failure_reason: Option<String>,
    /// Consecutive route-class rounds with no accepted request.
    pub consecutive_failures: u64,
    /// Timestamp of the latest route-class round with an accepted request.
    pub last_success_at: Option<u64>,
    /// Timestamp of the latest route-class round.
    pub last_at: Option<u64>,
}

/// Aggregate-only status for explicit client verified-onion submissions.
///
/// [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] The result vocabulary
/// mirrors the protocol constants but intentionally omits request ids, message
/// ids, routes, terminal receipts, peer identities, wallet keys, and payload
/// commitments. It is safe for heartbeat and nodeboard.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayVerifiedSubmitStatus {
    /// Total verified-submit requests observed by this relay runtime.
    pub total: u64,
    /// Requests with both terminal onion proof and entry custody.
    pub onion_and_entry_total: u64,
    /// Requests with terminal onion proof but failed entry custody.
    pub onion_only_total: u64,
    /// Requests stored by entry custody while onion delivery needs retry.
    pub entry_retry_total: u64,
    /// Requests rejected or lacking any acceptable custody evidence.
    pub rejected_total: u64,
    /// Defensive counter for impossible result codes from a future mismatch.
    pub unknown_result_total: u64,
    /// Exact retries served from the bounded process-local response cache.
    pub replayed_total: u64,
    /// Authenticated sender/request-id reuse with a different envelope.
    pub request_conflict_total: u64,
    /// Exact retries blocked by an unfinished crash-safe reservation.
    pub pending_rejected_total: u64,
    /// New requests rejected before side effects because replay slots were full.
    pub capacity_rejected_total: u64,
    /// Aggregate process-local evidence for restart entry-custody recovery.
    pub entry_recovery: ChatRelayVerifiedSubmitRecoveryStatus,
    /// Last closed result bucket observed.
    pub last_result: Option<String>,
    /// Timestamp of the last observed verified-submit result.
    pub last_at: Option<u64>,
}

/// Aggregate evidence for owner-fenced verified-submit entry recovery.
///
/// [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] This process-lifetime
/// status deliberately carries no request, message, wallet, route, peer,
/// receipt, endpoint, ciphertext, or payload-derived dimension.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayVerifiedSubmitRecoveryStatus {
    /// Foreign-process reservations admitted after the fixed owner grace.
    pub attempted_total: u64,
    /// Attempts that restored custody and durably retained the exact response.
    pub completed_total: u64,
    /// Attempts that durably closed without restoring entry custody.
    pub failed_total: u64,
    /// Attempts left recoverable because response persistence did not complete.
    pub deferred_total: u64,
    /// Last coarse transition: `attempted`, `completed`, `failed`, or `deferred`.
    pub last_outcome: Option<String>,
    /// Timestamp of the latest process-local recovery transition.
    pub last_event_at: Option<u64>,
}

/// Fixed-window target-bound direct relay delivery SLO.
///
/// All ratios use basis points (`10_000 == 100%`) so heartbeat consumers get a
/// stable integer contract. The window contains aggregate counts only and is
/// reset when the process restarts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayDirectPeerSloStatus {
    /// Width of the rolling aggregate window.
    pub window_seconds: u64,
    /// Delivery success target in basis points.
    pub slo_target_bps: u16,
    /// Minimum failures required before the window can be classified failed.
    pub failed_min_failures: u64,
    /// Delivery success ceiling for a failed classification, in basis points.
    pub failed_success_ceiling_bps: u16,
    /// Target-bound v3 deliveries observed in the current window.
    pub deliveries_total: u64,
    /// Deliveries that ended with a valid acknowledgement.
    pub delivered_total: u64,
    /// Deliveries that ended without a valid acknowledgement.
    pub failed_total: u64,
    /// Deliveries that entered the exact retry path.
    pub retry_triggered_total: u64,
    /// Exact retries that recovered a valid acknowledgement.
    pub retry_recovered_total: u64,
    /// Exact retries that exhausted their bounded retry budget.
    pub retry_exhausted_total: u64,
    /// Deliveries whose final typed failure was deterministic.
    pub deterministic_failure_total: u64,
    /// Delivery success ratio in basis points, or none when idle.
    pub delivery_success_bps: Option<u16>,
    /// Retry recovery ratio in basis points, or none when no retry occurred.
    pub retry_recovery_bps: Option<u16>,
    /// Whether the current non-empty window meets the configured SLO.
    pub meets_slo: Option<bool>,
    /// Current aggregate bucket: idle, healthy, degraded, or failed.
    pub status: String,
    /// Unix timestamp used to evaluate the rolling window.
    pub evaluated_at: u64,
}

impl Default for ChatRelayDirectPeerSloStatus {
    fn default() -> Self {
        Self {
            window_seconds: DIRECT_PEER_RETRY_SLO_WINDOW_SECS,
            slo_target_bps: DIRECT_PEER_RETRY_SLO_TARGET_BPS,
            failed_min_failures: DIRECT_PEER_RETRY_SLO_FAILED_MIN_FAILURES,
            failed_success_ceiling_bps: DIRECT_PEER_RETRY_SLO_FAILED_SUCCESS_BPS,
            deliveries_total: 0,
            delivered_total: 0,
            failed_total: 0,
            retry_triggered_total: 0,
            retry_recovered_total: 0,
            retry_exhausted_total: 0,
            deterministic_failure_total: 0,
            delivery_success_bps: None,
            retry_recovery_bps: None,
            meets_slo: None,
            status: "idle".to_string(),
            evaluated_at: 0,
        }
    }
}

/// Aggregate-only target-bound direct relay retry outcomes.
///
/// [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] These counters describe
/// transport reliability without recording which peer, envelope, request
/// commitment, endpoint, or wallet caused an event. Every recorded retry
/// trigger is classified as recovered or exhausted.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayDirectPeerRetryStatus {
    /// Deliveries that entered the one-shot exact retry path.
    pub retry_triggered_total: u64,
    /// Retried deliveries whose exact retry produced a valid ACK.
    pub retry_recovered_total: u64,
    /// Retried deliveries that still failed after the retry budget was spent.
    pub retry_exhausted_total: u64,
    /// Deliveries whose final failure was deterministic and not retryable.
    ///
    /// This may overlap `retry_exhausted_total` when an ambiguity retry receives
    /// a deterministic rejection on its second attempt.
    pub deterministic_failure_total: u64,
    /// Latest observed retry outcome: `recovered`, `exhausted`, or
    /// `deterministic_failure`.
    pub last_outcome: Option<String>,
    /// Unix timestamp of the latest retry or deterministic-failure observation.
    pub last_at: Option<u64>,
    /// Current fixed-memory target-bound delivery SLO window.
    #[serde(default)]
    pub recent_window: ChatRelayDirectPeerSloStatus,
    /// Source-blind target-bound delivery admission circuit.
    #[serde(default)]
    pub circuit: ChatRelayDirectPeerCircuitStatus,
}

/// Aggregate target-bound direct relay circuit status.
///
/// This contract intentionally exposes no circuit slot, peer identity, route,
/// endpoint, request commitment, message identifier, or payload dimension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayDirectPeerCircuitStatus {
    /// Current state: `closed`, `open`, or `half_open`.
    pub state: String,
    /// Fixed open-state cooldown in seconds.
    pub cooldown_seconds: u64,
    /// Maximum lease for one in-flight half-open probe.
    pub half_open_lease_seconds: u64,
    /// Consecutive successful probes required to close.
    pub half_open_successes_required: u8,
    /// Successful probes accumulated in the current recovery sequence.
    pub half_open_consecutive_successes: u8,
    /// Number of transitions into open state.
    pub opened_total: u64,
    /// Delivery rounds denied while open or while a probe is in flight.
    pub blocked_total: u64,
    /// Half-open probes admitted by this process.
    pub half_open_attempted_total: u64,
    /// Half-open probes that ended with valid acknowledgement.
    pub half_open_succeeded_total: u64,
    /// Half-open probes that failed or exceeded their lease.
    pub half_open_failed_total: u64,
    /// Half-open recovery sequences that closed the circuit.
    pub recovered_total: u64,
    /// Remaining open cooldown, or none outside a cooling open state.
    pub open_remaining_seconds: Option<u64>,
    /// Unix timestamp of the latest state transition.
    pub last_transition_at: Option<u64>,
    /// Whether the current safety state is protected across process restart.
    pub restart_protected: bool,
    /// Unix timestamp when this process loaded the durable checkpoint.
    pub checkpoint_loaded_at: Option<u64>,
    /// Unix timestamp of the latest successful checkpoint write.
    pub checkpoint_persisted_at: Option<u64>,
    /// Runtime checkpoint writes that failed closed in this process.
    pub checkpoint_failures_total: u64,
    /// Unix timestamp of the latest runtime checkpoint write failure.
    pub last_checkpoint_failure_at: Option<u64>,
}

impl Default for ChatRelayDirectPeerCircuitStatus {
    fn default() -> Self {
        Self {
            state: "closed".to_string(),
            cooldown_seconds: DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS,
            half_open_lease_seconds: DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS,
            half_open_successes_required: DIRECT_PEER_RELAY_HALF_OPEN_SUCCESSES,
            half_open_consecutive_successes: 0,
            opened_total: 0,
            blocked_total: 0,
            half_open_attempted_total: 0,
            half_open_succeeded_total: 0,
            half_open_failed_total: 0,
            recovered_total: 0,
            open_remaining_seconds: None,
            last_transition_at: None,
            restart_protected: false,
            checkpoint_loaded_at: None,
            checkpoint_persisted_at: None,
            checkpoint_failures_total: 0,
            last_checkpoint_failure_at: None,
        }
    }
}

/// Aggregate commit-durability evidence for encrypted relay custody.
///
/// [CHAT-RELAY-DURABILITY-STATUS 2026-08-16 by Codex] This is configuration
/// evidence, not traffic telemetry. It intentionally carries no database path,
/// message, wallet, peer, endpoint, payload, or row-count dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayCustodyDurabilityStatus {
    /// Stable state bucket: `unknown` or `full`.
    pub state: String,
    /// Whether activation read back `SQLite FULL`-or-stronger commit durability.
    pub full_durability_verified: bool,
    /// Effective `SQLite` synchronous level, when verified by the service.
    pub synchronous_level: Option<u8>,
}

impl Default for ChatRelayCustodyDurabilityStatus {
    fn default() -> Self {
        Self {
            state: "unknown".to_string(),
            full_durability_verified: false,
            synchronous_level: None,
        }
    }
}

impl ChatRelayCustodyDurabilityStatus {
    pub(super) fn verified_full(synchronous_level: u8) -> Self {
        Self {
            state: "full".to_string(),
            full_durability_verified: true,
            synchronous_level: Some(synchronous_level),
        }
    }
}

/// Aggregate evidence for restart reconciliation of armed blind routes.
///
/// [BLIND-ROUTE-RECOVERY-STATUS 2026-08-25 by Codex] These process-lifetime
/// counters prove that durable claims are being reconciled without exposing a
/// current pending count or any route, peer, endpoint, receipt, or ciphertext
/// dimension that could become a traffic-correlation surface.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayBlindRouteRecoveryStatus {
    /// Exact armed claims taken over after the prior process lease expired.
    pub attempted_total: u64,
    /// Recovery attempts that durably sealed and replayed the bounded ACK.
    pub completed_total: u64,
    /// Recovery attempts left armed for another exact retry.
    pub deferred_total: u64,
    /// Last coarse outcome: `attempted`, `completed`, or `deferred`.
    pub last_outcome: Option<String>,
    /// Timestamp of the last process-local recovery transition.
    pub last_event_at: Option<u64>,
}

/// Privacy-safe node-to-node encrypted chat relay health snapshot.
///
/// This structure intentionally contains only aggregate counters and stable
/// reason buckets. It must not include message IDs, wallet IDs, client IPs,
/// destinations, DNS contents, URLs, chat plaintext, ciphertext, private keys,
/// voucher secrets, or per-user traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayPeerStatus {
    /// Whether chat relay is enabled in local config.
    pub enabled: bool,
    /// Verified aggregate commit durability for encrypted custody.
    #[serde(default)]
    pub custody_durability: ChatRelayCustodyDurabilityStatus,
    /// Total outbound peer relay attempts.
    pub outbound_attempted_total: u64,
    /// Total outbound peer relay requests accepted by peer nodes.
    pub outbound_accepted_total: u64,
    /// Total outbound peer relay requests that failed or were rejected.
    pub outbound_failed_total: u64,
    /// Total outbound fanout rounds observed.
    pub outbound_rounds: u64,
    /// Number of peers attempted in the last outbound fanout round.
    pub last_outbound_attempted: u64,
    /// Number of peers that accepted the last outbound fanout round.
    pub last_outbound_accepted: u64,
    /// Number of peers that failed the last outbound fanout round.
    pub last_outbound_failed: u64,
    /// Last outbound health bucket: healthy, degraded, failed, idle.
    pub last_outbound_status: Option<String>,
    /// Privacy-safe reason bucket for the last outbound relay failure.
    pub last_outbound_failure_reason: Option<String>,
    /// Consecutive outbound rounds with zero accepted peer relays.
    pub consecutive_outbound_failures: u64,
    /// Timestamp of the last outbound round with at least one accepted peer.
    pub last_outbound_success_at: Option<u64>,
    /// Timestamp of the last outbound fanout round.
    pub last_outbound_at: Option<u64>,
    /// Authenticated receipt-verified onion relay health, isolated from fallback.
    #[serde(default)]
    pub authenticated_onion_outbound: ChatRelayOutboundRouteStatus,
    /// Compatibility direct peer relay health, isolated from onion delivery.
    #[serde(default)]
    pub direct_peer_outbound: ChatRelayOutboundRouteStatus,
    /// Aggregate target-bound direct relay retry reliability.
    #[serde(default)]
    pub direct_peer_retry: ChatRelayDirectPeerRetryStatus,
    /// Explicit verified-onion client submit outcomes.
    #[serde(default)]
    pub verified_submit: ChatRelayVerifiedSubmitStatus,
    /// Restart reconciliation outcomes for armed blind-route claims.
    #[serde(default)]
    pub blind_route_recovery: ChatRelayBlindRouteRecoveryStatus,
    /// Total inbound peer relay envelopes accepted for local processing.
    pub inbound_accepted_total: u64,
    /// Total inbound duplicate envelopes ignored idempotently.
    pub inbound_duplicate_total: u64,
    /// Total inbound envelopes delivered to online local sessions.
    pub inbound_delivered_online_total: u64,
    /// Total inbound envelopes stored in the local pending queue.
    pub inbound_stored_pending_total: u64,
    /// Total inbound peer relay requests rejected by local validation/storage.
    pub inbound_rejected_total: u64,
    /// Last inbound status bucket: accepted, duplicate, rejected.
    pub last_inbound_status: Option<String>,
    /// Privacy-safe reason bucket for the last inbound rejection.
    pub last_inbound_failure_reason: Option<String>,
    /// Timestamp of the last inbound peer relay request processed.
    pub last_inbound_at: Option<u64>,
}

impl ChatRelayPeerStatus {
    /// Creates an empty aggregate snapshot for one configured runtime state.
    ///
    /// [RELAY-HEALTH-DIAGNOSTICS 2026-08-15 by Codex] This is crate-visible so
    /// host-local health can publish the exact same typed contract when relay
    /// runtime initialization is unavailable, without duplicating counters or
    /// introducing a second telemetry model.
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            custody_durability: ChatRelayCustodyDurabilityStatus::default(),
            outbound_attempted_total: 0,
            outbound_accepted_total: 0,
            outbound_failed_total: 0,
            outbound_rounds: 0,
            last_outbound_attempted: 0,
            last_outbound_accepted: 0,
            last_outbound_failed: 0,
            last_outbound_status: None,
            last_outbound_failure_reason: None,
            consecutive_outbound_failures: 0,
            last_outbound_success_at: None,
            last_outbound_at: None,
            authenticated_onion_outbound: ChatRelayOutboundRouteStatus::default(),
            direct_peer_outbound: ChatRelayOutboundRouteStatus::default(),
            direct_peer_retry: ChatRelayDirectPeerRetryStatus::default(),
            verified_submit: ChatRelayVerifiedSubmitStatus::default(),
            blind_route_recovery: ChatRelayBlindRouteRecoveryStatus::default(),
            inbound_accepted_total: 0,
            inbound_duplicate_total: 0,
            inbound_delivered_online_total: 0,
            inbound_stored_pending_total: 0,
            inbound_rejected_total: 0,
            last_inbound_status: None,
            last_inbound_failure_reason: None,
            last_inbound_at: None,
        }
    }
}
