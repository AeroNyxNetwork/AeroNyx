// ============================================
// File: crates/aeronyx-server/src/api/chat_peer_replay.rs
// ============================================
//! # Blind Relay Replay Registry
//!
//! ## Creation Reason
//! Extracts process-local blind-route replay ownership from the public HTTP
//! orchestration module and closes stale-owner mutation races.
//!
//! ## Main Functionality
//! - Defines the replaceable [`BlindRelayReplayRegistry`] capability.
//! - Models new, in-flight, completed, conflicting, and saturated admission.
//! - Retains exact bounded ACKs from their completion boundary.
//! - Fences release and completion by route, commitment, and owner generation.
//! - Owns the versioned restart-durable ACK codec and legacy read compatibility.
//! - Bounds both live entries and stale queue generations without evicting
//!   unexpired at-most-once evidence.
//!
//! ## Dependencies
//! - Uses the stable `PeerBlindRelayResponse` contract from `chat_peer.rs`.
//! - Reuses capacity and retention constants owned by `ChatRelayService` so
//!   process-local compatibility behavior matches durable replay behavior.
//!
//! ## Main Logical Flow
//! 1. Normalize observation time and evict only expired current generations.
//! 2. Return an existing in-flight owner or exact completed ACK when present.
//! 3. Reserve a new generation when fixed capacity permits.
//! 4. Publish or release only when the caller still owns that generation.
//! 5. Compact stale queue generations without deleting current evidence.
//!
//! ## Important Note for Next Developer
//! - Generation fencing is an at-most-once invariant; never weaken it to a
//!   route-only or route-plus-commitment check.
//! - Do not retain payloads, endpoints, users, receivers, or source addresses.
//! - Keep serialized HTTP contracts and durable replay rows backward compatible.
//!
//! ## Last Modified
//! v2.8.34-ChatPeerReplayCodec - Own durable ACK encoding and validation.
//! ============================================

use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
};

use aeronyx_core::protocol::chat::{BlindRelayDeliveryReceipt, BlindRelayFailureReceipt};
use serde::{Deserialize, Serialize};

use super::chat_peer::PeerBlindRelayResponse;
use crate::services::chat_relay::{
    BLIND_RELAY_ROUTE_REPLAY_CAPACITY, BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
};

/// Maximum live and stale generations retained by the replay eviction queue.
const MAX_REPLAY_QUEUE_GENERATIONS: usize = BLIND_RELAY_ROUTE_REPLAY_CAPACITY * 2;
const DURABLE_RESPONSE_MAGIC: &[u8; 5] = b"ANBR\x01";
const DURABLE_RESPONSE_VERSION: u8 = 1;

/// Failure class for the private restart-durable response representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayReplayCodecError {
    Encode,
    Decode,
    UnsupportedVersion,
    InvalidCompletedState,
}

/// Private, versioned representation of one restart-durable route ACK.
///
/// [DURABLE-BLIND-RELAY-RESPONSE-CODEC 2026-08-25 by Codex] The public JSON
/// response omits absent receipt fields for rolling compatibility. Reusing that
/// Serde shape with bincode truncated trailing fields and made a successfully
/// sealed ACK unreadable after the first delivery. This storage-only frame has
/// no conditional fields and is prefixed independently from the public wire.
#[derive(Debug, Serialize, Deserialize)]
struct DurableBlindRelayResponseV1 {
    version: u8,
    accepted: bool,
    terminal: bool,
    forwarded: bool,
    ttl_remaining: u8,
    reason: Option<String>,
    delivery_receipt: Option<BlindRelayDeliveryReceipt>,
    failure_receipt: Option<BlindRelayFailureReceipt>,
}

type LegacyResponseWithDelivery = (
    bool,
    bool,
    bool,
    u8,
    Option<String>,
    Option<BlindRelayDeliveryReceipt>,
);
type LegacyResponseWithoutReceipts = (bool, bool, bool, u8, Option<String>);

impl From<&PeerBlindRelayResponse> for DurableBlindRelayResponseV1 {
    fn from(response: &PeerBlindRelayResponse) -> Self {
        Self {
            version: DURABLE_RESPONSE_VERSION,
            accepted: response.accepted,
            terminal: response.terminal,
            forwarded: response.forwarded,
            ttl_remaining: response.ttl_remaining,
            reason: response.reason.clone(),
            delivery_receipt: response.delivery_receipt.clone(),
            failure_receipt: response.failure_receipt.clone(),
        }
    }
}

impl From<DurableBlindRelayResponseV1> for PeerBlindRelayResponse {
    fn from(response: DurableBlindRelayResponseV1) -> Self {
        Self {
            accepted: response.accepted,
            terminal: response.terminal,
            forwarded: response.forwarded,
            ttl_remaining: response.ttl_remaining,
            reason: response.reason,
            delivery_receipt: response.delivery_receipt,
            failure_receipt: response.failure_receipt,
        }
    }
}

pub(super) fn encode_durable_blind_relay_response(
    response: &PeerBlindRelayResponse,
) -> Result<Vec<u8>, BlindRelayReplayCodecError> {
    let frame = DurableBlindRelayResponseV1::from(response);
    let body = bincode::serialize(&frame).map_err(|_| BlindRelayReplayCodecError::Encode)?;
    let mut encoded = Vec::with_capacity(DURABLE_RESPONSE_MAGIC.len() + body.len());
    encoded.extend_from_slice(DURABLE_RESPONSE_MAGIC);
    encoded.extend_from_slice(&body);
    Ok(encoded)
}

pub(super) fn decode_durable_blind_relay_response(
    encoded: &[u8],
) -> Result<PeerBlindRelayResponse, BlindRelayReplayCodecError> {
    if let Some(body) = encoded.strip_prefix(DURABLE_RESPONSE_MAGIC) {
        let frame: DurableBlindRelayResponseV1 =
            bincode::deserialize(body).map_err(|_| BlindRelayReplayCodecError::Decode)?;
        if frame.version != DURABLE_RESPONSE_VERSION {
            return Err(BlindRelayReplayCodecError::UnsupportedVersion);
        }
        return Ok(frame.into());
    }

    // [DURABLE-BLIND-RELAY-RESPONSE-CODEC 2026-08-25 by Codex] Read ACKs
    // sealed before v1 without rewriting them. The old public response omitted
    // absent trailing fields, so only the two successful shapes that could
    // enter this table are accepted: a delivery receipt or no receipts.
    if let Ok((accepted, terminal, forwarded, ttl_remaining, reason, delivery_receipt)) =
        bincode::deserialize::<LegacyResponseWithDelivery>(encoded)
    {
        return Ok(PeerBlindRelayResponse {
            accepted,
            terminal,
            forwarded,
            ttl_remaining,
            reason,
            delivery_receipt,
            failure_receipt: None,
        });
    }

    let (accepted, terminal, forwarded, ttl_remaining, reason) =
        bincode::deserialize::<LegacyResponseWithoutReceipts>(encoded)
            .map_err(|_| BlindRelayReplayCodecError::Decode)?;
    Ok(PeerBlindRelayResponse {
        accepted,
        terminal,
        forwarded,
        ttl_remaining,
        reason,
        delivery_receipt: None,
        failure_receipt: None,
    })
}

pub(super) fn validate_completed_blind_relay_response(
    response: &PeerBlindRelayResponse,
) -> Result<(), BlindRelayReplayCodecError> {
    let valid_reason = matches!(
        response.reason.as_deref(),
        Some(
            "terminal_next_hop"
                | "forwarded"
                | "onion_terminal_delivered"
                | "onion_forwarded"
                | "onion_middle_forwarded"
        )
    );
    if !response.accepted
        || response.terminal == response.forwarded
        || response.failure_receipt.is_some()
        || !valid_reason
    {
        return Err(BlindRelayReplayCodecError::InvalidCompletedState);
    }
    Ok(())
}

/// Result of observing one exact blind-route request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum BlindRelayRouteReplayDecision {
    New { generation: u64 },
    InFlight,
    Completed(Box<PeerBlindRelayResponse>),
    Conflict,
    Saturated,
}

/// Result of an owner-fenced replay state mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayReplayMutation {
    Applied,
    Missing,
    OwnershipLost,
}

/// Replaceable process-local blind-route replay capability.
///
/// [BLIND-REPLAY-DOMAIN 2026-08-26 by Codex] Callers never receive mutable
/// cache state. Every cleanup and completion mutation carries its owner fence.
pub(super) trait BlindRelayReplayRegistry: Send + Sync {
    fn observe(
        &self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        now: u64,
    ) -> BlindRelayRouteReplayDecision;

    fn complete(
        &self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        owner_generation: u64,
        now: u64,
        response: PeerBlindRelayResponse,
    ) -> BlindRelayReplayMutation;

    fn release(
        &self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        owner_generation: u64,
    ) -> BlindRelayReplayMutation;
}

/// Default synchronized in-memory replay registry.
#[derive(Debug, Default)]
pub(super) struct BlindRelayReplayDomain {
    state: Mutex<BlindRelayReplayState>,
}

impl BlindRelayReplayRegistry for BlindRelayReplayDomain {
    fn observe(
        &self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        now: u64,
    ) -> BlindRelayRouteReplayDecision {
        self.with_state(|state| state.observe(route_id, request_commitment, now))
    }

    fn complete(
        &self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        owner_generation: u64,
        now: u64,
        response: PeerBlindRelayResponse,
    ) -> BlindRelayReplayMutation {
        self.with_state(|state| {
            state.complete(
                route_id,
                request_commitment,
                owner_generation,
                now,
                response,
            )
        })
    }

    fn release(
        &self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        owner_generation: u64,
    ) -> BlindRelayReplayMutation {
        self.with_state(|state| state.release(route_id, request_commitment, owner_generation))
    }
}

impl BlindRelayReplayDomain {
    fn with_state<T>(&self, operation: impl FnOnce(&mut BlindRelayReplayState) -> T) -> T {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        operation(&mut state)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BlindRelayRouteReplayState {
    InFlight,
    Completed(Box<PeerBlindRelayResponse>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlindRelayRouteReplayEntry {
    request_commitment: [u8; 32],
    observed_at: u64,
    generation: u64,
    state: BlindRelayRouteReplayState,
}

#[derive(Debug, Default)]
struct BlindRelayReplayState {
    seen: HashMap<[u8; 16], BlindRelayRouteReplayEntry>,
    order: VecDeque<([u8; 16], u64)>,
    monotonic_observed_at: u64,
    generation_counter: u64,
}

impl BlindRelayReplayState {
    fn observe(
        &mut self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        now: u64,
    ) -> BlindRelayRouteReplayDecision {
        let now = self.normalize_observation_time(now);
        self.evict_expired(now);
        if let Some(entry) = self.seen.get(&route_id) {
            if entry.request_commitment != request_commitment {
                return BlindRelayRouteReplayDecision::Conflict;
            }
            return match &entry.state {
                BlindRelayRouteReplayState::InFlight => BlindRelayRouteReplayDecision::InFlight,
                BlindRelayRouteReplayState::Completed(response) => {
                    BlindRelayRouteReplayDecision::Completed(response.clone())
                }
            };
        }

        // [BLIND-RELAY-NO-EVICTION-ADMISSION 2026-08-24 by Codex] A completed
        // ACK is still the only proof that an ACK-loss retry must not repeat
        // forwarding or terminal delivery. Capacity rejects newer work rather
        // than deleting valid safety evidence.
        if self.seen.len() >= BLIND_RELAY_ROUTE_REPLAY_CAPACITY {
            self.compact_stale_generations_if_needed();
            return BlindRelayRouteReplayDecision::Saturated;
        }

        let generation = self.allocate_generation();
        self.seen.insert(
            route_id,
            BlindRelayRouteReplayEntry {
                request_commitment,
                observed_at: now,
                generation,
                state: BlindRelayRouteReplayState::InFlight,
            },
        );
        self.order.push_back((route_id, generation));
        self.compact_stale_generations_if_needed();
        BlindRelayRouteReplayDecision::New { generation }
    }

    fn complete(
        &mut self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        owner_generation: u64,
        now: u64,
        response: PeerBlindRelayResponse,
    ) -> BlindRelayReplayMutation {
        let now = self.normalize_observation_time(now);
        let Some(owner) = self.seen.get(&route_id) else {
            return BlindRelayReplayMutation::Missing;
        };
        if owner.request_commitment != request_commitment
            || owner.generation != owner_generation
            || !matches!(owner.state, BlindRelayRouteReplayState::InFlight)
        {
            return BlindRelayReplayMutation::OwnershipLost;
        }

        // [BLIND-REPLAY-GENERATION-FENCE 2026-08-26 by Codex] Completion gets
        // a fresh generation and completion-time TTL. A stale owner can no
        // longer overwrite a newer claim or an already-published exact ACK.
        let observed_at = now.max(owner.observed_at);
        let completed_generation = self.allocate_generation();
        if let Some(entry) = self.seen.get_mut(&route_id) {
            entry.observed_at = observed_at;
            entry.generation = completed_generation;
            entry.state = BlindRelayRouteReplayState::Completed(Box::new(response));
        }
        self.order.push_back((route_id, completed_generation));
        self.evict_expired(observed_at);
        self.compact_stale_generations_if_needed();
        BlindRelayReplayMutation::Applied
    }

    fn release(
        &mut self,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        owner_generation: u64,
    ) -> BlindRelayReplayMutation {
        let Some(owner) = self.seen.get(&route_id) else {
            return BlindRelayReplayMutation::Missing;
        };
        if owner.request_commitment != request_commitment
            || owner.generation != owner_generation
            || !matches!(owner.state, BlindRelayRouteReplayState::InFlight)
        {
            return BlindRelayReplayMutation::OwnershipLost;
        }
        self.seen.remove(&route_id);
        self.compact_stale_generations_if_needed();
        BlindRelayReplayMutation::Applied
    }

    fn normalize_observation_time(&mut self, now: u64) -> u64 {
        self.monotonic_observed_at = self.monotonic_observed_at.max(now);
        self.monotonic_observed_at
    }

    fn allocate_generation(&mut self) -> u64 {
        self.generation_counter = self.generation_counter.wrapping_add(1);
        if self.generation_counter == 0 {
            self.generation_counter = 1;
        }
        self.generation_counter
    }

    fn evict_expired(&mut self, now: u64) {
        while let Some((route_id, queued_generation)) = self.order.front().copied() {
            let Some(current) = self.seen.get(&route_id) else {
                self.order.pop_front();
                continue;
            };
            if current.generation != queued_generation {
                self.order.pop_front();
                continue;
            }
            if now.saturating_sub(current.observed_at) <= BLIND_RELAY_ROUTE_REPLAY_TTL_SECS {
                break;
            }
            self.order.pop_front();
            self.seen.remove(&route_id);
        }
    }

    fn compact_stale_generations_if_needed(&mut self) {
        if self.order.len() <= MAX_REPLAY_QUEUE_GENERATIONS {
            return;
        }
        let seen = &self.seen;
        self.order.retain(|(route_id, generation)| {
            seen.get(route_id)
                .is_some_and(|entry| entry.generation == *generation)
        });
        debug_assert!(self.order.len() <= self.seen.len());
    }
}

#[cfg(test)]
pub(super) const REPLAY_CAPACITY_FOR_TESTS: usize = BLIND_RELAY_ROUTE_REPLAY_CAPACITY;

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;

    fn forwarded_response() -> PeerBlindRelayResponse {
        PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("forwarded".to_string()),
            delivery_receipt: None,
            failure_receipt: None,
        }
    }

    fn begin_generation(
        state: &mut BlindRelayReplayState,
        route_id: [u8; 16],
        commitment: [u8; 32],
        now: u64,
    ) -> u64 {
        match state.observe(route_id, commitment, now) {
            BlindRelayRouteReplayDecision::New { generation } => generation,
            decision => panic!("unexpected replay decision: {decision:?}"),
        }
    }

    #[test]
    fn durable_response_codec_round_trips_and_reads_legacy_rows() {
        // [DURABLE-BLIND-RELAY-RESPONSE-CODEC 2026-08-25 by Codex] Protect
        // both the storage-only frame and already-sealed public-Serde rows.
        let terminal = IdentityKeyPair::generate();
        let response_with_receipt = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted(
                [0xD8; 16],
                [0xD9; 32],
                1_800_000_100,
                &terminal,
            )),
            failure_receipt: None,
        };
        let encoded = encode_durable_blind_relay_response(&response_with_receipt)
            .expect("encode versioned durable response");
        assert!(encoded.starts_with(DURABLE_RESPONSE_MAGIC));
        assert_eq!(
            decode_durable_blind_relay_response(&encoded)
                .expect("decode versioned durable response"),
            response_with_receipt
        );

        let legacy_with_receipt = bincode::serialize(&response_with_receipt)
            .expect("encode legacy response with receipt");
        assert_eq!(
            decode_durable_blind_relay_response(&legacy_with_receipt)
                .expect("decode legacy response with receipt"),
            response_with_receipt
        );

        let legacy_without_receipts = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 2,
            reason: Some("terminal_next_hop".to_string()),
            delivery_receipt: None,
            failure_receipt: None,
        };
        let legacy_without_receipts = bincode::serialize(&legacy_without_receipts)
            .expect("encode legacy response without receipts");
        assert_eq!(
            decode_durable_blind_relay_response(&legacy_without_receipts)
                .expect("decode legacy response without receipts")
                .reason
                .as_deref(),
            Some("terminal_next_hop")
        );
    }

    #[test]
    fn durable_response_codec_rejects_unsupported_version() {
        let mut encoded = encode_durable_blind_relay_response(&forwarded_response())
            .expect("encode durable response");
        encoded[DURABLE_RESPONSE_MAGIC.len()] = DURABLE_RESPONSE_VERSION + 1;
        assert_eq!(
            decode_durable_blind_relay_response(&encoded),
            Err(BlindRelayReplayCodecError::UnsupportedVersion)
        );
    }

    #[test]
    fn completed_response_validation_rejects_ambiguous_success() {
        let mut response = forwarded_response();
        response.forwarded = false;
        assert_eq!(
            validate_completed_blind_relay_response(&response),
            Err(BlindRelayReplayCodecError::InvalidCompletedState)
        );
    }

    #[test]
    fn returns_exact_completed_ack_from_completion_boundary() {
        let route_id = [0x41; 16];
        let commitment = [0xA1; 32];
        let started_at = 1_800_000_000;
        let completed_at = started_at + BLIND_RELAY_ROUTE_REPLAY_TTL_SECS;
        let response = forwarded_response();
        let mut state = BlindRelayReplayState::default();
        let generation = begin_generation(&mut state, route_id, commitment, started_at);

        assert_eq!(
            state.complete(
                route_id,
                commitment,
                generation,
                completed_at,
                response.clone(),
            ),
            BlindRelayReplayMutation::Applied
        );
        assert_eq!(
            state.observe(route_id, commitment, completed_at + 1),
            BlindRelayRouteReplayDecision::Completed(Box::new(response))
        );
        assert!(matches!(
            state.observe(
                route_id,
                commitment,
                completed_at + BLIND_RELAY_ROUTE_REPLAY_TTL_SECS + 1,
            ),
            BlindRelayRouteReplayDecision::New { .. }
        ));
    }

    #[test]
    fn release_requires_exact_owner_generation() {
        let route_id = [0x42; 16];
        let commitment = [0xA2; 32];
        let now = 1_800_000_000;
        let mut state = BlindRelayReplayState::default();
        let generation = begin_generation(&mut state, route_id, commitment, now);

        assert_eq!(
            state.release(route_id, commitment, generation + 1),
            BlindRelayReplayMutation::OwnershipLost
        );
        assert_eq!(
            state.observe(route_id, commitment, now),
            BlindRelayRouteReplayDecision::InFlight
        );
        assert_eq!(
            state.release(route_id, commitment, generation),
            BlindRelayReplayMutation::Applied
        );
        assert!(matches!(
            state.observe(route_id, commitment, now),
            BlindRelayRouteReplayDecision::New { .. }
        ));
    }

    #[test]
    fn stale_owner_cannot_release_or_complete_new_generation() {
        let route_id = [0x43; 16];
        let commitment = [0xA3; 32];
        let started_at = 1_800_000_000;
        let mut state = BlindRelayReplayState::default();
        let stale_generation = begin_generation(&mut state, route_id, commitment, started_at);
        let takeover_at = started_at + BLIND_RELAY_ROUTE_REPLAY_TTL_SECS + 1;
        let current_generation = begin_generation(&mut state, route_id, commitment, takeover_at);

        assert_ne!(stale_generation, current_generation);
        assert_eq!(
            state.release(route_id, commitment, stale_generation),
            BlindRelayReplayMutation::OwnershipLost
        );
        assert_eq!(
            state.complete(
                route_id,
                commitment,
                stale_generation,
                takeover_at,
                forwarded_response(),
            ),
            BlindRelayReplayMutation::OwnershipLost
        );
        assert_eq!(
            state.observe(route_id, commitment, takeover_at),
            BlindRelayRouteReplayDecision::InFlight
        );
    }

    #[test]
    fn capacity_preserves_completed_and_in_flight_evidence() {
        let completed_route = [0x51; 16];
        let live_route = [0x52; 16];
        let saturated_route = [0x53; 16];
        let commitment = [0xB1; 32];
        let now = 1_800_000_000;
        let response = forwarded_response();
        let mut state = BlindRelayReplayState::default();
        let completed_generation = begin_generation(&mut state, completed_route, commitment, now);
        assert_eq!(
            state.complete(
                completed_route,
                commitment,
                completed_generation,
                now,
                response.clone(),
            ),
            BlindRelayReplayMutation::Applied
        );
        begin_generation(&mut state, live_route, commitment, now);

        for sequence in 0..BLIND_RELAY_ROUTE_REPLAY_CAPACITY.saturating_sub(2) {
            let mut route_id = [0x54; 16];
            route_id[..8].copy_from_slice(&(sequence as u64).to_be_bytes());
            begin_generation(&mut state, route_id, commitment, now);
        }
        assert_eq!(
            state.observe(saturated_route, commitment, now),
            BlindRelayRouteReplayDecision::Saturated
        );
        assert_eq!(
            state.observe(completed_route, commitment, now),
            BlindRelayRouteReplayDecision::Completed(Box::new(response))
        );
        assert_eq!(
            state.observe(live_route, commitment, now),
            BlindRelayRouteReplayDecision::InFlight
        );
    }

    #[test]
    fn same_second_retries_keep_queue_bounded() {
        let live_route = [0x61; 16];
        let retried_route = [0x62; 16];
        let commitment = [0xC1; 32];
        let now = 1_800_000_000;
        let mut state = BlindRelayReplayState::default();
        begin_generation(&mut state, live_route, commitment, now);

        for _ in 0..=MAX_REPLAY_QUEUE_GENERATIONS {
            let generation = begin_generation(&mut state, retried_route, commitment, now);
            assert_eq!(
                state.release(retried_route, commitment, generation),
                BlindRelayReplayMutation::Applied
            );
        }
        assert!(state.order.len() <= MAX_REPLAY_QUEUE_GENERATIONS);
        assert_eq!(
            state.observe(live_route, commitment, now),
            BlindRelayRouteReplayDecision::InFlight
        );
    }
}
