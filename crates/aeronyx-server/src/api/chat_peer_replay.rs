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
//! v2.8.33-ChatPeerReplayDomain - Initial trait-based extraction and fencing.
//! ============================================

use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
};

use super::chat_peer::PeerBlindRelayResponse;
use crate::services::chat_relay::{
    BLIND_RELAY_ROUTE_REPLAY_CAPACITY, BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
};

/// Maximum live and stale generations retained by the replay eviction queue.
const MAX_REPLAY_QUEUE_GENERATIONS: usize = BLIND_RELAY_ROUTE_REPLAY_CAPACITY * 2;

/// Result of observing one exact blind-route request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum BlindRelayRouteReplayDecision {
    New { generation: u64 },
    InFlight,
    Completed(PeerBlindRelayResponse),
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
    Completed(PeerBlindRelayResponse),
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
            entry.state = BlindRelayRouteReplayState::Completed(response);
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
            BlindRelayRouteReplayDecision::Completed(response)
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
            BlindRelayRouteReplayDecision::Completed(response)
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
