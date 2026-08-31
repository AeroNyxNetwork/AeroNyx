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
//! - Bounds durable ACK bytes and rejects trailing data before allocation-heavy
//!   deserialization.
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
//! - Preserve fixed-int little-endian bincode settings for every legacy row.
//!
//! ## Last Modified
//! v2.8.36-DurableAckBounds - Bound all durable ACK codec paths and reject
//! trailing bytes without changing their versioned or legacy representation.
//! v2.8.35-OnionReplyReplay - Persist optional opaque terminal replies while
//! retaining versioned reads for earlier durable ACK formats.
//! v2.8.34-ChatPeerReplayCodec - Own durable ACK encoding and validation.
//! ============================================

use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
};

use aeronyx_core::protocol::chat::{
    BlindRelayDeliveryReceipt, BlindRelayFailureReceipt, BlindRelaySuccessReceipt,
};
use bincode::Options;
use serde::{Deserialize, Serialize};

use super::{chat_peer::PeerBlindRelayResponse, BLIND_RELAY_ACK_RESPONSE_MAX_BYTES};
use crate::services::chat_relay::{
    BLIND_RELAY_ROUTE_REPLAY_CAPACITY, BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
};

/// Maximum live and stale generations retained by the replay eviction queue.
const MAX_REPLAY_QUEUE_GENERATIONS: usize = BLIND_RELAY_ROUTE_REPLAY_CAPACITY * 2;
const LEGACY_DURABLE_RESPONSE_MAGIC: &[u8; 5] = b"ANBR\x01";
const LEGACY_DURABLE_RESPONSE_V2_MAGIC: &[u8; 5] = b"ANBR\x02";
const DURABLE_RESPONSE_MAGIC: &[u8; 5] = b"ANBR\x03";
const LEGACY_DURABLE_RESPONSE_VERSION: u8 = 1;
const LEGACY_DURABLE_RESPONSE_V2_VERSION: u8 = 2;
const DURABLE_RESPONSE_VERSION: u8 = 3;
/// Maximum bincode body inherited from the bounded public Blind Relay ACK.
const MAX_DURABLE_RESPONSE_BODY_BYTES: usize = BLIND_RELAY_ACK_RESPONSE_MAX_BYTES;
/// Maximum stored v3 frame, including its unchanged five-byte magic prefix.
const MAX_DURABLE_RESPONSE_BYTES: usize =
    DURABLE_RESPONSE_MAGIC.len() + MAX_DURABLE_RESPONSE_BODY_BYTES;

/// Failure class for the private restart-durable response representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayReplayCodecError {
    Encode,
    Decode,
    TooLarge,
    UnsupportedVersion,
    InvalidCompletedState,
}

/// Historical bincode settings plus an allocation-aware hard byte limit.
///
/// [CHAT-PEER-DURABLE-ACK-BOUNDS 2026-08-31 by Codex] The bincode crate's
/// convenience functions use fixed-width integers and little endian. Spell
/// those settings out before adding a limit and trailing-byte rejection so
/// already-persisted v1/v2/v3 and unversioned rows retain identical bytes.
fn durable_response_bincode_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(MAX_DURABLE_RESPONSE_BODY_BYTES as u64)
        .reject_trailing_bytes()
}

fn encode_durable_response_body<T: Serialize>(
    value: &T,
) -> Result<Vec<u8>, BlindRelayReplayCodecError> {
    durable_response_bincode_options()
        .serialize(value)
        .map_err(|error| {
            if matches!(error.as_ref(), bincode::ErrorKind::SizeLimit) {
                BlindRelayReplayCodecError::TooLarge
            } else {
                BlindRelayReplayCodecError::Encode
            }
        })
}

fn decode_durable_response_body<'de, T: Deserialize<'de>>(
    encoded: &'de [u8],
) -> Result<T, BlindRelayReplayCodecError> {
    if encoded.len() > MAX_DURABLE_RESPONSE_BODY_BYTES {
        return Err(BlindRelayReplayCodecError::TooLarge);
    }
    durable_response_bincode_options()
        .deserialize(encoded)
        .map_err(|error| {
            if matches!(error.as_ref(), bincode::ErrorKind::SizeLimit) {
                BlindRelayReplayCodecError::TooLarge
            } else {
                BlindRelayReplayCodecError::Decode
            }
        })
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

/// Legacy storage-only response frame including the optional opaque reply.
#[derive(Debug, Serialize, Deserialize)]
struct DurableBlindRelayResponseV2 {
    version: u8,
    accepted: bool,
    terminal: bool,
    forwarded: bool,
    ttl_remaining: u8,
    reason: Option<String>,
    delivery_receipt: Option<BlindRelayDeliveryReceipt>,
    failure_receipt: Option<BlindRelayFailureReceipt>,
    opaque_terminal_response_b64: Option<String>,
}

/// Current response frame including hop-local success authentication.
///
/// [BLIND-RELAY-SUCCESS-RECEIPT 2026-08-29 by Codex] A separate v3 frame is
/// required because bincode positional decoding cannot treat a newly appended
/// field as optional. V1/v2 readers below remain exact and read-only.
#[derive(Debug, Serialize, Deserialize)]
struct DurableBlindRelayResponseV3 {
    version: u8,
    accepted: bool,
    terminal: bool,
    forwarded: bool,
    ttl_remaining: u8,
    reason: Option<String>,
    delivery_receipt: Option<BlindRelayDeliveryReceipt>,
    success_receipt: Option<BlindRelaySuccessReceipt>,
    failure_receipt: Option<BlindRelayFailureReceipt>,
    opaque_terminal_response_b64: Option<String>,
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
type LegacyResponseWithOpaqueReply = (
    bool,
    bool,
    bool,
    u8,
    Option<String>,
    Option<BlindRelayDeliveryReceipt>,
    Option<BlindRelayFailureReceipt>,
    Option<String>,
);

impl From<&PeerBlindRelayResponse> for DurableBlindRelayResponseV3 {
    fn from(response: &PeerBlindRelayResponse) -> Self {
        Self {
            version: DURABLE_RESPONSE_VERSION,
            accepted: response.accepted,
            terminal: response.terminal,
            forwarded: response.forwarded,
            ttl_remaining: response.ttl_remaining,
            reason: response.reason.clone(),
            delivery_receipt: response.delivery_receipt.clone(),
            success_receipt: response.success_receipt.clone(),
            failure_receipt: response.failure_receipt.clone(),
            opaque_terminal_response_b64: response.opaque_terminal_response_b64.clone(),
        }
    }
}

impl From<DurableBlindRelayResponseV3> for PeerBlindRelayResponse {
    fn from(response: DurableBlindRelayResponseV3) -> Self {
        Self {
            accepted: response.accepted,
            terminal: response.terminal,
            forwarded: response.forwarded,
            ttl_remaining: response.ttl_remaining,
            reason: response.reason,
            delivery_receipt: response.delivery_receipt,
            success_receipt: response.success_receipt,
            failure_receipt: response.failure_receipt,
            opaque_terminal_response_b64: response.opaque_terminal_response_b64,
        }
    }
}

impl From<DurableBlindRelayResponseV2> for PeerBlindRelayResponse {
    fn from(response: DurableBlindRelayResponseV2) -> Self {
        Self {
            accepted: response.accepted,
            terminal: response.terminal,
            forwarded: response.forwarded,
            ttl_remaining: response.ttl_remaining,
            reason: response.reason,
            delivery_receipt: response.delivery_receipt,
            success_receipt: None,
            failure_receipt: response.failure_receipt,
            opaque_terminal_response_b64: response.opaque_terminal_response_b64,
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
            success_receipt: None,
            failure_receipt: response.failure_receipt,
            opaque_terminal_response_b64: None,
        }
    }
}

pub(super) fn encode_durable_blind_relay_response(
    response: &PeerBlindRelayResponse,
) -> Result<Vec<u8>, BlindRelayReplayCodecError> {
    let frame = DurableBlindRelayResponseV3::from(response);
    let body = encode_durable_response_body(&frame)?;
    let encoded_len = DURABLE_RESPONSE_MAGIC
        .len()
        .checked_add(body.len())
        .filter(|length| *length <= MAX_DURABLE_RESPONSE_BYTES)
        .ok_or(BlindRelayReplayCodecError::TooLarge)?;
    let mut encoded = Vec::with_capacity(encoded_len);
    encoded.extend_from_slice(DURABLE_RESPONSE_MAGIC);
    encoded.extend_from_slice(&body);
    Ok(encoded)
}

pub(super) fn decode_durable_blind_relay_response(
    encoded: &[u8],
) -> Result<PeerBlindRelayResponse, BlindRelayReplayCodecError> {
    // [CHAT-PEER-DURABLE-ACK-BOUNDS 2026-08-31 by Codex] Reject oversized
    // storage rows before inspecting a magic prefix or any bincode length.
    if encoded.len() > MAX_DURABLE_RESPONSE_BYTES {
        return Err(BlindRelayReplayCodecError::TooLarge);
    }
    if let Some(body) = encoded.strip_prefix(DURABLE_RESPONSE_MAGIC) {
        let frame: DurableBlindRelayResponseV3 = decode_durable_response_body(body)?;
        if frame.version != DURABLE_RESPONSE_VERSION {
            return Err(BlindRelayReplayCodecError::UnsupportedVersion);
        }
        return Ok(frame.into());
    }
    if let Some(body) = encoded.strip_prefix(LEGACY_DURABLE_RESPONSE_V2_MAGIC) {
        let frame: DurableBlindRelayResponseV2 = decode_durable_response_body(body)?;
        if frame.version != LEGACY_DURABLE_RESPONSE_V2_VERSION {
            return Err(BlindRelayReplayCodecError::UnsupportedVersion);
        }
        return Ok(frame.into());
    }
    if let Some(body) = encoded.strip_prefix(LEGACY_DURABLE_RESPONSE_MAGIC) {
        let frame: DurableBlindRelayResponseV1 = decode_durable_response_body(body)?;
        if frame.version != LEGACY_DURABLE_RESPONSE_VERSION {
            return Err(BlindRelayReplayCodecError::UnsupportedVersion);
        }
        return Ok(frame.into());
    }

    // [DURABLE-BLIND-RELAY-RESPONSE-CODEC 2026-08-25 by Codex] Read ACKs
    // sealed before v1 without rewriting them. The old public response omitted
    // absent trailing fields, so only the two successful shapes that could
    // enter this table are accepted: a delivery receipt or no receipts.
    if let Ok((
        accepted,
        terminal,
        forwarded,
        ttl_remaining,
        reason,
        delivery_receipt,
        failure_receipt,
        opaque_terminal_response_b64,
    )) = decode_durable_response_body::<LegacyResponseWithOpaqueReply>(encoded)
    {
        // [CHAT-PEER-DURABLE-ACK-BOUNDS 2026-08-31 by Codex] An unversioned
        // row reaches this longest historical shape only when the appended
        // opaque field is present. Requiring it prevents zero trailing bytes
        // on a shorter legacy row from masquerading as absent optional fields.
        if opaque_terminal_response_b64.is_none() {
            return Err(BlindRelayReplayCodecError::Decode);
        }
        return Ok(PeerBlindRelayResponse {
            accepted,
            terminal,
            forwarded,
            ttl_remaining,
            reason,
            delivery_receipt,
            success_receipt: None,
            failure_receipt,
            opaque_terminal_response_b64,
        });
    }

    if let Ok((accepted, terminal, forwarded, ttl_remaining, reason, delivery_receipt)) =
        decode_durable_response_body::<LegacyResponseWithDelivery>(encoded)
    {
        if delivery_receipt.is_none() {
            return Err(BlindRelayReplayCodecError::Decode);
        }
        return Ok(PeerBlindRelayResponse {
            accepted,
            terminal,
            forwarded,
            ttl_remaining,
            reason,
            delivery_receipt,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        });
    }

    let (accepted, terminal, forwarded, ttl_remaining, reason) =
        decode_durable_response_body::<LegacyResponseWithoutReceipts>(encoded)?;
    Ok(PeerBlindRelayResponse {
        accepted,
        terminal,
        forwarded,
        ttl_remaining,
        reason,
        delivery_receipt: None,
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
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
            // [BLIND-RELAY-REPLAY-TEST-CONTRACT 2026-08-31 by Codex] Keep
            // durable replay fixtures explicit as the signed response grows.
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        }
    }

    fn versioned_fixture<T: Serialize>(magic: &[u8; 5], frame: &T) -> Vec<u8> {
        let mut encoded = magic.to_vec();
        encoded.extend_from_slice(&bincode::serialize(frame).expect("encode durable fixture"));
        encoded
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
        // [CHAT-PEER-DURABLE-ACK-BOUNDS 2026-08-31 by Codex] Keep current and
        // every historical durable shape byte-compatible under bounded decode.
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
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };
        let encoded = encode_durable_blind_relay_response(&response_with_receipt)
            .expect("encode versioned durable response");
        assert!(encoded.starts_with(DURABLE_RESPONSE_MAGIC));
        assert_eq!(
            &encoded[DURABLE_RESPONSE_MAGIC.len()..],
            bincode::serialize(&DurableBlindRelayResponseV3::from(&response_with_receipt))
                .expect("encode historical v3 body")
        );
        assert_eq!(
            decode_durable_blind_relay_response(&encoded)
                .expect("decode versioned durable response"),
            response_with_receipt.clone()
        );

        let v1 = DurableBlindRelayResponseV1 {
            version: LEGACY_DURABLE_RESPONSE_VERSION,
            accepted: response_with_receipt.accepted,
            terminal: response_with_receipt.terminal,
            forwarded: response_with_receipt.forwarded,
            ttl_remaining: response_with_receipt.ttl_remaining,
            reason: response_with_receipt.reason.clone(),
            delivery_receipt: response_with_receipt.delivery_receipt.clone(),
            failure_receipt: None,
        };
        assert_eq!(
            decode_durable_blind_relay_response(&versioned_fixture(
                LEGACY_DURABLE_RESPONSE_MAGIC,
                &v1,
            ))
            .expect("decode v1 durable response"),
            response_with_receipt.clone()
        );

        let response_with_opaque = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: response_with_receipt.delivery_receipt.clone(),
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: Some("QUJDRA==".to_string()),
        };
        let v2 = DurableBlindRelayResponseV2 {
            version: LEGACY_DURABLE_RESPONSE_V2_VERSION,
            accepted: response_with_opaque.accepted,
            terminal: response_with_opaque.terminal,
            forwarded: response_with_opaque.forwarded,
            ttl_remaining: response_with_opaque.ttl_remaining,
            reason: response_with_opaque.reason.clone(),
            delivery_receipt: response_with_opaque.delivery_receipt.clone(),
            failure_receipt: None,
            opaque_terminal_response_b64: response_with_opaque.opaque_terminal_response_b64.clone(),
        };
        assert_eq!(
            decode_durable_blind_relay_response(&versioned_fixture(
                LEGACY_DURABLE_RESPONSE_V2_MAGIC,
                &v2,
            ))
            .expect("decode v2 durable response"),
            response_with_opaque.clone()
        );

        let legacy_with_opaque: LegacyResponseWithOpaqueReply = (
            response_with_opaque.accepted,
            response_with_opaque.terminal,
            response_with_opaque.forwarded,
            response_with_opaque.ttl_remaining,
            response_with_opaque.reason.clone(),
            response_with_opaque.delivery_receipt.clone(),
            None,
            response_with_opaque.opaque_terminal_response_b64.clone(),
        );
        assert_eq!(
            decode_durable_blind_relay_response(
                &bincode::serialize(&legacy_with_opaque).expect("encode opaque legacy response")
            )
            .expect("decode opaque legacy response"),
            response_with_opaque
        );

        let legacy_with_delivery: LegacyResponseWithDelivery = (
            response_with_receipt.accepted,
            response_with_receipt.terminal,
            response_with_receipt.forwarded,
            response_with_receipt.ttl_remaining,
            response_with_receipt.reason.clone(),
            response_with_receipt.delivery_receipt.clone(),
        );
        assert_eq!(
            decode_durable_blind_relay_response(
                &bincode::serialize(&legacy_with_delivery)
                    .expect("encode delivery legacy response")
            )
            .expect("decode delivery legacy response"),
            response_with_receipt
        );

        let legacy_without_receipts: LegacyResponseWithoutReceipts =
            (true, true, false, 2, Some("terminal_next_hop".to_string()));
        assert_eq!(
            decode_durable_blind_relay_response(
                &bincode::serialize(&legacy_without_receipts)
                    .expect("encode receipt-free legacy response")
            )
            .expect("decode receipt-free legacy response")
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
    fn durable_response_codec_enforces_protocol_derived_size_bound() {
        // [CHAT-PEER-DURABLE-ACK-BOUNDS 2026-08-31 by Codex] The maximum core
        // opaque-response class fits below the ACK-derived durable ceiling.
        let mut maximum_opaque_response = forwarded_response();
        maximum_opaque_response.terminal = true;
        maximum_opaque_response.forwarded = false;
        maximum_opaque_response.reason = Some("onion_terminal_delivered".to_string());
        maximum_opaque_response.opaque_terminal_response_b64 =
            Some("A".repeat(aeronyx_core::protocol::MAX_ONION_SEALED_RESPONSE_BASE64_BYTES));
        let encoded = encode_durable_blind_relay_response(&maximum_opaque_response)
            .expect("maximum protocol-bounded opaque ACK must fit");
        assert!(encoded.len() <= MAX_DURABLE_RESPONSE_BYTES);

        let mut oversized_response = forwarded_response();
        oversized_response.opaque_terminal_response_b64 =
            Some("A".repeat(MAX_DURABLE_RESPONSE_BODY_BYTES));
        assert_eq!(
            encode_durable_blind_relay_response(&oversized_response),
            Err(BlindRelayReplayCodecError::TooLarge)
        );

        assert_eq!(
            decode_durable_blind_relay_response(&vec![0; MAX_DURABLE_RESPONSE_BODY_BYTES]),
            Err(BlindRelayReplayCodecError::Decode)
        );
        assert_eq!(
            decode_durable_blind_relay_response(&vec![0; MAX_DURABLE_RESPONSE_BODY_BYTES + 1]),
            Err(BlindRelayReplayCodecError::TooLarge)
        );
        assert_eq!(
            decode_durable_blind_relay_response(&vec![0; MAX_DURABLE_RESPONSE_BYTES + 1]),
            Err(BlindRelayReplayCodecError::TooLarge)
        );

        let mut malicious_length = DURABLE_RESPONSE_MAGIC.to_vec();
        malicious_length.extend_from_slice(&[DURABLE_RESPONSE_VERSION, 1, 1, 0, 1, 1]);
        malicious_length.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            decode_durable_blind_relay_response(&malicious_length),
            Err(BlindRelayReplayCodecError::Decode)
        );
    }

    #[test]
    fn durable_response_codec_rejects_trailing_bytes_for_every_shape() {
        let terminal = IdentityKeyPair::generate();
        let delivery_receipt =
            BlindRelayDeliveryReceipt::accepted([0xE1; 16], [0xE2; 32], 1_800_000_200, &terminal);
        let v1 = DurableBlindRelayResponseV1 {
            version: LEGACY_DURABLE_RESPONSE_VERSION,
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(delivery_receipt.clone()),
            failure_receipt: None,
        };
        let v2 = DurableBlindRelayResponseV2 {
            version: LEGACY_DURABLE_RESPONSE_V2_VERSION,
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(delivery_receipt.clone()),
            failure_receipt: None,
            opaque_terminal_response_b64: Some("QUJDRA==".to_string()),
        };
        let legacy_with_opaque: LegacyResponseWithOpaqueReply = (
            true,
            true,
            false,
            1,
            Some("onion_terminal_delivered".to_string()),
            Some(delivery_receipt.clone()),
            None,
            Some("QUJDRA==".to_string()),
        );
        let legacy_with_delivery: LegacyResponseWithDelivery = (
            true,
            true,
            false,
            1,
            Some("onion_terminal_delivered".to_string()),
            Some(delivery_receipt),
        );
        let legacy_without_receipts: LegacyResponseWithoutReceipts =
            (true, true, false, 1, Some("terminal_next_hop".to_string()));
        let fixtures = [
            encode_durable_blind_relay_response(&forwarded_response()).expect("encode v3 fixture"),
            versioned_fixture(LEGACY_DURABLE_RESPONSE_V2_MAGIC, &v2),
            versioned_fixture(LEGACY_DURABLE_RESPONSE_MAGIC, &v1),
            bincode::serialize(&legacy_with_opaque).expect("encode opaque legacy fixture"),
            bincode::serialize(&legacy_with_delivery).expect("encode delivery legacy fixture"),
            bincode::serialize(&legacy_without_receipts)
                .expect("encode receipt-free legacy fixture"),
        ];

        for mut fixture in fixtures {
            fixture.push(0);
            assert_eq!(
                decode_durable_blind_relay_response(&fixture),
                Err(BlindRelayReplayCodecError::Decode)
            );
        }
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
