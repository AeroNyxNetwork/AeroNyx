// ============================================
// File: crates/aeronyx-server/src/api/chat_peer_admission.rs
// ============================================
// Version: 1.0.0-DirectPeerAdmissionDomain
//
// Creation Reason:
//   [CHAT-PEER-ADMISSION-DOMAIN 2026-08-26 by Codex] Extract authenticated
//   direct-relay admission, fairness, and exact ACK replay ownership from the
//   oversized public HTTP orchestration module.
//
// Main Functionality:
//   - Enforces one identity-independent monotonic request window.
//   - Applies bounded fairness only after a node identity is authenticated.
//   - Owns exact request commitments through generation-fenced RAII leases.
//   - Replays a completed privacy-normalized ACK without repeating effects.
//   - Starts completed ACK retention at durable completion, not request start.
//
// Dependencies:
//   - `chat_peer.rs` owns the stable HTTP and serialized response contracts.
//   - `PeerRelayRequestGate` composes this domain with parser-front in-flight
//     admission and aggregate rejection telemetry.
//
// Main Logical Flow:
//   1. Admit aggregate parser work through a fixed monotonic window.
//   2. After signature verification, apply a bounded per-node fair window.
//   3. Reserve the exact request commitment or replay its completed ACK.
//   4. Publish completion under a new generation and completion-time TTL.
//   5. Release only the matching in-flight generation on cancellation.
//
// Important Note for Next Developer:
//   - Never key aggregate admission with user, wallet, receiver, source IP,
//     endpoint, message id, or payload-derived data.
//   - A replay entry contains only a SHA-256 request commitment and ACK.
//   - Do not evict unexpired in-flight owners to admit newer work.
//   - Preserve generation fencing; stale leases must not mutate newer owners.
//   - Keep HTTP routes and JSON fields in `chat_peer.rs` backward compatible.
//
// Last Modified:
//   v1.0.0-DirectPeerAdmissionDomain - Initial trait-based composition
// ============================================

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use super::chat_peer::PeerChatRelayResponseV2;

/// Maximum authenticated direct-relay node buckets retained in memory.
const MAX_AUTHENTICATED_PEER_RELAY_BUCKETS: usize = 4096;
/// Maximum exact authenticated requests retained for safe ACK replay.
const MAX_AUTHENTICATED_PEER_RELAY_REPLAYS: usize = 4096;
/// Maximum live and stale generations retained by the ACK replay queue.
const MAX_AUTHENTICATED_PEER_RELAY_REPLAY_GENERATIONS: usize =
    MAX_AUTHENTICATED_PEER_RELAY_REPLAYS * 2;
/// Completed ACKs expire before their signed receipt freshness horizon.
const AUTHENTICATED_PEER_RELAY_REPLAY_TTL: Duration = Duration::from_secs(90);
/// Aggregate direct-relay admission uses exact one-minute monotonic windows.
const PEER_RELAY_RATE_WINDOW: Duration = Duration::from_secs(60);

/// Replaceable direct-peer admission capability consumed by HTTP orchestration.
pub(super) trait DirectPeerAdmissionPolicy: Send + Sync {
    /// Admits one identity-independent parser-front request.
    fn admit_global(&self, now: Instant) -> bool;
    /// Admits one already-authenticated previous-hop node.
    fn admit_authenticated(&self, node_id: [u8; 32], now: Instant) -> bool;
    /// Begins or replays one exact authenticated request commitment.
    fn begin_replay(
        &self,
        request_commitment: [u8; 32],
        now: Instant,
    ) -> AuthenticatedPeerRelayReplayStart;
}

/// One exact monotonic fixed-window counter.
#[derive(Debug)]
pub(super) struct PeerRelayRateLimitWindow {
    started_at: Instant,
    admitted: u32,
}

impl PeerRelayRateLimitWindow {
    pub(super) const fn new(started_at: Instant) -> Self {
        Self {
            started_at,
            admitted: 0,
        }
    }

    pub(super) fn allow(&mut self, now: Instant, limit: u32) -> bool {
        self.allow_for(now, limit, PEER_RELAY_RATE_WINDOW)
    }

    pub(super) fn allow_for(&mut self, now: Instant, limit: u32, window: Duration) -> bool {
        if now.saturating_duration_since(self.started_at) >= window {
            self.started_at = now;
            self.admitted = 0;
        }
        if self.admitted >= limit {
            return false;
        }
        self.admitted = self.admitted.saturating_add(1);
        true
    }

    #[cfg(test)]
    pub(super) const fn admitted(&self) -> u32 {
        self.admitted
    }
}

#[derive(Debug)]
struct AuthenticatedPeerRelayRateBucket {
    window: PeerRelayRateLimitWindow,
    last_seen: Instant,
}

#[derive(Debug, Default)]
struct AuthenticatedPeerRelayRateLimiter {
    buckets: HashMap<[u8; 32], AuthenticatedPeerRelayRateBucket>,
}

impl AuthenticatedPeerRelayRateLimiter {
    fn allow(&mut self, node_id: [u8; 32], now: Instant, limit: u32) -> bool {
        // [CHAT-PEER-ADMISSION-DOMAIN 2026-08-26 by Codex] A new verified
        // identity evicts only the least-recently-observed fixed-size bucket.
        if !self.buckets.contains_key(&node_id)
            && self.buckets.len() >= MAX_AUTHENTICATED_PEER_RELAY_BUCKETS
        {
            if let Some(oldest) = self
                .buckets
                .iter()
                .min_by_key(|(_, bucket)| bucket.last_seen)
                .map(|(node_id, _)| *node_id)
            {
                self.buckets.remove(&oldest);
            }
        }

        let bucket =
            self.buckets
                .entry(node_id)
                .or_insert_with(|| AuthenticatedPeerRelayRateBucket {
                    window: PeerRelayRateLimitWindow::new(now),
                    last_seen: now,
                });
        bucket.last_seen = now;
        bucket.window.allow(now, limit)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AuthenticatedPeerRelayReplayState {
    InFlight,
    Completed(PeerChatRelayResponseV2),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AuthenticatedPeerRelayReplayEntry {
    observed_at: Instant,
    generation: u64,
    state: AuthenticatedPeerRelayReplayState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AuthenticatedPeerRelayReplayDecision {
    New(u64),
    InFlight,
    Completed(PeerChatRelayResponseV2),
    Saturated,
}

pub(super) enum AuthenticatedPeerRelayReplayStart {
    Acquired(AuthenticatedPeerRelayReplayLease),
    InFlight,
    Completed(PeerChatRelayResponseV2),
    Saturated,
}

/// Owns one authenticated request until its exact ACK is published.
pub(super) struct AuthenticatedPeerRelayReplayLease {
    cache: Arc<Mutex<AuthenticatedPeerRelayReplayCache>>,
    request_commitment: [u8; 32],
    generation: u64,
    active: bool,
}

impl AuthenticatedPeerRelayReplayLease {
    const fn new(
        cache: Arc<Mutex<AuthenticatedPeerRelayReplayCache>>,
        request_commitment: [u8; 32],
        generation: u64,
    ) -> Self {
        Self {
            cache,
            request_commitment,
            generation,
            active: true,
        }
    }

    /// Publishes the exact ACK and disarms cancellation cleanup.
    pub(super) fn complete(mut self, response: PeerChatRelayResponseV2) {
        let completed_at = Instant::now();
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .complete(
                &self.request_commitment,
                self.generation,
                completed_at,
                response,
            );
        self.active = false;
    }
}

impl Drop for AuthenticatedPeerRelayReplayLease {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .forget(&self.request_commitment, self.generation);
    }
}

#[derive(Debug, Default)]
struct AuthenticatedPeerRelayReplayCache {
    entries: HashMap<[u8; 32], AuthenticatedPeerRelayReplayEntry>,
    order: VecDeque<([u8; 32], u64)>,
    next_generation: u64,
}

impl AuthenticatedPeerRelayReplayCache {
    fn begin(
        &mut self,
        request_commitment: [u8; 32],
        now: Instant,
    ) -> AuthenticatedPeerRelayReplayDecision {
        self.evict_expired(now);
        if let Some(entry) = self.entries.get(&request_commitment) {
            return match &entry.state {
                AuthenticatedPeerRelayReplayState::InFlight => {
                    AuthenticatedPeerRelayReplayDecision::InFlight
                }
                AuthenticatedPeerRelayReplayState::Completed(response) => {
                    AuthenticatedPeerRelayReplayDecision::Completed(response.clone())
                }
            };
        }

        let generation = self.allocate_generation();
        self.entries.insert(
            request_commitment,
            AuthenticatedPeerRelayReplayEntry {
                observed_at: now,
                generation,
                state: AuthenticatedPeerRelayReplayState::InFlight,
            },
        );
        self.order.push_back((request_commitment, generation));
        let retained = self.evict_over_capacity(request_commitment, generation);
        self.compact_stale_generations();
        if retained {
            AuthenticatedPeerRelayReplayDecision::New(generation)
        } else {
            AuthenticatedPeerRelayReplayDecision::Saturated
        }
    }

    fn complete(
        &mut self,
        request_commitment: &[u8; 32],
        owner_generation: u64,
        completed_at: Instant,
        response: PeerChatRelayResponseV2,
    ) {
        let Some(owner) = self.entries.get(request_commitment) else {
            return;
        };
        if owner.generation != owner_generation
            || !matches!(owner.state, AuthenticatedPeerRelayReplayState::InFlight)
        {
            return;
        }

        // [CHAT-PEER-ACK-COMPLETION-TTL 2026-08-26 by Codex] Retention begins
        // only after durable acceptance completes. A new generation makes the
        // request-start queue entry stale without letting an old owner mutate
        // this completed ACK.
        let observed_at = completed_at.max(owner.observed_at);
        let completed_generation = self.allocate_generation();
        if let Some(entry) = self.entries.get_mut(request_commitment) {
            entry.observed_at = observed_at;
            entry.generation = completed_generation;
            entry.state = AuthenticatedPeerRelayReplayState::Completed(response);
        }
        self.order
            .push_back((*request_commitment, completed_generation));
        self.compact_stale_generations();
    }

    fn forget(&mut self, request_commitment: &[u8; 32], generation: u64) {
        if self
            .entries
            .get(request_commitment)
            .is_some_and(|entry| entry.generation == generation)
        {
            self.entries.remove(request_commitment);
        }
    }

    fn evict_expired(&mut self, now: Instant) {
        while let Some((request_commitment, generation)) = self.order.front().copied() {
            let Some(entry) = self.entries.get(&request_commitment) else {
                self.order.pop_front();
                continue;
            };
            if entry.generation != generation {
                self.order.pop_front();
                continue;
            }
            if now.saturating_duration_since(entry.observed_at)
                <= AUTHENTICATED_PEER_RELAY_REPLAY_TTL
            {
                break;
            }
            self.order.pop_front();
            self.entries.remove(&request_commitment);
        }
    }

    fn evict_over_capacity(
        &mut self,
        new_request_commitment: [u8; 32],
        new_generation: u64,
    ) -> bool {
        while self.entries.len() > MAX_AUTHENTICATED_PEER_RELAY_REPLAYS {
            let completed_position = self.order.iter().position(|(commitment, generation)| {
                self.entries.get(commitment).is_some_and(|entry| {
                    entry.generation == *generation
                        && matches!(entry.state, AuthenticatedPeerRelayReplayState::Completed(_))
                })
            });
            let Some(completed_position) = completed_position else {
                self.forget(&new_request_commitment, new_generation);
                return false;
            };
            let Some((commitment, generation)) = self.order.remove(completed_position) else {
                self.forget(&new_request_commitment, new_generation);
                return false;
            };
            self.forget(&commitment, generation);
        }
        true
    }

    fn compact_stale_generations(&mut self) {
        if self.order.len() <= MAX_AUTHENTICATED_PEER_RELAY_REPLAY_GENERATIONS {
            return;
        }
        self.order.retain(|(commitment, generation)| {
            self.entries
                .get(commitment)
                .is_some_and(|entry| entry.generation == *generation)
        });
    }

    fn allocate_generation(&mut self) -> u64 {
        self.next_generation = self.next_generation.wrapping_add(1);
        if self.next_generation == 0 {
            self.next_generation = 1;
        }
        self.next_generation
    }
}

/// Default in-memory admission domain for direct peer requests.
#[derive(Debug)]
pub(super) struct DirectPeerAdmissionDomain {
    global_rate_limit: Mutex<PeerRelayRateLimitWindow>,
    global_requests_per_minute: u32,
    authenticated_rate_limit: Mutex<AuthenticatedPeerRelayRateLimiter>,
    authenticated_requests_per_minute: u32,
    authenticated_replays: Arc<Mutex<AuthenticatedPeerRelayReplayCache>>,
}

impl DirectPeerAdmissionDomain {
    pub(super) fn new(
        global_requests_per_minute: u32,
        authenticated_requests_per_minute: u32,
    ) -> Self {
        Self {
            global_rate_limit: Mutex::new(PeerRelayRateLimitWindow::new(Instant::now())),
            global_requests_per_minute,
            authenticated_rate_limit: Mutex::new(AuthenticatedPeerRelayRateLimiter::default()),
            authenticated_requests_per_minute,
            authenticated_replays: Arc::new(Mutex::new(
                AuthenticatedPeerRelayReplayCache::default(),
            )),
        }
    }
}

impl DirectPeerAdmissionPolicy for DirectPeerAdmissionDomain {
    fn admit_global(&self, now: Instant) -> bool {
        self.global_rate_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .allow(now, self.global_requests_per_minute)
    }

    fn admit_authenticated(&self, node_id: [u8; 32], now: Instant) -> bool {
        self.authenticated_rate_limit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .allow(node_id, now, self.authenticated_requests_per_minute)
    }

    fn begin_replay(
        &self,
        request_commitment: [u8; 32],
        now: Instant,
    ) -> AuthenticatedPeerRelayReplayStart {
        let mut cache = self
            .authenticated_replays
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match cache.begin(request_commitment, now) {
            AuthenticatedPeerRelayReplayDecision::New(generation) => {
                AuthenticatedPeerRelayReplayStart::Acquired(AuthenticatedPeerRelayReplayLease::new(
                    Arc::clone(&self.authenticated_replays),
                    request_commitment,
                    generation,
                ))
            }
            AuthenticatedPeerRelayReplayDecision::InFlight => {
                AuthenticatedPeerRelayReplayStart::InFlight
            }
            AuthenticatedPeerRelayReplayDecision::Completed(response) => {
                AuthenticatedPeerRelayReplayStart::Completed(response)
            }
            AuthenticatedPeerRelayReplayDecision::Saturated => {
                AuthenticatedPeerRelayReplayStart::Saturated
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::chat_peer::{PeerChatRelayResponse, PeerChatRelayResponseV2};

    fn accepted_response() -> PeerChatRelayResponseV2 {
        PeerChatRelayResponseV2 {
            relay: PeerChatRelayResponse {
                accepted: true,
                duplicate: false,
                delivered_online: 0,
                stored_pending: true,
            },
            receipt: None,
        }
    }

    #[test]
    fn replay_cache_returns_exact_completed_ack() {
        let now = Instant::now();
        let commitment = [0x41; 32];
        let response = accepted_response();
        let mut cache = AuthenticatedPeerRelayReplayCache::default();

        let generation = match cache.begin(commitment, now) {
            AuthenticatedPeerRelayReplayDecision::New(generation) => generation,
            decision => panic!("unexpected first replay decision: {decision:?}"),
        };
        assert_eq!(
            cache.begin(commitment, now),
            AuthenticatedPeerRelayReplayDecision::InFlight
        );
        cache.complete(&commitment, generation, now, response.clone());
        assert_eq!(
            cache.begin(commitment, now),
            AuthenticatedPeerRelayReplayDecision::Completed(response)
        );
    }

    #[test]
    fn completed_ack_ttl_starts_at_completion_boundary() {
        let started_at = Instant::now();
        let completed_at = started_at + Duration::from_secs(45);
        let commitment = [0x44; 32];
        let mut cache = AuthenticatedPeerRelayReplayCache::default();
        let generation = match cache.begin(commitment, started_at) {
            AuthenticatedPeerRelayReplayDecision::New(generation) => generation,
            decision => panic!("unexpected replay decision: {decision:?}"),
        };

        cache.complete(&commitment, generation, completed_at, accepted_response());
        assert!(matches!(
            cache.begin(
                commitment,
                completed_at + AUTHENTICATED_PEER_RELAY_REPLAY_TTL
            ),
            AuthenticatedPeerRelayReplayDecision::Completed(_)
        ));
        assert!(matches!(
            cache.begin(
                commitment,
                completed_at + AUTHENTICATED_PEER_RELAY_REPLAY_TTL + Duration::from_millis(1)
            ),
            AuthenticatedPeerRelayReplayDecision::New(_)
        ));
    }

    #[test]
    fn stale_owner_cannot_mutate_new_generation() {
        let now = Instant::now();
        let commitment = [0x42; 32];
        let mut cache = AuthenticatedPeerRelayReplayCache::default();
        let first_generation = match cache.begin(commitment, now) {
            AuthenticatedPeerRelayReplayDecision::New(generation) => generation,
            decision => panic!("unexpected first replay decision: {decision:?}"),
        };
        cache.forget(&commitment, first_generation);
        let second_generation = match cache.begin(commitment, now) {
            AuthenticatedPeerRelayReplayDecision::New(generation) => generation,
            decision => panic!("unexpected second replay decision: {decision:?}"),
        };
        assert_ne!(first_generation, second_generation);

        cache.complete(&commitment, first_generation, now, accepted_response());
        cache.forget(&commitment, first_generation);
        assert_eq!(
            cache.begin(commitment, now),
            AuthenticatedPeerRelayReplayDecision::InFlight
        );
    }

    #[test]
    fn replay_capacity_preserves_in_flight_owner() {
        let now = Instant::now();
        let response = accepted_response();
        let mut cache = AuthenticatedPeerRelayReplayCache::default();
        for index in 0..MAX_AUTHENTICATED_PEER_RELAY_REPLAYS {
            let mut commitment = [0u8; 32];
            commitment[..8].copy_from_slice(&(index as u64).to_be_bytes());
            let generation = (index as u64).saturating_add(1);
            let state = if index == 0 {
                AuthenticatedPeerRelayReplayState::InFlight
            } else {
                AuthenticatedPeerRelayReplayState::Completed(response.clone())
            };
            cache.entries.insert(
                commitment,
                AuthenticatedPeerRelayReplayEntry {
                    observed_at: now,
                    generation,
                    state,
                },
            );
            cache.order.push_back((commitment, generation));
        }

        let oldest_in_flight = cache.order.front().copied().unwrap();
        let evicted_completed = cache.order.get(1).copied().unwrap();
        let new_commitment = [0xFF; 32];
        let new_generation = u64::MAX;
        cache.entries.insert(
            new_commitment,
            AuthenticatedPeerRelayReplayEntry {
                observed_at: now,
                generation: new_generation,
                state: AuthenticatedPeerRelayReplayState::InFlight,
            },
        );
        cache.order.push_back((new_commitment, new_generation));

        assert!(cache.evict_over_capacity(new_commitment, new_generation));
        assert_eq!(cache.order.front().copied(), Some(oldest_in_flight));
        assert!(!cache.entries.contains_key(&evicted_completed.0));
        assert!(cache.entries.contains_key(&oldest_in_flight.0));
        assert!(cache.entries.contains_key(&new_commitment));
    }

    #[test]
    fn authenticated_rate_limit_isolated_by_verified_node_id() {
        let started_at = Instant::now();
        let first = [0x31; 32];
        let second = [0x32; 32];
        let mut limiter = AuthenticatedPeerRelayRateLimiter::default();

        assert!(limiter.allow(first, started_at, 1));
        assert!(!limiter.allow(first, started_at + Duration::from_secs(59), 1));
        assert!(limiter.allow(second, started_at + Duration::from_secs(59), 1));
        assert!(limiter.allow(first, started_at + Duration::from_secs(60), 1));
        assert_eq!(limiter.buckets.len(), 2);
    }

    #[test]
    fn rate_limit_uses_exact_monotonic_windows() {
        let started_at = Instant::now();
        let mut window = PeerRelayRateLimitWindow::new(started_at);

        assert!(window.allow(started_at, 2));
        assert!(window.allow(started_at + Duration::from_secs(59), 2));
        assert!(!window.allow(started_at + Duration::from_secs(59), 2));
        assert!(window.allow(started_at + Duration::from_secs(60), 2));
        assert_eq!(window.admitted(), 1);
    }
}
