// ============================================
// File: crates/aeronyx-server/src/api/chat_peer_abuse_guard.rs
// ============================================
//! # Blind Relay Abuse Guard Domain
//!
//! ## Creation Reason
//! Separates privacy-safe blind-relay admission, previous-hop rate limiting,
//! failure decay, quarantine, and bounded identity retention from the public
//! HTTP orchestration in `chat_peer.rs`.
//!
//! ## Main Functionality
//! - Defines the replaceable [`BlindRelayAbusePolicy`] capability.
//! - Owns synchronized process-wide and verified previous-hop admission state.
//! - Preserves active quarantines while bounding permissionless identity churn.
//! - Projects monotonic enforcement deadlines into coarse epoch timestamps for
//!   the existing operator telemetry contract.
//!
//! ## Dependencies
//! - Reuses the fixed-window primitive from `chat_peer_admission.rs`.
//! - Is composed by `ChatPeerState`; it does not know HTTP, payloads, routes,
//!   receivers, endpoints, source IP addresses, or encrypted message contents.
//!
//! ## Main Logical Flow
//! 1. Apply an aggregate parser-front window before expensive request work.
//! 2. Admit only cryptographically verified previous-hop node identities.
//! 3. Quarantine rate or validation abuse using monotonic time.
//! 4. Evict idle non-quarantined buckets when fixed capacity is reached.
//!
//! ## Important Note for Next Developer
//! - Keep keys limited to authenticated node identities; never key this policy
//!   by user, receiver, route, source IP, endpoint, or ciphertext-derived data.
//! - Keep all enforcement clocks monotonic. Epoch values are observability only.
//! - Maintain `BlindRelayAbusePolicy` compatibility with `chat_peer.rs`.
//!
//! ## Last Modified
//! v2.8.32-ChatPeerAbuseDomain - Initial domain extraction.
//! ============================================

use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

use super::chat_peer_admission::PeerRelayRateLimitWindow;

/// Per previous-hop accepted relay attempts allowed in the short window.
const PREVIOUS_HOP_RATE_LIMIT: u32 = 120;
/// Sliding window for previous-hop relay rate limiting.
const PREVIOUS_HOP_RATE_WINDOW_SECS: u64 = 60;
/// Privacy-safe failure score that puts one previous-hop node into quarantine.
pub(super) const PREVIOUS_HOP_FAILURE_THRESHOLD: u32 = 12;
/// Failure score decay horizon before a previous-hop gets a clean bucket.
const PREVIOUS_HOP_FAILURE_WINDOW_SECS: u64 = 5 * 60;
/// Short local quarantine for noisy previous-hop nodes.
const PREVIOUS_HOP_QUARANTINE_SECS: u64 = 5 * 60;
/// Maximum previous-hop abuse buckets retained by this process.
const MAX_PREVIOUS_HOP_BUCKETS: usize = 4096;

/// Outcome of privacy-safe admission for one authenticated previous-hop node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayAbuseDecision {
    Allowed,
    CapacityLimited,
    RateLimited { quarantine_until: u64 },
    Quarantined { quarantine_until: u64 },
}

/// Replaceable blind-relay abuse-control capability.
///
/// [CHAT-PEER-ABUSE-DOMAIN 2026-08-26 by Codex] The HTTP layer depends on this
/// behavioral boundary rather than owning locks or mutable policy internals.
pub(super) trait BlindRelayAbusePolicy: Send + Sync {
    fn admit_global(&self, now: Instant, limit: u32) -> bool;

    fn observe_request(
        &self,
        previous_hop: [u8; 32],
        observed_at_epoch: u64,
    ) -> BlindRelayAbuseDecision;

    fn record_failure(&self, previous_hop: [u8; 32], observed_at_epoch: u64) -> Option<u64>;

    fn record_success(&self, previous_hop: [u8; 32]);
}

/// Production composition of the blind-relay abuse policy.
pub(super) struct BlindRelayAbuseDomain {
    state: Mutex<BlindRelayAbuseState>,
}

impl Default for BlindRelayAbuseDomain {
    fn default() -> Self {
        Self {
            state: Mutex::new(BlindRelayAbuseState::new(Instant::now())),
        }
    }
}

impl BlindRelayAbusePolicy for BlindRelayAbuseDomain {
    fn admit_global(&self, now: Instant, limit: u32) -> bool {
        self.with_state(|state| state.admit_global(now, limit))
    }

    fn observe_request(
        &self,
        previous_hop: [u8; 32],
        observed_at_epoch: u64,
    ) -> BlindRelayAbuseDecision {
        self.with_state(|state| {
            state.observe_request_at(previous_hop, observed_at_epoch, Instant::now())
        })
    }

    fn record_failure(&self, previous_hop: [u8; 32], observed_at_epoch: u64) -> Option<u64> {
        self.with_state(|state| {
            state.record_failure_at(previous_hop, observed_at_epoch, Instant::now())
        })
    }

    fn record_success(&self, previous_hop: [u8; 32]) {
        self.with_state(|state| state.record_success_at(previous_hop, Instant::now()));
    }
}

impl BlindRelayAbuseDomain {
    fn with_state<T>(&self, operation: impl FnOnce(&mut BlindRelayAbuseState) -> T) -> T {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        operation(&mut state)
    }
}

#[derive(Debug)]
struct PreviousHopBucket {
    rate_window: PeerRelayRateLimitWindow,
    failure_window_started_at: Instant,
    failure_score: u32,
    quarantine_until: Option<Instant>,
    last_seen_at: Instant,
}

impl PreviousHopBucket {
    const fn new(now: Instant) -> Self {
        Self {
            rate_window: PeerRelayRateLimitWindow::new(now),
            failure_window_started_at: now,
            failure_score: 0,
            quarantine_until: None,
            last_seen_at: now,
        }
    }
}

#[derive(Debug)]
struct BlindRelayAbuseState {
    buckets: HashMap<[u8; 32], PreviousHopBucket>,
    global_rate_limit: PeerRelayRateLimitWindow,
}

impl BlindRelayAbuseState {
    fn new(now: Instant) -> Self {
        Self {
            buckets: HashMap::new(),
            global_rate_limit: PeerRelayRateLimitWindow::new(now),
        }
    }

    fn admit_global(&mut self, now: Instant, limit: u32) -> bool {
        self.global_rate_limit.allow(now, limit)
    }

    fn observe_request_at(
        &mut self,
        previous_hop: [u8; 32],
        observed_at_epoch: u64,
        observed_at: Instant,
    ) -> BlindRelayAbuseDecision {
        let Some(bucket) = self.bucket_mut(previous_hop, observed_at) else {
            return BlindRelayAbuseDecision::CapacityLimited;
        };
        bucket.last_seen_at = observed_at;

        if let Some(quarantine_until) = bucket
            .quarantine_until
            .filter(|quarantine_until| observed_at < *quarantine_until)
        {
            return BlindRelayAbuseDecision::Quarantined {
                quarantine_until: project_monotonic_deadline_to_epoch(
                    observed_at_epoch,
                    observed_at,
                    quarantine_until,
                ),
            };
        }
        bucket.quarantine_until = None;

        if !bucket.rate_window.allow_for(
            observed_at,
            PREVIOUS_HOP_RATE_LIMIT,
            Duration::from_secs(PREVIOUS_HOP_RATE_WINDOW_SECS),
        ) {
            let quarantine_deadline = observed_at
                .checked_add(Duration::from_secs(PREVIOUS_HOP_QUARANTINE_SECS))
                .unwrap_or(observed_at);
            bucket.quarantine_until = Some(quarantine_deadline);
            return BlindRelayAbuseDecision::RateLimited {
                quarantine_until: observed_at_epoch.saturating_add(PREVIOUS_HOP_QUARANTINE_SECS),
            };
        }

        BlindRelayAbuseDecision::Allowed
    }

    fn record_failure_at(
        &mut self,
        previous_hop: [u8; 32],
        observed_at_epoch: u64,
        observed_at: Instant,
    ) -> Option<u64> {
        let bucket = self.bucket_mut(previous_hop, observed_at)?;
        bucket.last_seen_at = observed_at;
        if observed_at.saturating_duration_since(bucket.failure_window_started_at)
            >= Duration::from_secs(PREVIOUS_HOP_FAILURE_WINDOW_SECS)
        {
            bucket.failure_window_started_at = observed_at;
            bucket.failure_score = 0;
        }

        bucket.failure_score = bucket.failure_score.saturating_add(1);
        if bucket.failure_score >= PREVIOUS_HOP_FAILURE_THRESHOLD {
            let quarantine_deadline = observed_at
                .checked_add(Duration::from_secs(PREVIOUS_HOP_QUARANTINE_SECS))
                .unwrap_or(observed_at);
            bucket.quarantine_until = Some(quarantine_deadline);
            bucket.failure_score = 0;
            return Some(observed_at_epoch.saturating_add(PREVIOUS_HOP_QUARANTINE_SECS));
        }
        None
    }

    fn record_success_at(&mut self, previous_hop: [u8; 32], observed_at: Instant) {
        if let Some(bucket) = self.buckets.get_mut(&previous_hop) {
            bucket.last_seen_at = observed_at;
            if observed_at.saturating_duration_since(bucket.failure_window_started_at)
                >= Duration::from_secs(PREVIOUS_HOP_FAILURE_WINDOW_SECS)
            {
                bucket.failure_window_started_at = observed_at;
                bucket.failure_score = 0;
            }
        }
    }

    fn bucket_mut(
        &mut self,
        previous_hop: [u8; 32],
        observed_at: Instant,
    ) -> Option<&mut PreviousHopBucket> {
        if !self.buckets.contains_key(&previous_hop) {
            // [BLIND-RELAY-BUCKET-FAIRNESS 2026-08-21 by Codex] Scan only on
            // new verified identities. Identity churn is already bounded by
            // the parser-front global window, so fixed-capacity LRU selection
            // cannot become unbounded per-request work. Active quarantines are
            // never removed merely to admit a fresh permissionless identity.
            self.evict_idle(observed_at);
            if !self.make_room_for_new_bucket(observed_at, MAX_PREVIOUS_HOP_BUCKETS) {
                return None;
            }
        }
        Some(
            self.buckets
                .entry(previous_hop)
                .or_insert_with(|| PreviousHopBucket::new(observed_at)),
        )
    }

    fn evict_idle(&mut self, observed_at: Instant) {
        let retention_secs = PREVIOUS_HOP_FAILURE_WINDOW_SECS + PREVIOUS_HOP_QUARANTINE_SECS;
        self.buckets.retain(|_, bucket| {
            let quarantine_active = bucket
                .quarantine_until
                .is_some_and(|quarantine_until| observed_at < quarantine_until);
            quarantine_active
                || observed_at.saturating_duration_since(bucket.last_seen_at)
                    <= Duration::from_secs(retention_secs)
        });
    }

    fn make_room_for_new_bucket(&mut self, observed_at: Instant, capacity: usize) -> bool {
        if capacity == 0 {
            return false;
        }
        while self.buckets.len() >= capacity {
            let eviction_candidate = self
                .buckets
                .iter()
                .filter(|(_, bucket)| {
                    !bucket
                        .quarantine_until
                        .is_some_and(|quarantine_until| observed_at < quarantine_until)
                })
                .min_by(|(left_id, left), (right_id, right)| {
                    left.last_seen_at
                        .cmp(&right.last_seen_at)
                        .then_with(|| left_id.cmp(right_id))
                })
                .map(|(node_id, _)| *node_id);
            let Some(eviction_candidate) = eviction_candidate else {
                return false;
            };
            self.buckets.remove(&eviction_candidate);
        }
        true
    }
}

fn project_monotonic_deadline_to_epoch(
    observed_at_epoch: u64,
    observed_at: Instant,
    deadline: Instant,
) -> u64 {
    // [BLIND-RELAY-MONOTONIC-ABUSE-CLOCK 2026-08-21 by Codex] Enforcement
    // remains monotonic. This projection exists only for the established
    // PeerStore/API timestamp contract and rounds up so observability never
    // reports an active quarantine as expired one second too early.
    let remaining = deadline.saturating_duration_since(observed_at);
    let remaining_secs = remaining
        .as_secs()
        .saturating_add(u64::from(remaining.subsec_nanos() != 0));
    observed_at_epoch.saturating_add(remaining_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_rate_limit_uses_exact_monotonic_windows() {
        let started_at = Instant::now();
        let mut state = BlindRelayAbuseState::new(started_at);

        assert!(state.admit_global(started_at, 2));
        assert!(state.admit_global(started_at + Duration::from_secs(59), 2));
        assert!(!state.admit_global(started_at + Duration::from_secs(59), 2));
        assert!(state.admit_global(started_at + Duration::from_secs(60), 2));
        assert_eq!(state.global_rate_limit.admitted(), 1);
    }

    #[test]
    fn rate_limits_previous_hop_without_payload_data() {
        let started_at = Instant::now();
        let mut state = BlindRelayAbuseState::new(started_at);
        let previous_hop = [0x52u8; 32];
        let now_epoch = 1_800_000_000;

        for _ in 0..PREVIOUS_HOP_RATE_LIMIT {
            assert_eq!(
                state.observe_request_at(previous_hop, now_epoch, started_at),
                BlindRelayAbuseDecision::Allowed
            );
        }
        assert_eq!(
            state.observe_request_at(previous_hop, now_epoch, started_at),
            BlindRelayAbuseDecision::RateLimited {
                quarantine_until: now_epoch + PREVIOUS_HOP_QUARANTINE_SECS
            }
        );
        assert_eq!(
            state.observe_request_at(
                previous_hop,
                now_epoch + 1,
                started_at + Duration::from_secs(1),
            ),
            BlindRelayAbuseDecision::Quarantined {
                quarantine_until: now_epoch + PREVIOUS_HOP_QUARANTINE_SECS
            }
        );
    }

    #[test]
    fn uses_exact_monotonic_rate_window() {
        let started_at = Instant::now();
        let mut state = BlindRelayAbuseState::new(started_at);
        let previous_hop = [0x54u8; 32];
        let now_epoch = 1_800_000_000;

        for _ in 0..PREVIOUS_HOP_RATE_LIMIT {
            assert_eq!(
                state.observe_request_at(previous_hop, now_epoch, started_at),
                BlindRelayAbuseDecision::Allowed
            );
        }
        assert_eq!(
            state.observe_request_at(
                previous_hop,
                now_epoch + PREVIOUS_HOP_RATE_WINDOW_SECS,
                started_at + Duration::from_secs(PREVIOUS_HOP_RATE_WINDOW_SECS),
            ),
            BlindRelayAbuseDecision::Allowed
        );
    }

    #[test]
    fn ignores_wall_clock_rollback() {
        let started_at = Instant::now();
        let mut state = BlindRelayAbuseState::new(started_at);
        let previous_hop = [0x55u8; 32];
        let started_epoch = 1_800_000_000;
        let rolled_back_epoch = started_epoch - 3_600;

        for _ in 0..PREVIOUS_HOP_RATE_LIMIT {
            assert_eq!(
                state.observe_request_at(previous_hop, started_epoch, started_at),
                BlindRelayAbuseDecision::Allowed
            );
        }
        assert_eq!(
            state.observe_request_at(
                previous_hop,
                rolled_back_epoch,
                started_at + Duration::from_secs(59),
            ),
            BlindRelayAbuseDecision::RateLimited {
                quarantine_until: rolled_back_epoch + PREVIOUS_HOP_QUARANTINE_SECS
            }
        );
        assert_eq!(
            state.observe_request_at(
                previous_hop,
                rolled_back_epoch + PREVIOUS_HOP_QUARANTINE_SECS,
                started_at + Duration::from_secs(59 + PREVIOUS_HOP_QUARANTINE_SECS),
            ),
            BlindRelayAbuseDecision::Allowed
        );
    }

    #[test]
    fn quarantines_repeated_bad_previous_hop() {
        let started_at = Instant::now();
        let mut state = BlindRelayAbuseState::new(started_at);
        let previous_hop = [0x53u8; 32];
        let now_epoch = 1_800_000_000;

        for offset in 0..(PREVIOUS_HOP_FAILURE_THRESHOLD - 1) {
            assert_eq!(
                state.record_failure_at(
                    previous_hop,
                    now_epoch + u64::from(offset),
                    started_at + Duration::from_secs(u64::from(offset)),
                ),
                None
            );
        }
        let quarantine_offset = u64::from(PREVIOUS_HOP_FAILURE_THRESHOLD);
        let quarantine_epoch = now_epoch + quarantine_offset;
        let quarantine_at = started_at + Duration::from_secs(quarantine_offset);
        assert_eq!(
            state.record_failure_at(previous_hop, quarantine_epoch, quarantine_at),
            Some(quarantine_epoch + PREVIOUS_HOP_QUARANTINE_SECS)
        );
        assert_eq!(
            state.observe_request_at(
                previous_hop,
                quarantine_epoch + 1,
                quarantine_at + Duration::from_secs(1),
            ),
            BlindRelayAbuseDecision::Quarantined {
                quarantine_until: quarantine_epoch + PREVIOUS_HOP_QUARANTINE_SECS
            }
        );
    }

    #[test]
    fn decays_failures_at_exact_monotonic_boundary() {
        let started_at = Instant::now();
        let mut state = BlindRelayAbuseState::new(started_at);
        let previous_hop = [0x56u8; 32];
        let now_epoch = 1_800_000_000;

        for _ in 0..(PREVIOUS_HOP_FAILURE_THRESHOLD - 1) {
            assert_eq!(
                state.record_failure_at(previous_hop, now_epoch, started_at),
                None
            );
        }
        assert_eq!(
            state.record_failure_at(
                previous_hop,
                now_epoch + PREVIOUS_HOP_FAILURE_WINDOW_SECS,
                started_at + Duration::from_secs(PREVIOUS_HOP_FAILURE_WINDOW_SECS),
            ),
            None
        );
        assert_eq!(state.buckets[&previous_hop].failure_score, 1);
    }

    #[test]
    fn removes_idle_buckets_behind_active_identity() {
        let started_at = Instant::now();
        let mut state = BlindRelayAbuseState::new(started_at);
        let active = [0x61u8; 32];
        let idle_peer = [0x62u8; 32];
        let newcomer = [0x63u8; 32];
        let started_at_epoch = 1_800_000_000;
        let retention_secs = PREVIOUS_HOP_FAILURE_WINDOW_SECS + PREVIOUS_HOP_QUARANTINE_SECS;
        let now_epoch = started_at_epoch + retention_secs + 2;
        let now = started_at + Duration::from_secs(retention_secs + 2);

        assert_eq!(
            state.observe_request_at(active, started_at_epoch, started_at),
            BlindRelayAbuseDecision::Allowed
        );
        assert_eq!(
            state.observe_request_at(
                idle_peer,
                started_at_epoch + 1,
                started_at + Duration::from_secs(1),
            ),
            BlindRelayAbuseDecision::Allowed
        );
        state.record_success_at(active, now);
        assert_eq!(
            state.observe_request_at(newcomer, now_epoch, now),
            BlindRelayAbuseDecision::Allowed
        );
        assert!(state.buckets.contains_key(&active));
        assert!(state.buckets.contains_key(&newcomer));
        assert!(!state.buckets.contains_key(&idle_peer));
    }

    #[test]
    fn evicts_lru_without_erasing_active_quarantine() {
        let baseline = Instant::now();
        let now = baseline + Duration::from_secs(100);
        let mut state = BlindRelayAbuseState::new(now);
        let quarantined = [0x71u8; 32];
        let evictable = [0x72u8; 32];
        let mut quarantined_bucket = PreviousHopBucket::new(baseline);
        quarantined_bucket.quarantine_until = Some(now + Duration::from_secs(60));
        state.buckets.insert(quarantined, quarantined_bucket);
        state.buckets.insert(
            evictable,
            PreviousHopBucket::new(baseline + Duration::from_secs(90)),
        );

        assert!(state.make_room_for_new_bucket(now, 2));
        assert!(state.buckets.contains_key(&quarantined));
        assert!(!state.buckets.contains_key(&evictable));
    }

    #[test]
    fn rejects_new_identity_when_all_buckets_quarantined() {
        let now_epoch = 1_800_000_000;
        let now = Instant::now();
        let mut state = BlindRelayAbuseState::new(now);
        for index in 0..MAX_PREVIOUS_HOP_BUCKETS {
            let mut node_id = [0u8; 32];
            node_id[..8].copy_from_slice(&(index as u64).to_be_bytes());
            let mut bucket = PreviousHopBucket::new(now);
            bucket.quarantine_until = Some(now + Duration::from_secs(60));
            state.buckets.insert(node_id, bucket);
        }

        assert_eq!(
            state.observe_request_at([0xffu8; 32], now_epoch, now),
            BlindRelayAbuseDecision::CapacityLimited
        );
        assert_eq!(state.buckets.len(), MAX_PREVIOUS_HOP_BUCKETS);
    }
}
