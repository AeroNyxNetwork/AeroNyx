// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer_observer.rs
// ============================================================================
//! # Blind Relay Forward Observer
//!
//! ## Creation Reason
//! Separates aggregate forwarding observations from the retry loop so relay
//! orchestration no longer depends directly on `PeerStore` persistence.
//!
//! ## Main Functionality
//! - Defines the replaceable [`BlindRelayForwardObserver`] capability.
//! - Records retry attempts, recoveries, and exhausted attempts.
//! - Records aggregate peer rejection without route attribution.
//! - Records descriptor-bound route failure plus aggregate rejection.
//! - Provides the production [`PeerStoreBlindRelayForwardObserver`] adapter.
//!
//! ## Dependencies
//! - Uses signed node descriptors only for descriptor-bound route evidence.
//! - Delegates durable aggregate counters and lifecycle events to `PeerStore`.
//! - Performs no transport, retry, receipt, identity, or payload processing.
//!
//! ## Main Logical Flow
//! 1. Receive one already-classified forwarding observation.
//! 2. Persist only aggregate privacy-safe reason buckets.
//! 3. Bind immediate-hop route failures to the exact signed descriptor.
//! 4. Keep deeper-hop declared failures unattributed.
//!
//! ## Important Note for Next Developer
//! - Never add ciphertext, route ids, users, wallets, endpoints, or source IPs.
//! - `rejected` must not mutate immediate-hop route health.
//! - `route_failed` must retain descriptor binding to avoid stale-surface writes.
//! - Keep retry counters and reason buckets compatible with nodeboard telemetry.
//!
//! ## Last Modified
//! v2.8.38-ChatPeerObserverDomain - Initial trait-based observation adapter.
//! ============================================================================

use aeronyx_core::protocol::discovery::SignedNodeDescriptor;

use crate::services::peer_store::PeerStore;

/// Replaceable aggregate forwarding observation capability.
///
/// [BLIND-FORWARD-OBSERVER 2026-08-26 by Codex] This interface receives only
/// decisions that have already passed transport and response policy. Keeping
/// observation write-only prevents persistence from influencing relay control.
pub(super) trait BlindRelayForwardObserver: Send + Sync {
    fn retry_attempted(&self, observed_at: u64, reason: &str);
    fn retry_succeeded(&self, observed_at: u64, attempt: usize);
    fn retry_exhausted(&self, observed_at: u64, attempt: usize, reason: &str);
    fn rejected(&self, observed_at: u64, reason: &str);
    fn route_failed(&self, descriptor: &SignedNodeDescriptor, observed_at: u64, reason: &str);
}

/// Production observer backed by the node's shared peer store.
#[derive(Debug, Clone, Copy)]
pub(super) struct PeerStoreBlindRelayForwardObserver<'a> {
    peer_store: &'a PeerStore,
}

impl<'a> PeerStoreBlindRelayForwardObserver<'a> {
    pub(super) const fn new(peer_store: &'a PeerStore) -> Self {
        Self { peer_store }
    }
}

impl BlindRelayForwardObserver for PeerStoreBlindRelayForwardObserver<'_> {
    fn retry_attempted(&self, observed_at: u64, reason: &str) {
        self.peer_store
            .record_blind_relay_retry_attempt(observed_at, reason);
    }

    fn retry_succeeded(&self, observed_at: u64, attempt: usize) {
        self.peer_store
            .record_blind_relay_retry_succeeded(observed_at, attempt);
    }

    fn retry_exhausted(&self, observed_at: u64, attempt: usize, reason: &str) {
        self.peer_store
            .record_blind_relay_retry_exhausted(observed_at, attempt, reason);
    }

    fn rejected(&self, observed_at: u64, reason: &str) {
        self.peer_store
            .record_blind_relay_rejected(observed_at, reason);
    }

    fn route_failed(&self, descriptor: &SignedNodeDescriptor, observed_at: u64, reason: &str) {
        let _ = self.peer_store.record_route_forward_failure_for_descriptor(
            descriptor,
            observed_at,
            reason,
        );
        self.rejected(observed_at, reason);
    }
}
