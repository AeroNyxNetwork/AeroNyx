// ============================================
// File: crates/aeronyx-server/src/api/chat_peer_retry.rs
// ============================================
//! # Blind Relay Retry Policy
//!
//! ## Creation Reason
//! Separates deterministic retry and downstream-failure classification from
//! asynchronous HTTP forwarding, route-health writes, and telemetry effects.
//!
//! ## Main Functionality
//! - Defines the replaceable [`BlindRelayRetryPolicy`] capability.
//! - Models retry, deterministic rejection, and exhausted outcomes explicitly.
//! - Classifies authenticated coarse peer failures without assigning deeper-hop blame.
//! - Produces bounded deterministic jitter from route metadata only.
//! - Keeps transport failure categories free of payload and user dimensions.
//!
//! ## Dependencies
//! - Reads the stable `PeerBlindRelayResponse` shape from `chat_peer.rs`.
//! - Uses `reqwest::StatusCode` only as the shared HTTP status value type.
//! - Performs no network I/O and owns no clocks, storage, telemetry, or identities.
//!
//! ## Main Logical Flow
//! 1. Classify a bounded peer-declared failure when its response shape is valid.
//! 2. Otherwise classify the bare HTTP status or coarse transport failure.
//! 3. Retry only retryable categories while attempt capacity remains.
//! 4. Derive deterministic bounded delay from route id, next hop, and attempt.
//! 5. Return a domain action for the HTTP orchestrator to execute and observe.
//!
//! ## Important Note for Next Developer
//! - Never add ciphertext, sender, receiver, wallet, endpoint, or source IP inputs.
//! - `Reject` and `Exhausted` are observably distinct; preserve that distinction.
//! - Peer-declared deeper-hop failures must not mutate immediate-hop reputation.
//! - Keep default timing and attempt count compatible across rolling upgrades.
//!
//! ## Last Modified
//! v2.8.35-ChatPeerRetryDomain - Initial trait-based retry policy extraction.
//! ============================================

use std::{num::NonZeroUsize, time::Duration};

use reqwest::StatusCode;

use super::chat_peer::PeerBlindRelayResponse;

const DEFAULT_MAX_ATTEMPTS: usize = 3;
const DEFAULT_RETRY_BASE_MS: u64 = 25;
const DEFAULT_RETRY_JITTER_MS: u64 = 35;

/// Coarse transport category accepted by retry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayTransportFailureKind {
    Timeout,
    Connect,
    Request,
    Other,
}

impl BlindRelayTransportFailureKind {
    const fn is_retryable(self) -> bool {
        matches!(self, Self::Timeout | Self::Connect | Self::Request)
    }
}

/// Stable downstream failure classes mapped by the HTTP orchestration layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayDownstreamFailure {
    OnionTerminalCapacityExhausted,
    ForwardFailed,
    DownstreamRejected,
}

/// Retry decision for one bare status or transport failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayRetryAction {
    RetryAfter(Duration),
    /// Deterministic failure that must not be counted as retry exhaustion.
    Reject(BlindRelayDownstreamFailure),
    /// Retryable capacity was consumed, or the category is not retryable.
    Exhausted,
}

/// Payload-blind metadata used to derive one retry decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlindRelayRetryContext {
    route_id: [u8; 16],
    next_hop: [u8; 32],
    attempt: NonZeroUsize,
}

impl BlindRelayRetryContext {
    pub(super) const fn new(
        route_id: [u8; 16],
        next_hop: [u8; 32],
        attempt: usize,
    ) -> Option<Self> {
        let Some(attempt) = NonZeroUsize::new(attempt) else {
            return None;
        };
        Some(Self {
            route_id,
            next_hop,
            attempt,
        })
    }
}

/// Replaceable blind-relay retry and failure-classification capability.
///
/// [BLIND-RETRY-DOMAIN 2026-08-26 by Codex] This trait deliberately excludes
/// I/O and telemetry so policy tests are deterministic and alternate transport
/// implementations can reuse the exact same compatibility behavior.
pub(super) trait BlindRelayRetryPolicy: Send + Sync {
    fn max_attempts(&self) -> NonZeroUsize;

    fn classify_declared_failure(
        &self,
        status: StatusCode,
        response: Option<&PeerBlindRelayResponse>,
    ) -> Option<BlindRelayDownstreamFailure>;

    fn status_action(
        &self,
        status: StatusCode,
        context: BlindRelayRetryContext,
    ) -> BlindRelayRetryAction;

    fn transport_action(
        &self,
        failure: BlindRelayTransportFailureKind,
        context: BlindRelayRetryContext,
    ) -> BlindRelayRetryAction;
}

/// Production retry policy matching the established blind-relay contract.
#[derive(Debug, Clone)]
pub(super) struct BlindRelayRetryDomain {
    max_attempts: usize,
    retry_base: Duration,
    retry_jitter: Duration,
}

impl Default for BlindRelayRetryDomain {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            retry_base: Duration::from_millis(DEFAULT_RETRY_BASE_MS),
            retry_jitter: Duration::from_millis(DEFAULT_RETRY_JITTER_MS),
        }
    }
}

impl BlindRelayRetryPolicy for BlindRelayRetryDomain {
    fn max_attempts(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.max_attempts).unwrap_or(NonZeroUsize::MIN)
    }

    fn classify_declared_failure(
        &self,
        status: StatusCode,
        response: Option<&PeerBlindRelayResponse>,
    ) -> Option<BlindRelayDownstreamFailure> {
        // [SIGNED-FAILURE-RECEIPT 2026-08-11 by Codex] Classification validates
        // only bounded error shape and transport class. Signature verification
        // remains in orchestration, and classification grants no blame authority.
        // [DOWNSTREAM-FAILURE-ATTRIBUTION 2026-08-11 by Codex] Valid protocol
        // errors preserve coarse control flow without poisoning route health.
        let response = response?;
        if response.accepted
            || response.terminal
            || response.forwarded
            || response.ttl_remaining != 0
            || response.delivery_receipt.is_some()
        {
            return None;
        }
        let declared_reason = response.reason.as_deref();
        if status == StatusCode::SERVICE_UNAVAILABLE
            && declared_reason == Some("onion_terminal_capacity_exhausted")
        {
            // [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] Only the exact
            // bounded peer error marks deterministic terminal capacity.
            return Some(BlindRelayDownstreamFailure::OnionTerminalCapacityExhausted);
        }
        if status == StatusCode::BAD_GATEWAY
            && matches!(
                declared_reason,
                Some("forward_failed" | "no_route" | "invalid_endpoint")
            )
        {
            return Some(BlindRelayDownstreamFailure::ForwardFailed);
        }
        if status.is_client_error() && status != StatusCode::TOO_MANY_REQUESTS {
            return Some(BlindRelayDownstreamFailure::DownstreamRejected);
        }
        None
    }

    fn status_action(
        &self,
        status: StatusCode,
        context: BlindRelayRetryContext,
    ) -> BlindRelayRetryAction {
        // A bare deterministic client status is direct endpoint evidence. Keep
        // it distinct from authenticated deeper-hop failures and exhaustion.
        if status.is_client_error() && status != StatusCode::TOO_MANY_REQUESTS {
            return BlindRelayRetryAction::Reject(BlindRelayDownstreamFailure::DownstreamRejected);
        }
        if context.attempt.get() < self.max_attempts
            && (status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error())
        {
            return BlindRelayRetryAction::RetryAfter(self.retry_delay(context));
        }
        BlindRelayRetryAction::Exhausted
    }

    fn transport_action(
        &self,
        failure: BlindRelayTransportFailureKind,
        context: BlindRelayRetryContext,
    ) -> BlindRelayRetryAction {
        if context.attempt.get() < self.max_attempts && failure.is_retryable() {
            return BlindRelayRetryAction::RetryAfter(self.retry_delay(context));
        }
        BlindRelayRetryAction::Exhausted
    }
}

impl BlindRelayRetryDomain {
    fn retry_delay(&self, context: BlindRelayRetryContext) -> Duration {
        let mut seed = u64::try_from(context.attempt.get()).unwrap_or(u64::MAX);
        for byte in context.route_id.iter().chain(context.next_hop.iter()) {
            seed = seed.wrapping_mul(31).wrapping_add(u64::from(*byte));
        }
        let jitter_window_ms = u64::try_from(self.retry_jitter.as_millis()).unwrap_or(u64::MAX);
        let jitter_ms = seed % jitter_window_ms.saturating_add(1);
        self.retry_base
            .saturating_add(Duration::from_millis(jitter_ms))
    }
}

#[cfg(test)]
pub(super) const DEFAULT_MAX_ATTEMPTS_FOR_TESTS: usize = DEFAULT_MAX_ATTEMPTS;

#[cfg(test)]
mod tests {
    use super::*;

    fn failure_response(reason: &str) -> PeerBlindRelayResponse {
        PeerBlindRelayResponse {
            accepted: false,
            terminal: false,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some(reason.to_string()),
            delivery_receipt: None,
            failure_receipt: None,
        }
    }

    fn context(attempt: usize) -> BlindRelayRetryContext {
        match BlindRelayRetryContext::new([0x41; 16], [0x42; 32], attempt) {
            Some(context) => context,
            None => panic!("test retry attempt must be non-zero"),
        }
    }

    #[test]
    fn declared_failures_require_exact_bounded_shape() {
        let policy = BlindRelayRetryDomain::default();
        assert_eq!(
            policy.classify_declared_failure(
                StatusCode::SERVICE_UNAVAILABLE,
                Some(&failure_response("onion_terminal_capacity_exhausted")),
            ),
            Some(BlindRelayDownstreamFailure::OnionTerminalCapacityExhausted)
        );
        assert_eq!(
            policy.classify_declared_failure(
                StatusCode::BAD_GATEWAY,
                Some(&failure_response("forward_failed")),
            ),
            Some(BlindRelayDownstreamFailure::ForwardFailed)
        );
        assert_eq!(
            policy.classify_declared_failure(
                StatusCode::BAD_REQUEST,
                Some(&failure_response("downstream_rejected")),
            ),
            Some(BlindRelayDownstreamFailure::DownstreamRejected)
        );

        let mut malformed = failure_response("forward_failed");
        malformed.accepted = true;
        assert_eq!(
            policy.classify_declared_failure(StatusCode::BAD_GATEWAY, Some(&malformed)),
            None
        );
        assert_eq!(
            policy.classify_declared_failure(
                StatusCode::SERVICE_UNAVAILABLE,
                Some(&failure_response("proxy_unavailable")),
            ),
            None
        );
    }

    #[test]
    fn status_policy_preserves_reject_retry_and_exhausted_distinction() {
        let policy = BlindRelayRetryDomain::default();
        assert_eq!(
            policy.status_action(StatusCode::BAD_REQUEST, context(1)),
            BlindRelayRetryAction::Reject(BlindRelayDownstreamFailure::DownstreamRejected)
        );
        assert!(matches!(
            policy.status_action(StatusCode::TOO_MANY_REQUESTS, context(1)),
            BlindRelayRetryAction::RetryAfter(_)
        ));
        assert!(matches!(
            policy.status_action(StatusCode::BAD_GATEWAY, context(2)),
            BlindRelayRetryAction::RetryAfter(_)
        ));
        assert_eq!(
            policy.status_action(StatusCode::BAD_GATEWAY, context(DEFAULT_MAX_ATTEMPTS)),
            BlindRelayRetryAction::Exhausted
        );
        assert_eq!(
            policy.status_action(StatusCode::FOUND, context(1)),
            BlindRelayRetryAction::Exhausted
        );
    }

    #[test]
    fn transport_policy_retries_only_transient_categories_with_capacity() {
        let policy = BlindRelayRetryDomain::default();
        for failure in [
            BlindRelayTransportFailureKind::Timeout,
            BlindRelayTransportFailureKind::Connect,
            BlindRelayTransportFailureKind::Request,
        ] {
            assert!(matches!(
                policy.transport_action(failure, context(1)),
                BlindRelayRetryAction::RetryAfter(_)
            ));
        }
        assert_eq!(
            policy.transport_action(BlindRelayTransportFailureKind::Other, context(1)),
            BlindRelayRetryAction::Exhausted
        );
        assert_eq!(
            policy.transport_action(
                BlindRelayTransportFailureKind::Timeout,
                context(DEFAULT_MAX_ATTEMPTS),
            ),
            BlindRelayRetryAction::Exhausted
        );
    }

    #[test]
    fn retry_delay_is_deterministic_and_bounded() {
        let policy = BlindRelayRetryDomain::default();
        let first = match policy.status_action(StatusCode::BAD_GATEWAY, context(1)) {
            BlindRelayRetryAction::RetryAfter(delay) => delay,
            action => panic!("unexpected retry action: {action:?}"),
        };
        let repeated = match policy.status_action(StatusCode::BAD_GATEWAY, context(1)) {
            BlindRelayRetryAction::RetryAfter(delay) => delay,
            action => panic!("unexpected retry action: {action:?}"),
        };
        assert_eq!(first, repeated);
        assert!(first >= Duration::from_millis(DEFAULT_RETRY_BASE_MS));
        assert!(first <= Duration::from_millis(DEFAULT_RETRY_BASE_MS + DEFAULT_RETRY_JITTER_MS));
    }

    #[test]
    fn retry_context_rejects_zero_attempt() {
        assert_eq!(BlindRelayRetryContext::new([0x41; 16], [0x42; 32], 0), None);
    }
}
