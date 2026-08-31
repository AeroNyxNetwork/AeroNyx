// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer_transport.rs
// ============================================================================
//! # Blind Relay HTTP Transport
//!
//! ## Creation Reason
//! Separates peer HTTP I/O and bounded response decoding from blind-relay
//! routing, receipt verification, retry policy, and route-health effects.
//!
//! ## Main Functionality
//! - Defines the replaceable [`BlindRelayTransport`] capability.
//! - Models successful HTTP status, rejected HTTP status, and request failure.
//! - Reads every peer response through the shared bounded JSON decoder.
//! - Classifies request failures into privacy-safe operational categories.
//! - Provides the production [`ReqwestBlindRelayTransport`] adapter.
//!
//! ## Dependencies
//! - Uses the stable blind-relay request/response types from `chat_peer.rs`.
//! - Uses the coarse retry failure category from `chat_peer_retry.rs`.
//! - Uses the shared peer response byte limit and bounded decoder in `api/mod.rs`.
//!
//! ## Main Logical Flow
//! 1. Send one already-prepared opaque blind-relay body to the selected URL.
//! 2. Classify request-layer failure without retaining endpoint or payload text.
//! 3. For every HTTP response, consume JSON under the shared byte ceiling.
//! 4. Return a typed transport outcome to the route orchestrator.
//!
//! ## Important Note for Next Developer
//! - Never add retry, signature, receipt, route-health, or telemetry side effects.
//! - Never retain or log URLs, ciphertext, users, wallets, or source addresses.
//! - Non-success response decoding is intentionally best-effort for legacy peers.
//! - A successful HTTP status with malformed JSON remains a protocol failure.
//! - Keep the shared response size ceiling unchanged across rolling upgrades.
//!
//! ## Last Modified
//! v2.8.37-PreparedBlindForwardBody - Reuse one immutable serialized body for
//! every exact retry and keep ciphertext encoding outside async HTTP workers.
//! v2.8.36-ChatPeerTransportDomain - Initial trait-based HTTP extraction.
//! ============================================================================

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use reqwest::{Client, StatusCode};

use super::chat_peer::PeerBlindRelayResponse;
use super::chat_peer_retry::BlindRelayTransportFailureKind;
use crate::api::{
    decode_bounded_json_response, BoundedHttpResponseError, BLIND_RELAY_ACK_RESPONSE_MAX_BYTES,
};

/// Result of one outbound blind-relay HTTP exchange.
///
/// [BLIND-TRANSPORT-DOMAIN 2026-08-26 by Codex] The transport boundary owns
/// HTTP and bounded decoding only. It deliberately does not decide whether a
/// decoded ACK is authentic, retryable, attributable, or healthy.
#[derive(Debug)]
pub(super) enum BlindRelayTransportOutcome {
    /// The peer returned a successful HTTP status. Decoding remains explicit
    /// because malformed success responses are protocol failures, not I/O.
    SuccessStatus {
        response: Result<Box<PeerBlindRelayResponse>, BoundedHttpResponseError>,
    },
    /// The peer returned a non-success status. A bounded declared response is
    /// optional so legacy status-only peers remain compatible.
    RejectedStatus {
        status: StatusCode,
        declared_response: Option<Box<PeerBlindRelayResponse>>,
    },
    /// The request failed before a usable HTTP response was available.
    RequestFailed(BlindRelayTransportFailure),
}

/// Privacy-safe request-layer failure returned by the transport adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlindRelayTransportFailure {
    kind: BlindRelayTransportErrorKind,
}

impl BlindRelayTransportFailure {
    fn from_reqwest(error: &reqwest::Error) -> Self {
        let kind = if error.is_timeout() {
            BlindRelayTransportErrorKind::Timeout
        } else if error.is_connect() {
            BlindRelayTransportErrorKind::Connect
        } else if error.is_status() {
            BlindRelayTransportErrorKind::HttpStatus(error.status().map(|status| status.as_u16()))
        } else if error.is_decode() {
            BlindRelayTransportErrorKind::Decode
        } else if error.is_body() {
            BlindRelayTransportErrorKind::Body
        } else if error.is_request() {
            BlindRelayTransportErrorKind::Request
        } else {
            BlindRelayTransportErrorKind::Unknown
        };
        Self { kind }
    }

    /// Stable aggregate telemetry bucket matching the legacy classifier.
    pub(super) fn reason_bucket(self) -> String {
        match self.kind {
            BlindRelayTransportErrorKind::Timeout => "blind_relay_request_timeout".to_owned(),
            BlindRelayTransportErrorKind::Connect => "blind_relay_request_connect".to_owned(),
            BlindRelayTransportErrorKind::HttpStatus(Some(status)) => {
                format!("blind_relay_request_http_{status}")
            }
            BlindRelayTransportErrorKind::HttpStatus(None) => {
                "blind_relay_request_http_status".to_owned()
            }
            BlindRelayTransportErrorKind::Decode => "blind_relay_request_decode".to_owned(),
            BlindRelayTransportErrorKind::Body => "blind_relay_request_body".to_owned(),
            BlindRelayTransportErrorKind::Request => "blind_relay_request_request".to_owned(),
            BlindRelayTransportErrorKind::Unknown => "blind_relay_request_unknown".to_owned(),
        }
    }

    /// Coarse category consumed by the payload-blind retry policy.
    pub(super) const fn retry_kind(self) -> BlindRelayTransportFailureKind {
        match self.kind {
            BlindRelayTransportErrorKind::Timeout => BlindRelayTransportFailureKind::Timeout,
            BlindRelayTransportErrorKind::Connect => BlindRelayTransportFailureKind::Connect,
            BlindRelayTransportErrorKind::Request => BlindRelayTransportFailureKind::Request,
            BlindRelayTransportErrorKind::HttpStatus(_)
            | BlindRelayTransportErrorKind::Decode
            | BlindRelayTransportErrorKind::Body
            | BlindRelayTransportErrorKind::Unknown => BlindRelayTransportFailureKind::Other,
        }
    }
}

/// Internal request failure detail. No variant stores peer-controlled text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlindRelayTransportErrorKind {
    Timeout,
    Connect,
    HttpStatus(Option<u16>),
    Decode,
    Body,
    Request,
    Unknown,
}

/// Replaceable outbound blind-relay transport capability.
#[async_trait]
pub(super) trait BlindRelayTransport: Send + Sync {
    async fn send(&self, url: &str, body: Bytes) -> BlindRelayTransportOutcome;
}

/// Production HTTP adapter backed by the server's shared reqwest client.
#[derive(Debug, Clone)]
pub(super) struct ReqwestBlindRelayTransport {
    client: Arc<Client>,
}

impl ReqwestBlindRelayTransport {
    pub(super) const fn new(client: Arc<Client>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl BlindRelayTransport for ReqwestBlindRelayTransport {
    async fn send(&self, url: &str, body: Bytes) -> BlindRelayTransportOutcome {
        // [PREPARED-BLIND-FORWARD-CARRIER 2026-08-31 by Codex] The caller
        // prepared and bounded these exact bytes before arming route effects.
        // Transport retries clone only the reference-counted byte carrier.
        let response = match self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                return BlindRelayTransportOutcome::RequestFailed(
                    BlindRelayTransportFailure::from_reqwest(&error),
                );
            }
        };

        let status = response.status();
        let decoded = decode_bounded_json_response::<PeerBlindRelayResponse>(
            response,
            BLIND_RELAY_ACK_RESPONSE_MAX_BYTES,
        )
        .await;
        if status.is_success() {
            BlindRelayTransportOutcome::SuccessStatus {
                response: decoded.map(Box::new),
            }
        } else {
            BlindRelayTransportOutcome::RejectedStatus {
                status,
                declared_response: decoded.ok().map(Box::new),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_buckets_and_retry_kinds_preserve_legacy_contract() {
        let cases = [
            (
                BlindRelayTransportErrorKind::Timeout,
                "blind_relay_request_timeout",
                BlindRelayTransportFailureKind::Timeout,
            ),
            (
                BlindRelayTransportErrorKind::Connect,
                "blind_relay_request_connect",
                BlindRelayTransportFailureKind::Connect,
            ),
            (
                BlindRelayTransportErrorKind::Request,
                "blind_relay_request_request",
                BlindRelayTransportFailureKind::Request,
            ),
            (
                BlindRelayTransportErrorKind::Body,
                "blind_relay_request_body",
                BlindRelayTransportFailureKind::Other,
            ),
            (
                BlindRelayTransportErrorKind::HttpStatus(Some(503)),
                "blind_relay_request_http_503",
                BlindRelayTransportFailureKind::Other,
            ),
        ];

        for (kind, expected_bucket, expected_retry_kind) in cases {
            let failure = BlindRelayTransportFailure { kind };
            assert_eq!(failure.reason_bucket(), expected_bucket);
            assert_eq!(failure.retry_kind(), expected_retry_kind);
        }
    }
}
