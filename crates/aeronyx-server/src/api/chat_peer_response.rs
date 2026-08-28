// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer_response.rs
// ============================================================================
//! # Blind Relay Response Policy
//!
//! ## Creation Reason
//! Separates deterministic peer-response interpretation from asynchronous
//! forwarding, sleeping, route-health persistence, and aggregate telemetry.
//!
//! ## Main Functionality
//! - Defines the replaceable [`BlindRelayResponsePolicy`] capability.
//! - Validates success ACK shape, freshness, signatures, and signer binding.
//! - Validates negotiated immediate-hop failure receipts without widening blame.
//! - Composes transport outcomes with the existing retry policy.
//! - Returns typed decisions for the HTTP orchestrator to execute and observe.
//!
//! ## Dependencies
//! - Reads blind-relay request/response wire types from `chat_peer.rs`.
//! - Reads transport outcomes from `chat_peer_transport.rs`.
//! - Delegates retry timing and coarse failure classification to
//!   `chat_peer_retry.rs`.
//!
//! ## Main Logical Flow
//! 1. Interpret a bounded transport outcome under one immutable attempt context.
//! 2. Validate successful ACK state and optional delivery evidence.
//! 3. Authenticate a peer-declared failure when the negotiated receipt requires it.
//! 4. Delegate bare status or request failure to the payload-blind retry policy.
//! 5. Return a side-effect-free decision to the forwarding orchestrator.
//!
//! ## Important Note for Next Developer
//! - Never add `PeerStore` writes, sleeps, logs, clocks, URLs, or network I/O here.
//! - Never expose route ids, node ids, payloads, users, wallets, or endpoints in
//!   reason buckets; diagnostics remain fixed protocol constants only.
//! - A valid failure receipt authenticates only the immediate responder and must
//!   never assign blame to a deeper onion hop.
//! - Legacy receipt-less success remains compatible until feature negotiation
//!   explicitly changes the wire contract.
//!
//! ## Last Modified
//! v2.8.38-OnionReplyValidation - Validate fixed-size opaque reply envelopes.
//! v2.8.37-ChatPeerResponseDomain - Initial trait-based response policy.
//! ============================================================================

use std::time::Duration;

use aeronyx_core::protocol::{decode_onion_sealed_response, ONION_REPLY_RESPONSE_SIZE_CLASSES};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use reqwest::StatusCode;

use super::chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse};
use super::chat_peer_retry::{
    BlindRelayDownstreamFailure, BlindRelayRetryAction, BlindRelayRetryContext,
    BlindRelayRetryPolicy,
};
use super::chat_peer_transport::{BlindRelayTransportFailure, BlindRelayTransportOutcome};
use crate::api::BoundedHttpResponseError;

/// Terminal delivery receipts are short-lived acknowledgements, not durable tokens.
pub(super) const BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS: u64 = 120;
/// Delivery receipt future skew is intentionally tighter than relay-frame skew.
pub(super) const BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS: u64 = 30;
/// Signed failure ACKs use the same short replay horizon as success receipts.
pub(super) const BLIND_RELAY_FAILURE_RECEIPT_MAX_AGE_SECS: u64 = 120;
/// Failure receipts tolerate only bounded peer clock skew.
pub(super) const BLIND_RELAY_FAILURE_RECEIPT_MAX_FUTURE_SKEW_SECS: u64 = 30;

/// Origin of one retry or exhaustion decision, without retaining a URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayResponseSource {
    HttpStatus(StatusCode),
    Transport,
}

/// Protocol surface that made an otherwise bounded response invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindRelayInvalidResponseKind {
    SuccessAck,
    DeliveryReceipt,
    OpaqueTerminalResponse,
    FailureReceipt,
}

/// Side-effect-free decision for one outbound relay attempt.
#[derive(Debug)]
pub(super) enum BlindRelayResponseDecision {
    Accepted(Box<PeerBlindRelayResponse>),
    PeerDeclaredFailure {
        failure: BlindRelayDownstreamFailure,
        status: StatusCode,
        receipt_authenticated: bool,
    },
    RetryAfter {
        delay: Duration,
        reason: String,
        source: BlindRelayResponseSource,
    },
    Reject(BlindRelayDownstreamFailure),
    InvalidResponse {
        kind: BlindRelayInvalidResponseKind,
        diagnostic: &'static str,
        health_reason: &'static str,
        counts_as_retry_exhaustion: bool,
    },
    Exhausted {
        reason: String,
        source: BlindRelayResponseSource,
    },
}

/// Immutable, payload-blind inputs needed to interpret one response.
pub(super) struct BlindRelayResponseContext<'a> {
    pub(super) request: &'a PeerBlindRelayRequest,
    pub(super) next_hop: [u8; 32],
    pub(super) observed_at: u64,
    pub(super) failure_receipt_required: bool,
    pub(super) retry_context: BlindRelayRetryContext,
    pub(super) retry_policy: &'a dyn BlindRelayRetryPolicy,
}

/// Replaceable deterministic response interpretation capability.
pub(super) trait BlindRelayResponsePolicy: Send + Sync {
    fn evaluate(
        &self,
        outcome: BlindRelayTransportOutcome,
        context: BlindRelayResponseContext<'_>,
    ) -> BlindRelayResponseDecision;
}

/// Production response policy preserving the established relay contract.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct BlindRelayResponseDomain;

impl BlindRelayResponsePolicy for BlindRelayResponseDomain {
    fn evaluate(
        &self,
        outcome: BlindRelayTransportOutcome,
        context: BlindRelayResponseContext<'_>,
    ) -> BlindRelayResponseDecision {
        match outcome {
            BlindRelayTransportOutcome::SuccessStatus { response } => {
                evaluate_success_status(response, &context)
            }
            BlindRelayTransportOutcome::RejectedStatus {
                status,
                declared_response,
            } => evaluate_rejected_status(status, declared_response.as_deref(), &context),
            BlindRelayTransportOutcome::RequestFailed(failure) => {
                evaluate_transport_failure(failure, &context)
            }
        }
    }
}

fn evaluate_success_status(
    response: Result<Box<PeerBlindRelayResponse>, BoundedHttpResponseError>,
    context: &BlindRelayResponseContext<'_>,
) -> BlindRelayResponseDecision {
    match response {
        Ok(response) if response.accepted => {
            if let Err(diagnostic) = validate_downstream_delivery_receipt(
                &response,
                &context.request.envelope.route_id,
                &context.next_hop,
                context.observed_at,
            ) {
                return BlindRelayResponseDecision::InvalidResponse {
                    kind: BlindRelayInvalidResponseKind::DeliveryReceipt,
                    diagnostic,
                    health_reason: "delivery_receipt_invalid",
                    counts_as_retry_exhaustion: false,
                };
            }
            if let Err(diagnostic) = validate_opaque_terminal_response(&response) {
                return BlindRelayResponseDecision::InvalidResponse {
                    kind: BlindRelayInvalidResponseKind::OpaqueTerminalResponse,
                    diagnostic,
                    health_reason: "opaque_terminal_response_invalid",
                    counts_as_retry_exhaustion: false,
                };
            }
            BlindRelayResponseDecision::Accepted(response)
        }
        Ok(_) => BlindRelayResponseDecision::InvalidResponse {
            kind: BlindRelayInvalidResponseKind::SuccessAck,
            diagnostic: "ack_not_accepted",
            health_reason: "forward_failed",
            counts_as_retry_exhaustion: true,
        },
        Err(error) => BlindRelayResponseDecision::InvalidResponse {
            kind: BlindRelayInvalidResponseKind::SuccessAck,
            diagnostic: error.as_str(),
            health_reason: "forward_failed",
            counts_as_retry_exhaustion: true,
        },
    }
}

/// Validates only the public, payload-blind envelope of an inline response.
///
/// The source remains responsible for decryption and terminal-signature
/// verification. Immediate hops enforce the fixed v1 size class so a peer
/// cannot smuggle a variable or oversized response through the ACK channel.
fn validate_opaque_terminal_response(ack: &PeerBlindRelayResponse) -> Result<(), &'static str> {
    let Some(encoded) = ack.opaque_terminal_response_b64.as_deref() else {
        return Ok(());
    };
    if ack.delivery_receipt.is_none() {
        return Err("opaque_response_receipt_missing");
    }
    let bytes = BASE64
        .decode(encoded)
        .map_err(|_| "opaque_response_base64_invalid")?;
    let response =
        decode_onion_sealed_response(&bytes).map_err(|_| "opaque_response_envelope_invalid")?;
    if response
        .validate()
        .map_err(|_| "opaque_response_envelope_invalid")?
        != ONION_REPLY_RESPONSE_SIZE_CLASSES[0]
    {
        return Err("opaque_response_size_class_invalid");
    }
    Ok(())
}

fn evaluate_rejected_status(
    status: StatusCode,
    declared_response: Option<&PeerBlindRelayResponse>,
    context: &BlindRelayResponseContext<'_>,
) -> BlindRelayResponseDecision {
    if let (Some(failure), Some(response)) = (
        context
            .retry_policy
            .classify_declared_failure(status, declared_response),
        declared_response,
    ) {
        return match validate_downstream_failure_receipt(
            response,
            context.request,
            &context.next_hop,
            context.observed_at,
            context.failure_receipt_required,
        ) {
            Ok(receipt_authenticated) => BlindRelayResponseDecision::PeerDeclaredFailure {
                failure,
                status,
                receipt_authenticated,
            },
            Err(diagnostic) => BlindRelayResponseDecision::InvalidResponse {
                kind: BlindRelayInvalidResponseKind::FailureReceipt,
                diagnostic,
                health_reason: if diagnostic == "failure_receipt_required" {
                    "failure_receipt_downgrade"
                } else {
                    "failure_receipt_invalid"
                },
                counts_as_retry_exhaustion: false,
            },
        };
    }

    let reason = format!("http_{}", status.as_u16());
    match context
        .retry_policy
        .status_action(status, context.retry_context)
    {
        BlindRelayRetryAction::RetryAfter(delay) => BlindRelayResponseDecision::RetryAfter {
            delay,
            reason,
            source: BlindRelayResponseSource::HttpStatus(status),
        },
        BlindRelayRetryAction::Reject(failure) => BlindRelayResponseDecision::Reject(failure),
        BlindRelayRetryAction::Exhausted => BlindRelayResponseDecision::Exhausted {
            reason,
            source: BlindRelayResponseSource::HttpStatus(status),
        },
    }
}

fn evaluate_transport_failure(
    failure: BlindRelayTransportFailure,
    context: &BlindRelayResponseContext<'_>,
) -> BlindRelayResponseDecision {
    let reason = failure.reason_bucket();
    match context
        .retry_policy
        .transport_action(failure.retry_kind(), context.retry_context)
    {
        BlindRelayRetryAction::RetryAfter(delay) => BlindRelayResponseDecision::RetryAfter {
            delay,
            reason,
            source: BlindRelayResponseSource::Transport,
        },
        BlindRelayRetryAction::Reject(failure) => BlindRelayResponseDecision::Reject(failure),
        BlindRelayRetryAction::Exhausted => BlindRelayResponseDecision::Exhausted {
            reason,
            source: BlindRelayResponseSource::Transport,
        },
    }
}

/// Validates the downstream success state and receipt surface visible at this hop.
///
/// [MULTIHOP-RECEIPT-VALIDATION 2026-08-01 by Codex] A direct terminal ACK must
/// be signed by `immediate_next_hop`. A forwarded ACK may carry a receipt from a
/// deeper terminal, so this hop validates route, freshness, disposition, and
/// signature without requiring the terminal to equal the immediate next hop.
pub(super) fn validate_downstream_delivery_receipt(
    ack: &PeerBlindRelayResponse,
    route_id: &[u8; 16],
    immediate_next_hop: &[u8; 32],
    observed_at: u64,
) -> Result<(), &'static str> {
    if !ack.accepted {
        return Err("ack_not_accepted");
    }
    if ack.terminal == ack.forwarded {
        return Err("invalid_ack_shape");
    }
    if ack.failure_receipt.is_some() {
        return Err("unexpected_failure_receipt");
    }

    let Some(receipt) = ack.delivery_receipt.as_ref() else {
        return Ok(());
    };
    if &receipt.route_id != route_id {
        return Err("receipt_route_mismatch");
    }
    if receipt.delivered_at
        > observed_at.saturating_add(BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS)
    {
        return Err("receipt_timestamp_in_future");
    }
    if observed_at.saturating_sub(receipt.delivered_at) > BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS
    {
        return Err("receipt_timestamp_expired");
    }
    receipt
        .verify_signature()
        .map_err(|_| "receipt_signature_invalid")?;
    if ack.terminal && &receipt.terminal_node_id != immediate_next_hop {
        return Err("terminal_receipt_signer_mismatch");
    }
    Ok(())
}

/// Verifies an optional immediate-hop failure receipt without widening blame.
pub(super) fn validate_downstream_failure_receipt(
    ack: &PeerBlindRelayResponse,
    request: &PeerBlindRelayRequest,
    immediate_next_hop: &[u8; 32],
    observed_at: u64,
    receipt_required: bool,
) -> Result<bool, &'static str> {
    if ack.opaque_terminal_response_b64.is_some() {
        return Err("unexpected_opaque_terminal_response");
    }
    let Some(receipt) = ack.failure_receipt.as_ref() else {
        return if receipt_required {
            Err("failure_receipt_required")
        } else {
            Ok(false)
        };
    };
    let reason = ack.reason.as_deref().ok_or("failure_reason_missing")?;
    if receipt.failed_at
        > observed_at.saturating_add(BLIND_RELAY_FAILURE_RECEIPT_MAX_FUTURE_SKEW_SECS)
    {
        return Err("failure_receipt_timestamp_in_future");
    }
    if observed_at.saturating_sub(receipt.failed_at) > BLIND_RELAY_FAILURE_RECEIPT_MAX_AGE_SECS {
        return Err("failure_receipt_timestamp_expired");
    }
    receipt
        .verify_expected(&request.envelope, reason, immediate_next_hop)
        .map_err(|_| "failure_receipt_binding_invalid")?;
    Ok(true)
}
