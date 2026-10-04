// ============================================
// File: crates/aeronyx-server/src/api/reverse_onion.rs
// ============================================
//! Unregistered adjacent-hop reverse-onion Claim/Result adapter.
//!
//! [REVERSE-ONION-API-ADAPTER 2026-10-04 by Codex] This module is deliberately
//! not registered by this change. Router/auth/state ownership remains with the
//! integration owner. The adapter accepts only canonical core frames and uses
//! the durable queue boundary; it never exposes source completion or route
//! payload data.
//!
//! The immediate recipient is pinned by authenticated/configured state. A
//! caller cannot select a recipient, route deadline, queue key, or envelope
//! commitment. Result recovery uses the queue's read-only authenticated
//! context lookup and calls the core Result verifier before any CAS write.
//!
//! Last Modified: v0.1.0-ReverseOnionApiAdapter - Unregistered Claim/Result
//! boundary with bounded blocking and deterministic coarse replies.

use std::sync::Arc;

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::chat::decode_blind_relay_envelope;
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, ReverseOnionKindV1, MAX_REVERSE_ONION_FRAME_BYTES,
};
use axum::body::Bytes;
use axum::http::StatusCode;
use rand::{rngs::OsRng, RngCore};
use tokio::sync::Semaphore;

use crate::services::reverse_onion_queue::{
    ReverseOnionQueueCompletion, ReverseOnionQueueError, ReverseOnionQueueIssue,
    ReverseOnionQueueLeaseMaterial, ReverseOnionQueueResultContext,
};
use crate::services::reverse_onion_queue_db::{
    ReverseOnionQueueDb, ReverseOnionQueueDbError,
};

/// The raw core Claim/Result frame bound is also the HTTP body bound. Axum
/// route registration must install the same DefaultBodyLimit before this
/// adapter is exposed; this check remains mandatory at the function boundary.
pub(crate) const MAX_REVERSE_ONION_API_BODY_BYTES: usize =
    MAX_REVERSE_ONION_FRAME_BYTES;

/// Coarse body/status response. Success bodies are exact canonical core frame
/// bytes; every failure body is empty and carries no IDs, paths, or payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReverseOnionApiReply {
    status: StatusCode,
    body: Bytes,
}

impl ReverseOnionApiReply {
    pub(crate) fn status(&self) -> StatusCode {
        self.status
    }

    pub(crate) fn body(&self) -> &[u8] {
        self.body.as_ref()
    }

    fn frame(status: StatusCode, frame: Vec<u8>) -> Self {
        Self {
            status,
            body: Bytes::from(frame),
        }
    }

    fn empty(status: StatusCode) -> Self {
        Self {
            status,
            body: Bytes::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReverseOnionApiConfigError {
    Invalid,
}

/// One configured/authenticated recipient view of the queue.
///
/// A separate instance is required for each allowlisted recipient identity;
/// no request field can replace `recipient` after construction.
pub(crate) struct ReverseOnionApi {
    queue: Arc<ReverseOnionQueueDb>,
    relay: Arc<IdentityKeyPair>,
    recipient: [u8; 32],
    blocking: Arc<Semaphore>,
}

impl ReverseOnionApi {
    /// Defaults are intentionally absent: router/state wiring must opt in with
    /// an explicitly opened queue and an authenticated recipient allowlist.
    pub(crate) fn new(
        queue: Arc<ReverseOnionQueueDb>,
        relay: Arc<IdentityKeyPair>,
        recipient: [u8; 32],
        max_in_flight: usize,
    ) -> Result<Self, ReverseOnionApiConfigError> {
        validate_recipient(relay.public_key_bytes(), recipient, max_in_flight)?;
        Ok(Self {
            queue,
            relay,
            recipient,
            blocking: Arc::new(Semaphore::new(max_in_flight)),
        })
    }

    /// Handles one raw canonical Claim frame. The authenticated/configured P
    /// is the only accepted immediate recipient; the body cannot select P.
    pub(crate) async fn handle_claim(&self, body: Bytes, now: u64) -> ReverseOnionApiReply {
        if body.is_empty() || body.len() > MAX_REVERSE_ONION_API_BODY_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let permit = match self.blocking.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => return ReverseOnionApiReply::empty(StatusCode::TOO_MANY_REQUESTS),
        };
        let queue = Arc::clone(&self.queue);
        let relay = Arc::clone(&self.relay);
        let recipient = self.recipient;
        let body = body.to_vec();
        match tokio::task::spawn_blocking(move || {
            // [REVERSE-ONION-API-CANCELLATION 2026-10-04 by Codex] Keep the
            // owned permit inside the blocking closure. Dropping a cancelled
            // HTTP JoinHandle must not advertise capacity while SQLite work
            // continues.
            let _permit = permit;
            handle_claim_blocking(&queue, &relay, recipient, &body, now)
        })
        .await
        {
            Ok(reply) => reply,
            Err(_) => ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
        }
    }

    /// Handles one raw canonical Result frame. The Result signer P is checked
    /// against the pinned recipient before lookup; the queue supplies the
    /// stored Claim/Lease context used by the core verifier.
    pub(crate) async fn handle_result(&self, body: Bytes, now: u64) -> ReverseOnionApiReply {
        if body.is_empty() || body.len() > MAX_REVERSE_ONION_API_BODY_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let permit = match self.blocking.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => return ReverseOnionApiReply::empty(StatusCode::TOO_MANY_REQUESTS),
        };
        let queue = Arc::clone(&self.queue);
        let relay = Arc::clone(&self.relay);
        let recipient = self.recipient;
        let body = body.to_vec();
        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            handle_result_blocking(&queue, &relay, recipient, &body, now)
        })
        .await
        {
            Ok(reply) => reply,
            Err(_) => ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
        }
    }
}

fn handle_claim_blocking(
    queue: &ReverseOnionQueueDb,
    relay: &Arc<IdentityKeyPair>,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
) -> ReverseOnionApiReply {
    let claim = match ReverseOnionFrameV1::decode(body, now) {
        Ok(claim) => claim,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    if claim.kind() != ReverseOnionKindV1::Claim
        || claim.relay() != relay.public_key_bytes()
        || claim.immediate_recipient() != recipient
        || relay.public_key_bytes() == recipient
        || claim.encode().as_slice() != body
    {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    if claim.verify_claim(relay.public_key_bytes(), recipient, now).is_err() {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    let claim_id = claim.claim_id();
    let claim_commitment = claim.commitment();
    let claim_frame = body.to_vec();
    let relay_for_verify = Arc::clone(relay);
    let relay_for_builder = Arc::clone(relay);
    let issue = queue.issue_lease(
        recipient,
        claim_id,
        claim_commitment,
        claim_frame,
        now,
        move |frame| {
            let claim = ReverseOnionFrameV1::decode(frame, now)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            claim
                .verify_claim(relay_for_verify.public_key_bytes(), recipient, now)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            Ok(claim.commitment())
        },
        {
            let relay = relay_for_builder;
            move |stored, claim_bytes| {
                let claim = ReverseOnionFrameV1::decode(claim_bytes, now)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let envelope = decode_blind_relay_envelope(stored.envelope())
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let mut lease_id = [0u8; 16];
                OsRng.fill_bytes(&mut lease_id);
                let lease = ReverseOnionFrameV1::lease(
                    &claim,
                    &envelope,
                    lease_id,
                    stored.route_deadline(),
                    now,
                    &relay,
                )
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
                ReverseOnionQueueLeaseMaterial::new(
                    lease_id,
                    lease.commitment(),
                    lease.encode(),
                    lease.expires_at(),
                    lease
                        .replay_evidence_deadline()
                        .map_err(|_| ReverseOnionQueueError::Rejected)?,
                )
            }
        },
    );
    match issue {
        Ok(ReverseOnionQueueIssue::Issued(lease))
        | Ok(ReverseOnionQueueIssue::Existing(lease)) => {
            ReverseOnionApiReply::frame(StatusCode::OK, lease.frame().to_vec())
        }
        Ok(ReverseOnionQueueIssue::Result(result)) => {
            ReverseOnionApiReply::frame(StatusCode::OK, result.frame().to_vec())
        }
        Ok(ReverseOnionQueueIssue::NoWork) => {
            ReverseOnionApiReply::empty(StatusCode::NO_CONTENT)
        }
        Ok(ReverseOnionQueueIssue::Ambiguous) => {
            ReverseOnionApiReply::empty(StatusCode::CONFLICT)
        }
        Err(error) => map_queue_error(error),
    }
}

fn handle_result_blocking(
    queue: &ReverseOnionQueueDb,
    relay: &IdentityKeyPair,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
) -> ReverseOnionApiReply {
    let result = match ReverseOnionFrameV1::decode_for_recovery(body) {
        Ok(result) => result,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    if result.kind() != ReverseOnionKindV1::Result
        || result.relay() != relay.public_key_bytes()
        || result.immediate_recipient() != recipient
        || relay.public_key_bytes() == recipient
        || result.claim_id() == [0; 16]
        || result.lease_id() == [0; 16]
        || result.route_id() == [0; 16]
        || result.encode().as_slice() != body
    {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    let context = match queue.lookup_result_context(
        recipient,
        result.claim_id(),
        result.lease_id(),
        result.route_id(),
        now,
    ) {
        Ok(context) => context,
        Err(error) => return map_queue_db_error(error),
    };
    match context {
        ReverseOnionQueueResultContext::Armed(lease) => {
            let candidate = body.to_vec();
            let completion = queue.complete(&lease, body, now, |stored, result_bytes| {
                if result_bytes != candidate.as_slice() {
                    return Err(ReverseOnionQueueError::Conflict);
                }
                let claim = ReverseOnionFrameV1::decode_for_recovery(stored.claim_frame())
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let persisted_lease = ReverseOnionFrameV1::decode_for_recovery(stored.lease_frame())
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let result = ReverseOnionFrameV1::decode_for_recovery(result_bytes)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                result
                    .verify_result(&claim, &persisted_lease, stored.route_deadline(), now)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                Ok(result.commitment())
            });
            match completion {
                Ok(ReverseOnionQueueCompletion::Stored)
                | Ok(ReverseOnionQueueCompletion::AlreadyComplete) => {
                    ReverseOnionApiReply::frame(StatusCode::OK, body.to_vec())
                }
                Err(error) => map_queue_db_error(error),
            }
        }
        ReverseOnionQueueResultContext::Result(stored) => {
            let stored_frame = match ReverseOnionFrameV1::decode_for_recovery(stored.frame()) {
                Ok(frame) => frame,
                Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
            };
            if stored.frame() != body
                || stored_frame.kind() != ReverseOnionKindV1::Result
                || stored_frame.relay() != relay.public_key_bytes()
                || stored_frame.immediate_recipient() != recipient
                || stored_frame.claim_id() != result.claim_id()
                || stored_frame.lease_id() != result.lease_id()
                || stored_frame.route_id() != result.route_id()
                || stored_frame.commitment() != stored.commitment()
            {
                return ReverseOnionApiReply::empty(StatusCode::CONFLICT);
            }
            ReverseOnionApiReply::frame(StatusCode::OK, stored.frame().to_vec())
        }
        ReverseOnionQueueResultContext::NoWork => {
            ReverseOnionApiReply::empty(StatusCode::NO_CONTENT)
        }
        ReverseOnionQueueResultContext::Ambiguous => {
            ReverseOnionApiReply::empty(StatusCode::CONFLICT)
        }
    }
}

fn map_queue_error(error: ReverseOnionQueueError) -> ReverseOnionApiReply {
    map_queue_db_error(error.into())
}

fn map_queue_db_error(error: ReverseOnionQueueDbError) -> ReverseOnionApiReply {
    let status = match error {
        ReverseOnionQueueDbError::NoWork => StatusCode::NO_CONTENT,
        ReverseOnionQueueDbError::Conflict | ReverseOnionQueueDbError::Ambiguous => {
            StatusCode::CONFLICT
        }
        ReverseOnionQueueDbError::Busy => StatusCode::TOO_MANY_REQUESTS,
        ReverseOnionQueueDbError::Rejected => StatusCode::BAD_REQUEST,
        ReverseOnionQueueDbError::Capacity
        | ReverseOnionQueueDbError::Corrupt
        | ReverseOnionQueueDbError::MigrationRequired
        | ReverseOnionQueueDbError::Unavailable
        | ReverseOnionQueueDbError::LeaseLost
        | ReverseOnionQueueDbError::AlreadyComplete => StatusCode::SERVICE_UNAVAILABLE,
    };
    ReverseOnionApiReply::empty(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_rejects_self_recipient_and_zero_concurrency() {
        let relay = Arc::new(IdentityKeyPair::from_bytes(&[7; 32]).unwrap());
        assert_eq!(
            validate_recipient(relay.public_key_bytes(), relay.public_key_bytes(), 1),
            Err(ReverseOnionApiConfigError::Invalid)
        );
        assert_eq!(
            validate_recipient(relay.public_key_bytes(), [0; 32], 1),
            Err(ReverseOnionApiConfigError::Invalid)
        );
        assert_eq!(
            validate_recipient(relay.public_key_bytes(), [8; 32], 0),
            Err(ReverseOnionApiConfigError::Invalid)
        );
    }

    #[test]
    fn replies_are_empty_for_coarse_errors_and_exact_for_frames() {
        assert!(ReverseOnionApiReply::empty(StatusCode::NO_CONTENT)
            .body()
            .is_empty());
        assert_eq!(
            ReverseOnionApiReply::frame(StatusCode::OK, vec![1, 2, 3]).body(),
            &[1, 2, 3]
        );
        assert_eq!(
            map_queue_error(ReverseOnionQueueError::Conflict).status(),
            StatusCode::CONFLICT
        );
    }
}

fn validate_recipient(
    relay: [u8; 32],
    recipient: [u8; 32],
    max_in_flight: usize,
) -> Result<(), ReverseOnionApiConfigError> {
    if max_in_flight == 0
        || relay == recipient
        || IdentityPublicKey::from_bytes(&recipient).is_err()
    {
        return Err(ReverseOnionApiConfigError::Invalid);
    }
    Ok(())
}
