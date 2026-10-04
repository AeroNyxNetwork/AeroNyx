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
//! [REVERSE-ONION-CLAIM-RETRY 2026-10-04 by Codex] Historical exact Claim
//! bytes are admitted through queue idempotence after signature/R/P checks;
//! only a genuinely new Claim is subjected to fresh 30-second validation.
//!
//! [REVERSE-ONION-SOURCE-QUERY 2026-10-04 by Codex] The source evidence
//! handler is deliberately unmounted. It verifies a signed fixed-size query,
//! reads only the exact durable source-bound snapshot, and never mutates the
//! queue or emits unsigned absence claims.
//!
//! Last Modified: v0.1.0-ReverseOnionApiAdapter - Unregistered Claim/Result
//! boundary with bounded blocking, source evidence, and deterministic coarse
//! replies.

use std::sync::Arc;

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::chat::{
    decode_blind_relay_envelope, encode_blind_relay_envelope, BlindRelayEnvelope,
};
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, ReverseOnionKindV1, ReverseOnionSourceEvidenceV1,
    ReverseOnionSourceQueryV1, SourceEvidencePartV1, SourceEvidenceStateV1,
    MAX_REVERSE_ONION_FRAME_BYTES,
    MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES, REVERSE_ONION_SOURCE_QUERY_BYTES,
};
use aeronyx_core::protocol::onion_reply::{
    encode_onion_sealed_response, seal_onion_reply, OnionReplySession,
    ONION_REPLY_RESPONSE_SIZE_CLASSES,
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

    /// Handles one canonical source query. This method is intentionally not
    /// mounted by any router: it is a bounded read-only bridge for the source
    /// evidence protocol, not a new public readiness signal.
    pub(crate) async fn handle_source_query(
        &self,
        body: Bytes,
        now: u64,
    ) -> ReverseOnionApiReply {
        if body.len() != REVERSE_ONION_SOURCE_QUERY_BYTES {
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
            handle_source_query_blocking(&queue, &relay, recipient, &body, now)
        })
        .await
        {
            Ok(reply) => reply,
            Err(_) => ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
        }
    }
}

fn handle_source_query_blocking(
    queue: &ReverseOnionQueueDb,
    relay: &IdentityKeyPair,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
) -> ReverseOnionApiReply {
    let query = match ReverseOnionSourceQueryV1::decode(body) {
        Ok(query) => query,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    if body.len() != REVERSE_ONION_SOURCE_QUERY_BYTES
        || query.relay() != relay.public_key_bytes()
        || query.route_id() == [0; 16]
        || query.original_request_commitment() == [0; 32]
    {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    let authority = match query.verify_binding(
        query.source(),
        relay.public_key_bytes(),
        query.route_id(),
        query.original_request_commitment(),
        now,
    ) {
        Ok(authority) => authority,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    let snapshot = match queue.lookup_source(
        query.source(),
        query.route_id(),
        query.original_request_commitment(),
        now,
    ) {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) | Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    if snapshot.immediate_recipient() != recipient {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    let (Some(claim_bytes), Some(lease_bytes), Some(result_bytes)) = (
        snapshot.claim_frame(),
        snapshot.lease_frame(),
        snapshot.result_frame(),
    ) else {
        let evidence = match ReverseOnionSourceEvidenceV1::pending(
            &authority,
            now,
            query.expires_at(),
            relay,
        ) {
            Ok(evidence) => evidence,
            Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
        };
        let encoded = evidence.encode();
        if encoded.len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE);
        }
        return ReverseOnionApiReply::frame(StatusCode::OK, encoded);
    };
    let claim = match ReverseOnionFrameV1::decode_for_recovery(claim_bytes) {
        Ok(frame) => frame,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
    };
    let lease = match ReverseOnionFrameV1::decode_for_recovery(lease_bytes) {
        Ok(frame) => frame,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
    };
    let result = match ReverseOnionFrameV1::decode_for_recovery(result_bytes) {
        Ok(frame) => frame,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
    };
    let evidence = match ReverseOnionSourceEvidenceV1::available(
        &authority,
        &claim,
        &lease,
        &result,
        snapshot.route_deadline(),
        now,
        query.expires_at(),
        relay,
    ) {
        Ok(evidence) => evidence,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
    };
    let encoded = evidence.encode();
    if encoded.len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
        return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE);
    }
    ReverseOnionApiReply::frame(StatusCode::OK, encoded)
}

fn handle_claim_blocking(
    queue: &ReverseOnionQueueDb,
    relay: &Arc<IdentityKeyPair>,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
) -> ReverseOnionApiReply {
    let claim = match ReverseOnionFrameV1::decode_for_recovery(body) {
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
    use crate::services::reverse_onion_queue::{
        ReverseOnionQueueItem, ReverseOnionQueueLeaseMaterial, ReverseOnionQueueLimits,
        ReverseOnionQueueIssue,
    };
    use crate::services::reverse_onion_queue_db::{ReverseOnionQueueDbConfig};
    use tempfile::TempDir;

    const NOW: u64 = 1_800_000_000;

    #[cfg(unix)]
    fn fixture() -> (
        TempDir,
        Arc<ReverseOnionQueueDb>,
        ReverseOnionApi,
        Arc<IdentityKeyPair>,
        Arc<IdentityKeyPair>,
        Arc<IdentityKeyPair>,
    ) {
        let directory = tempfile::Builder::new()
            .prefix("r6-reverse-onion-api-")
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp")
            .expect("fixture directory");
        let limits = ReverseOnionQueueLimits::new(4, 8 * 1024 * 1024, 4, 60, 120)
            .expect("queue limits");
        let config = ReverseOnionQueueDbConfig::new(
            directory.path().join("queue.sqlite"),
            16 * 1024 * 1024,
            limits,
        )
        .expect("queue config");
        let queue = Arc::new(ReverseOnionQueueDb::open(config, NOW).expect("queue open"));
        let relay = Arc::new(IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap());
        let recipient = Arc::new(IdentityKeyPair::from_bytes(&[0x75; 32]).unwrap());
        let source = Arc::new(IdentityKeyPair::from_bytes(&[0x76; 32]).unwrap());
        let api = ReverseOnionApi::new(
            Arc::clone(&queue),
            Arc::clone(&relay),
            recipient.public_key_bytes(),
            1,
        )
        .expect("api");
        (directory, queue, api, relay, recipient, source)
    }

    #[cfg(unix)]
    fn query_bytes(
        source: &IdentityKeyPair,
        relay: &IdentityKeyPair,
        route: [u8; 16],
        original_request: [u8; 32],
        issued_at: u64,
        expires_at: u64,
    ) -> Vec<u8> {
        ReverseOnionSourceQueryV1::sign(
            source,
            relay.public_key_bytes(),
            route,
            original_request,
            SourceEvidencePartV1::Claim,
            [0x76; 32],
            issued_at,
            expires_at,
        )
        .unwrap()
        .encode()
    }

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

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_rejects_bounds_wrong_relay_and_stale_query() {
        let (_directory, _queue, api, relay, _recipient, source) = fixture();
        assert_eq!(
            api.handle_source_query(Bytes::from(vec![0; REVERSE_ONION_SOURCE_QUERY_BYTES - 1]), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
        let canonical = query_bytes(&source, &relay, [1; 16], [2; 32], NOW, NOW + 10);
        let mut trailing = canonical.clone();
        trailing.push(0);
        assert_eq!(
            api.handle_source_query(Bytes::from(trailing), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
        let wrong_relay = IdentityKeyPair::from_bytes(&[0x77; 32]).unwrap();
        let wrong = query_bytes(&source, &wrong_relay, [1; 16], [2; 32], NOW, NOW + 10);
        assert_eq!(
            api.handle_source_query(Bytes::from(wrong), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
        let stale = query_bytes(&source, &relay, [1; 16], [2; 32], NOW - 20, NOW - 1);
        assert_eq!(
            api.handle_source_query(Bytes::from(stale), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_releases_no_permit_for_preflight_rejection() {
        let (_directory, _queue, api, _relay, _recipient, _source) = fixture();
        for _ in 0..2 {
            assert_eq!(
                api.handle_source_query(Bytes::from_static(b"short"), NOW)
                    .await
                    .status(),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_pending_is_signed_only_for_partial_durable_snapshot() {
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let item = ReverseOnionQueueItem::new(
            [0x81; 32],
            [1; 16],
            [2; 32],
            source.public_key_bytes(),
            [3; 32],
            api.recipient,
            [4; 32],
            vec![5; 3],
            NOW + 60,
        )
        .unwrap();
        queue.enqueue(&item, NOW).unwrap();
        let query = query_bytes(&source, &relay, [1; 16], [2; 32], NOW, NOW + 20);
        let reply = api
            .handle_source_query(Bytes::from(query.clone()), NOW + 1)
            .await;
        assert_eq!(reply.status(), StatusCode::OK);
        let evidence = ReverseOnionSourceEvidenceV1::decode(reply.body()).unwrap();
        let query = ReverseOnionSourceQueryV1::decode(&query).unwrap();
        assert_eq!(evidence.state(), SourceEvidenceStateV1::Pending);
        evidence.verify_for_query(&query, NOW + 1).unwrap();
        let other_recipient = IdentityKeyPair::from_bytes(&[0x78; 32]).unwrap();
        let other_item = ReverseOnionQueueItem::new(
            [0x89; 32],
            [7; 16],
            [8; 32],
            source.public_key_bytes(),
            [9; 32],
            other_recipient.public_key_bytes(),
            [10; 32],
            vec![11; 3],
            NOW + 60,
        )
        .unwrap();
        queue.enqueue(&other_item, NOW).unwrap();
        let wrong_target = query_bytes(&source, &relay, [7; 16], [8; 32], NOW, NOW + 20);
        assert_eq!(
            api.handle_source_query(Bytes::from(wrong_target), NOW + 1)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_available_requires_complete_verified_chain() {
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let envelope = BlindRelayEnvelope {
            route_id: [11; 16],
            next_hop: api.recipient,
            ttl: 1,
            encrypted_blob: vec![12; 4],
            timestamp: NOW,
            signature: [0; 64],
        }
        .sign_with(&relay);
        let envelope_bytes = encode_blind_relay_envelope(&envelope).unwrap();
        let item = ReverseOnionQueueItem::new(
            [13; 32],
            [11; 16],
            [14; 32],
            source.public_key_bytes(),
            [15; 32],
            api.recipient,
            [16; 32],
            envelope_bytes,
            NOW + 60,
        )
        .unwrap();
        queue.enqueue(&item, NOW).unwrap();
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(),
            [17; 16],
            NOW,
            NOW + 30,
            &recipient,
        )
        .unwrap();
        let lease = ReverseOnionFrameV1::lease(
            &claim,
            &envelope,
            [18; 16],
            NOW + 60,
            NOW,
            &relay,
        )
        .unwrap();
        let claim_commitment = claim.commitment();
        let lease_id = lease.lease_id();
        let lease_commitment = lease.commitment();
        let lease_frame = lease.encode();
        let lease_expiry = lease.expires_at();
        let replay_deadline = lease.replay_evidence_deadline().unwrap();
        let issue = queue
            .issue_lease(
                api.recipient,
                claim.claim_id(),
                claim_commitment,
                claim.encode(),
                NOW,
                move |_| Ok(claim_commitment),
                move |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        lease_id,
                        lease_commitment,
                        lease_frame,
                        lease_expiry,
                        replay_deadline,
                    )
                },
            )
            .unwrap();
        let issued = match issue {
            ReverseOnionQueueIssue::Issued(issued) => issued,
            _ => panic!("expected issued lease"),
        };
        let sealed = seal_onion_reply(
            lease.route_id(),
            &OnionReplySession::prepare_source_sealed(
                lease.route_id(),
                api.recipient,
                ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
                vec![19; 3],
            )
            .unwrap()
            .0,
            b"opaque result",
            &recipient,
        )
        .unwrap();
        let result = ReverseOnionFrameV1::result(
            &claim,
            &lease,
            &encode_onion_sealed_response(&sealed).unwrap(),
            NOW + 60,
            NOW + 1,
            &recipient,
        )
        .unwrap();
        queue
            .complete(&issued, &result.encode(), NOW + 1, |_, _| Ok(result.commitment()))
            .unwrap();
        let query_bytes = query_bytes(&source, &relay, [11; 16], [14; 32], NOW, NOW + 20);
        let reply = api
            .handle_source_query(Bytes::from(query_bytes.clone()), NOW + 2)
            .await;
        assert_eq!(reply.status(), StatusCode::OK);
        let evidence = ReverseOnionSourceEvidenceV1::decode(reply.body()).unwrap();
        let query = ReverseOnionSourceQueryV1::decode(&query_bytes).unwrap();
        assert_eq!(evidence.state(), SourceEvidenceStateV1::Available);
        evidence.verify_for_query(&query, NOW + 2).unwrap();
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
