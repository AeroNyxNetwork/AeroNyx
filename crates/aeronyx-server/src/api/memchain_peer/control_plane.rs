//! Authenticated MemChain control-plane route composition.
//!
//! This module owns only the lease, handover, and external witness route
//! surface. Block/checkpoint/certificate synchronization remains in the
//! parent module. Keeping the route set behind one typed function prevents
//! callers from accidentally mounting control traffic on a public router.
// [MEMCHAIN-CONTROL-PLANE-SPLIT 2026-09-24 by Codex]

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use tracing::{debug, warn};

use crate::services::memchain::storage_ops::{
    CustodyAuditAnchorWitnessOutcome, RecordCoordinatorLeaseGrantOutcome,
    RecordCoordinatorLeaseReleaseOutcome, VerifiedDeliveryAnchorWitnessOutcome,
};
use aeronyx_core::crypto::IdentityPublicKey;
use aeronyx_core::ledger::AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
use aeronyx_core::protocol::chat::{
    custody_audit_anchor_frame_sha256, custody_audit_witness_receipt_frame_sha256,
    CustodyAuditAnchorV1, CustodyAuditWitnessReceiptV1, CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
    CUSTODY_AUDIT_WITNESS_CONFLICT_V1, CUSTODY_AUDIT_WITNESS_GAP_V1,
    CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1, CUSTODY_AUDIT_WITNESS_STALE_V1,
};
use aeronyx_core::protocol::memchain::{
    custody_audit_anchor_witness_request_signing_bytes,
    custody_audit_anchor_witness_response_signing_bytes, decode_memchain, encode_memchain,
    record_coordinator_handover_request_signing_bytes,
    record_coordinator_handover_response_signing_bytes,
    record_coordinator_lease_release_request_signing_bytes,
    record_coordinator_lease_release_response_signing_bytes,
    record_coordinator_lease_request_signing_bytes,
    record_coordinator_lease_response_signing_bytes,
    verified_delivery_anchor_witness_request_signing_bytes,
    verified_delivery_anchor_witness_response_signing_bytes, MemChainMessage,
    MAX_COORDINATOR_LEASE_TTL_SECS_V1, MEMCHAIN_MAGIC, MIN_COORDINATOR_LEASE_TTL_SECS_V1,
    VERIFIED_DELIVERY_WITNESS_ADVANCED_V1, VERIFIED_DELIVERY_WITNESS_CONFLICT_V1,
    VERIFIED_DELIVERY_WITNESS_GAP_V1, VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
    VERIFIED_DELIVERY_WITNESS_STALE_V1,
};

use super::{
    coordinator_control_requester_is_admitted, now_secs, protocol_error,
    runtime_authorized_coordinator_for_height, runtime_authorized_coordinator_for_next_height,
    verified_local_commitment_tip, MemChainPeerState, VerifiedCoordinatorHandoverResponse,
    REQUEST_TIMESTAMP_SKEW_SECS,
};

async fn coordinator_handover_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorHandoverRequestV1 {
        chain_id,
        after_authority_epoch,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || after_authority_epoch == u64::MAX {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_handover_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = record_coordinator_handover_request_signing_bytes(
        &chain_id,
        after_authority_epoch,
        &request_id,
        &requester,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [AUTHORITY-HANDOVER-ADMISSION 2026-08-14 by Codex] Authenticate before
    // consulting PeerStore. Otherwise a forged request can distinguish a
    // known public key from an unknown one by comparing HTTP error classes.
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let page = match state
        .storage
        .next_record_coordinator_handover_page(after_authority_epoch)
        .await
    {
        Ok(page) => page,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused authority handover snapshot");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "handover_history_unavailable",
            );
        }
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let has_handover = page.handover.is_some();
    let response_signing_bytes = record_coordinator_handover_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        page.handover.as_ref(),
        page.latest_authority_epoch,
    );
    let response = MemChainMessage::RecordCoordinatorHandoverResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        handover: page.handover,
        latest_authority_epoch: page.latest_authority_epoch,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        has_handover,
        latest_authority_epoch = page.latest_authority_epoch,
        "[MEMCHAIN_BLOCK] Served authenticated authority handover snapshot"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn coordinator_lease_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorLeaseRequestV1 {
        chain_id,
        coordinator,
        instance_id,
        known_tip_height,
        known_tip_hash,
        requested_ttl_secs,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || instance_id.iter().all(|byte| *byte == 0)
        || !(MIN_COORDINATOR_LEASE_TTL_SECS_V1..=MAX_COORDINATOR_LEASE_TTL_SECS_V1)
            .contains(&requested_ttl_secs)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = record_coordinator_lease_request_signing_bytes(
        &chain_id,
        &coordinator,
        &instance_id,
        known_tip_height,
        &known_tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&coordinator)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [COORDINATOR-CONTROL-ADMISSION 2026-08-14 by Codex] Only an
    // authenticated coordinator may trigger the storage-backed authority
    // lookup or learn its peer-admission result.
    let Some(next_height) = known_tip_height.checked_add(1) else {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_request");
    };
    let authorized_coordinator = match runtime_authorized_coordinator_for_height(
        &state.storage,
        state.lease_authorized_coordinator,
        next_height,
    )
    .await
    {
        Ok(Some(authorized)) => authorized,
        Ok(None) => return protocol_error(StatusCode::FORBIDDEN, "follower_sync_disabled"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited lease authority");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "coordinator_authority_unavailable",
            );
        }
    };
    if authorized_coordinator != coordinator {
        return protocol_error(StatusCode::FORBIDDEN, "unauthorized_coordinator");
    }
    if !coordinator_control_requester_is_admitted(&state, &coordinator, now) {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state.guard.lock().await.admit(coordinator, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let witness_tip = match verified_local_commitment_tip(&state.storage).await {
        Ok(tip) => tip,
        Err(_) => {
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "witness_tip_unavailable");
        }
    };
    if witness_tip != (known_tip_height, known_tip_hash) {
        return protocol_error(StatusCode::CONFLICT, "lease_tip_mismatch");
    }
    let grant = match state
        .storage
        .grant_record_commitment_coordinator_lease(
            &chain_id,
            &coordinator,
            &instance_id,
            known_tip_height,
            &known_tip_hash,
            now,
            requested_ttl_secs,
        )
        .await
    {
        Ok(RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch,
            lease_expires_at,
        }) => (lease_epoch, lease_expires_at),
        Ok(RecordCoordinatorLeaseGrantOutcome::TipMismatch) => {
            return protocol_error(StatusCode::CONFLICT, "lease_tip_mismatch");
        }
        Ok(RecordCoordinatorLeaseGrantOutcome::Contended) => {
            return protocol_error(StatusCode::CONFLICT, "lease_contended");
        }
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Coordinator lease persistence failed");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "lease_persist_failed");
        }
    };
    let witness = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        response_timestamp,
        grant.0,
        grant.1,
        witness_tip.0,
        &witness_tip.1,
    );
    let response = MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        response_timestamp,
        lease_epoch: grant.0,
        lease_expires_at: grant.1,
        witness_tip_height: witness_tip.0,
        witness_tip_hash: witness_tip.1,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        lease_epoch = grant.0,
        lease_ttl_secs = requested_ttl_secs,
        "[MEMCHAIN_BLOCK] Granted authenticated coordinator lease"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn coordinator_lease_release_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
        chain_id,
        coordinator,
        instance_id,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || instance_id.iter().all(|byte| *byte == 0) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_release_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
        &chain_id,
        &coordinator,
        &instance_id,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&coordinator)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [COORDINATOR-CONTROL-ADMISSION 2026-08-14 by Codex] Release requests
    // use the same authenticate-before-authorize ordering as acquisition.
    let authorized_coordinator = match runtime_authorized_coordinator_for_next_height(
        &state.storage,
        state.lease_authorized_coordinator,
    )
    .await
    {
        Ok(Some(authorized)) => authorized,
        Ok(None) => return protocol_error(StatusCode::FORBIDDEN, "follower_sync_disabled"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited lease release authority");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "coordinator_authority_unavailable",
            );
        }
    };
    if authorized_coordinator != coordinator {
        return protocol_error(StatusCode::FORBIDDEN, "unauthorized_coordinator");
    }
    if !coordinator_control_requester_is_admitted(&state, &coordinator, now) {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state.guard.lock().await.admit(coordinator, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let (lease_epoch, released_at) = match state
        .storage
        .release_record_commitment_coordinator_lease(&chain_id, &coordinator, &instance_id, now)
        .await
    {
        Ok(RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch,
            released_at,
        }) => (lease_epoch, released_at),
        Ok(RecordCoordinatorLeaseReleaseOutcome::NotHolder) => {
            return protocol_error(StatusCode::CONFLICT, "lease_release_not_holder");
        }
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Coordinator lease release persistence failed");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "lease_release_persist_failed",
            );
        }
    };
    let witness = state.identity.public_key_bytes();
    let response_signing_bytes = record_coordinator_lease_release_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        released_at,
        lease_epoch,
    );
    let response = MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        released_at,
        lease_epoch,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        lease_epoch,
        "[MEMCHAIN_BLOCK] Released authenticated coordinator lease"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn verified_delivery_anchor_witness_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let canonical = match encode_memchain(&message) {
        Ok(canonical) => canonical,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    if canonical.as_slice() != body.as_ref() {
        // [WITNESS-ADMISSION-PRIVACY 2026-08-16 by Codex] Witness writes are
        // security evidence. Reject alternate encodings instead of allowing
        // one signed request to acquire multiple replay identities.
        return protocol_error(StatusCode::BAD_REQUEST, "noncanonical_frame");
    }
    let MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
        requester,
        generation,
        anchor_digest,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if generation == 0 || generation > i64::MAX as u64 || anchor_digest == [0u8; 32] {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_delivery_witness_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
        &requester,
        generation,
        &anchor_digest,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    // [WITNESS-ADMISSION-PRIVACY 2026-08-16 by Codex] Authenticate before
    // consulting private operator pins, then collapse pin and discovery
    // failures into one response. An unauthenticated caller cannot use status
    // differences to enumerate a witness node's trust relationships.
    if !state
        .peer_store
        .verified_delivery_witness_requester_allowed(&requester)
        || state.peer_store.get_valid(&requester, now).is_none()
    {
        return protocol_error(StatusCode::FORBIDDEN, "witness_requester_not_authorized");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let (outcome, witness_generation, witness_anchor_digest) = match state
        .storage
        .witness_verified_delivery_anchor(&requester, generation, &anchor_digest, now)
        .await
    {
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Advanced {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_ADVANCED_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Idempotent {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Stale {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_STALE_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Conflict {
            generation,
            anchor_digest,
        }) => (
            VERIFIED_DELIVERY_WITNESS_CONFLICT_V1,
            generation,
            anchor_digest,
        ),
        Ok(VerifiedDeliveryAnchorWitnessOutcome::Gap {
            generation,
            anchor_digest,
        }) => (VERIFIED_DELIVERY_WITNESS_GAP_V1, generation, anchor_digest),
        Err(error) => {
            warn!(error = %error, "[DISCOVERY] Delivery-anchor witness persistence failed");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "delivery_witness_persist_failed",
            );
        }
    };

    let witness = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        generation,
        &anchor_digest,
        &witness,
        response_timestamp,
        witness_generation,
        &witness_anchor_digest,
        outcome,
    );
    let response = MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
        request_id,
        requester,
        requested_generation: generation,
        requested_anchor_digest: anchor_digest,
        witness,
        response_timestamp,
        witness_generation,
        witness_anchor_digest,
        outcome,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        generation,
        outcome, "[DISCOVERY] Served authenticated delivery-anchor witness decision"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn custody_audit_anchor_witness_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let canonical = match encode_memchain(&message) {
        Ok(canonical) => canonical,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    if canonical.as_slice() != body.as_ref() {
        return protocol_error(StatusCode::BAD_REQUEST, "noncanonical_frame");
    }
    let MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
        request_id,
        requester,
        request_timestamp,
        anchor,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if anchor.checkpoint_generation == 0
        || anchor.checkpoint_generation > i64::MAX as u64
        || requester != anchor.producer_node_id
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_custody_witness_request");
    }
    // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Keep freshness failures
    // distinct from structural failures so operators can identify replay or
    // clock-skew incidents without exposing request identities in logs.
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    let witness = state.identity.public_key_bytes();
    if anchor
        .verify_expected(&requester, anchor.checkpoint_generation)
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_anchor_signature");
    }
    let anchor_sha256 = match custody_audit_anchor_frame_sha256(&anchor) {
        Ok(digest) if digest != [0u8; 32] => digest,
        Ok(_) | Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_custody_anchor"),
    };
    let signing_bytes = custody_audit_anchor_witness_request_signing_bytes(
        &request_id,
        &requester,
        request_timestamp,
        &anchor_sha256,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if requester == witness {
        // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Reject only after
        // authentication and before monotonic state changes. Self-witness
        // evidence remains invalid without becoming a signature oracle.
        return protocol_error(StatusCode::FORBIDDEN, "independent_witness_required");
    }
    if !state
        .peer_store
        .custody_audit_witness_requester_allowed(&requester)
        || state.peer_store.get_valid(&requester, now).is_none()
    {
        return protocol_error(StatusCode::FORBIDDEN, "custody_witness_not_authorized");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let (outcome, witness_generation, witness_anchor_sha256) = match state
        .storage
        .witness_custody_audit_anchor(
            &requester,
            anchor.checkpoint_generation,
            &anchor_sha256,
            now,
        )
        .await
    {
        Ok(CustodyAuditAnchorWitnessOutcome::Advanced {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_ADVANCED_V1, generation, anchor_digest),
        Ok(CustodyAuditAnchorWitnessOutcome::Idempotent {
            generation,
            anchor_digest,
        }) => (
            CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1,
            generation,
            anchor_digest,
        ),
        Ok(CustodyAuditAnchorWitnessOutcome::Stale {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_STALE_V1, generation, anchor_digest),
        Ok(CustodyAuditAnchorWitnessOutcome::Conflict {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_CONFLICT_V1, generation, anchor_digest),
        Ok(CustodyAuditAnchorWitnessOutcome::Gap {
            generation,
            anchor_digest,
        }) => (CUSTODY_AUDIT_WITNESS_GAP_V1, generation, anchor_digest),
        Err(_) => {
            warn!(
                generation = anchor.checkpoint_generation,
                "[MEMCHAIN] Custody-anchor witness persistence failed"
            );
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "custody_witness_persist_failed",
            );
        }
    };

    let receipt = match CustodyAuditWitnessReceiptV1::signed(
        requester,
        anchor.checkpoint_generation,
        anchor_sha256,
        now,
        witness_generation,
        witness_anchor_sha256,
        outcome,
        &state.identity,
    ) {
        Ok(receipt) => receipt,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "receipt_sign_failed"),
    };
    let receipt_sha256 = match custody_audit_witness_receipt_frame_sha256(&receipt) {
        Ok(digest) => digest,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    let response_signing_bytes = custody_audit_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        &witness,
        receipt.observed_at,
        &receipt_sha256,
    );
    let response = MemChainMessage::CustodyAuditAnchorWitnessResponseV1 {
        request_id,
        requester,
        witness,
        response_timestamp: receipt.observed_at,
        receipt,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    if verify_custody_audit_anchor_witness_response(
        &encoded,
        &request_id,
        &requester,
        &witness,
        &anchor,
        &anchor_sha256,
        now,
    )
    .is_err()
    {
        // A locally generated frame that fails the public verification
        // contract must never leave the process or become apparent evidence.
        return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "receipt_verify_failed");
    }
    debug!(
        generation = anchor.checkpoint_generation,
        outcome, "[MEMCHAIN] Served authenticated custody-anchor witness decision"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

pub(super) fn verify_record_coordinator_handover_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    expected_previous_coordinator: &[u8; 32],
    expected_authority_epoch: u64,
    expected_next_block_height: u64,
    now: u64,
) -> Result<VerifiedCoordinatorHandoverResponse, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_handover_response_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_handover_response_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_handover_response_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_handover_response".to_string());
    }
    let MemChainMessage::RecordCoordinatorHandoverResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        handover,
        latest_authority_epoch,
        signature,
    } = response
    else {
        return Err("unexpected_handover_response".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("handover_response_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("handover_response_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("handover_response_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_handover_response".to_string());
    }
    if latest_authority_epoch < expected_authority_epoch {
        return Err("handover_history_rollback".to_string());
    }
    let signing_bytes = record_coordinator_handover_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        handover.as_ref(),
        latest_authority_epoch,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_handover_response_signature".to_string())?;

    match handover.as_ref() {
        Some(proof) => {
            proof
                .verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID)
                .map_err(|_| "invalid_handover_proof".to_string())?;
            let expected_epoch = expected_authority_epoch
                .checked_add(1)
                .ok_or_else(|| "authority_epoch_exhausted".to_string())?;
            if proof.header.authority_epoch != expected_epoch {
                return Err("handover_epoch_discontinuity".to_string());
            }
            if proof.header.previous_coordinator != *expected_previous_coordinator {
                return Err("handover_previous_coordinator_mismatch".to_string());
            }
            if proof.header.activation_height < expected_next_block_height {
                return Err("handover_activation_rollback".to_string());
            }
            if latest_authority_epoch < proof.header.authority_epoch {
                return Err("handover_history_head_mismatch".to_string());
            }
        }
        None if latest_authority_epoch != expected_authority_epoch => {
            return Err("handover_proof_omitted".to_string());
        }
        None => {}
    }

    Ok(VerifiedCoordinatorHandoverResponse {
        handover,
        latest_authority_epoch,
    })
}

pub(super) fn verify_delivery_anchor_witness_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_requester: &[u8; 32],
    expected_generation: u64,
    expected_anchor_digest: &[u8; 32],
    expected_witness: &[u8; 32],
    now: u64,
) -> Result<u8, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_delivery_witness_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_delivery_witness_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_delivery_witness_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_delivery_witness_frame".to_string());
    }
    let MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
        request_id,
        requester,
        requested_generation,
        requested_anchor_digest,
        witness,
        response_timestamp,
        witness_generation,
        witness_anchor_digest,
        outcome,
        signature,
    } = response
    else {
        return Err("unexpected_delivery_witness_message".to_string());
    };
    if request_id != *expected_request_id
        || requester != *expected_requester
        || requested_generation != expected_generation
        || requested_anchor_digest != *expected_anchor_digest
    {
        return Err("delivery_witness_request_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("delivery_witness_identity_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
        || witness_generation == 0
        || witness_generation > i64::MAX as u64
        || witness_anchor_digest == [0u8; 32]
    {
        return Err("delivery_witness_state_invalid".to_string());
    }

    let relation_valid = match outcome {
        VERIFIED_DELIVERY_WITNESS_ADVANCED_V1 | VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1 => {
            witness_generation == expected_generation
                && witness_anchor_digest == *expected_anchor_digest
        }
        VERIFIED_DELIVERY_WITNESS_STALE_V1 => witness_generation > expected_generation,
        VERIFIED_DELIVERY_WITNESS_CONFLICT_V1 => {
            witness_generation == expected_generation
                && witness_anchor_digest != *expected_anchor_digest
        }
        VERIFIED_DELIVERY_WITNESS_GAP_V1 => witness_generation
            .checked_add(1)
            .is_some_and(|next| expected_generation > next),
        _ => false,
    };
    if !relation_valid {
        return Err("delivery_witness_outcome_invalid".to_string());
    }

    let signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        requested_generation,
        &requested_anchor_digest,
        &witness,
        response_timestamp,
        witness_generation,
        &witness_anchor_digest,
        outcome,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_delivery_witness_signature".to_string())?;
    Ok(outcome)
}

pub(super) fn verify_custody_audit_anchor_witness_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_requester: &[u8; 32],
    expected_witness: &[u8; 32],
    anchor: &CustodyAuditAnchorV1,
    anchor_frame_sha256: &[u8; 32],
    now: u64,
) -> Result<CustodyAuditWitnessReceiptV1, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_custody_witness_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_custody_witness_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_custody_witness_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_custody_witness_frame".to_string());
    }
    let MemChainMessage::CustodyAuditAnchorWitnessResponseV1 {
        request_id,
        requester,
        witness,
        response_timestamp,
        receipt,
        signature,
    } = response
    else {
        return Err("unexpected_custody_witness_message".to_string());
    };
    if request_id != *expected_request_id || requester != *expected_requester {
        return Err("custody_witness_request_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("custody_witness_identity_mismatch".to_string());
    }
    if response_timestamp != receipt.observed_at
        || now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
    {
        return Err("custody_witness_timestamp_invalid".to_string());
    }

    let receipt_sha256 = custody_audit_witness_receipt_frame_sha256(&receipt)
        .map_err(|_| "invalid_custody_witness_receipt".to_string())?;
    let signing_bytes = custody_audit_anchor_witness_response_signing_bytes(
        &request_id,
        &requester,
        &witness,
        response_timestamp,
        &receipt_sha256,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_custody_witness_response_signature".to_string())?;
    receipt
        .verify_for_anchor(
            anchor,
            anchor_frame_sha256,
            expected_requester,
            expected_witness,
            1,
        )
        .map_err(|_| "invalid_custody_witness_receipt".to_string())?;
    Ok(receipt)
}

/// Mounts the authenticated control-plane routes onto the already-scoped peer
/// router. The caller owns the state construction and body limit; this helper
/// cannot create a second router or expose the control routes elsewhere.
pub(super) fn mount_routes(router: Router<MemChainPeerState>) -> Router<MemChainPeerState> {
    router
        .route(
            "/api/memchain/peer/coordinator-lease",
            post(coordinator_lease_handler),
        )
        .route(
            "/api/memchain/peer/coordinator-lease/release",
            post(coordinator_lease_release_handler),
        )
        .route(
            "/api/memchain/peer/coordinator-handover",
            post(coordinator_handover_handler),
        )
        .route(
            "/api/discovery/peer/verified-delivery-anchor-witness",
            post(verified_delivery_anchor_witness_handler),
        )
        .route(
            "/api/memchain/peer/custody-audit-anchor-witness",
            post(custody_audit_anchor_witness_handler),
        )
}
