// ============================================
// File: crates/aeronyx-server/src/api/memchain_peer.rs
// ============================================
//! # MemChain Node Peer API — Commitment Block Synchronisation
//!
//! ## Creation Reason
//! Block Sync v1 needs a node-to-node transport that is separate from the VPN
//! client tunnel and from public discovery metadata. Reusing either surface
//! would let ordinary clients enumerate commitments or couple ledger catch-up
//! to unrelated descriptor gossip.
//!
//! ## Main Functionality
//! - `POST /api/memchain/peer/block-range`
//! - Bincode `MemChainMessage` request/response with the existing magic byte.
//! - Signed discovery-peer admission, timestamp freshness, replay protection,
//!   per-peer rate limiting, and bounded pagination.
//! - Response signing that binds request id, block order, pagination, and tip.
//!
//! ## Calling Relationships
//! - Mounted by `server.rs` on the public node peer listener and local operator
//!   listener when Local-mode `MemoryStorage` is available.
//! - Reads verified blocks through `MemoryStorage::get_record_commitment_block_range`.
//! - Uses `PeerStore::get_valid` as the node admission boundary.
//! - Uses canonical signing bytes from `aeronyx_core::protocol::memchain`.
//!
//! ## Privacy Invariant
//! This API returns only signed commitment blocks. It never returns memory
//! records, ciphertext, owners, tags, embeddings, client IPs, destinations,
//! routes, endpoints, or social graph metadata.
//!
//! ## Important Note for Next Developer
//! - Do not mount this handler without `PeerStore` admission.
//! - Do not add a JSON/debug response containing raw commitments.
//! - Do not increase body/page limits without memory and abuse testing.
//! - Sealed payload replication requires a separate owner-authorised protocol.
//!
//! ## Last Modified
//! v2.7.0-BlockSync - Initial signed node-blind block range protocol.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
use aeronyx_core::protocol::memchain::{
    decode_memchain, encode_memchain, record_block_range_request_signing_bytes,
    record_block_range_response_signing_bytes, MemChainMessage, MEMCHAIN_MAGIC,
};

use crate::services::memchain::MemoryStorage;
use crate::services::PeerStore;

const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024;
const MAX_BLOCKS_PER_RESPONSE: usize = 16;
const MAX_REQUESTS_PER_PEER_PER_MINUTE: u32 = 30;
const REQUEST_TIMESTAMP_SKEW_SECS: u64 = 60;
const REPLAY_RETENTION_SECS: u64 = 120;

#[derive(Clone)]
struct MemChainPeerState {
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    guard: Arc<Mutex<PeerRequestGuard>>,
}

#[derive(Debug, Default)]
struct PeerRequestGuard {
    rate_windows: HashMap<[u8; 32], PeerRateWindow>,
    seen_requests: HashMap<([u8; 32], [u8; 16]), u64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct PeerRateWindow {
    minute: u64,
    used: u32,
}

impl PeerRequestGuard {
    fn admit(&mut self, requester: [u8; 32], request_id: [u8; 16], now: u64) -> bool {
        self.seen_requests
            .retain(|_, seen_at| now.saturating_sub(*seen_at) <= REPLAY_RETENTION_SECS);
        if self.seen_requests.contains_key(&(requester, request_id)) {
            return false;
        }

        let minute = now / 60;
        let window = self.rate_windows.entry(requester).or_default();
        if window.minute != minute {
            *window = PeerRateWindow { minute, used: 0 };
        }
        if window.used >= MAX_REQUESTS_PER_PEER_PER_MINUTE {
            return false;
        }
        window.used += 1;
        self.seen_requests.insert((requester, request_id), now);
        true
    }
}

/// Builds the signed node-to-node commitment block sync router.
#[must_use]
pub fn build_memchain_peer_router(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
) -> Router {
    let state = MemChainPeerState {
        storage,
        peer_store,
        identity,
        guard: Arc::new(Mutex::new(PeerRequestGuard::default())),
    };
    Router::new()
        .route("/api/memchain/peer/block-range", post(block_range_handler))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}

async fn block_range_handler(State(state): State<MemChainPeerState>, body: Bytes) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id,
        from_height,
        limit,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || from_height == 0 || limit == 0 {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_range");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_block_range_request_signing_bytes(
        &chain_id,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    let signature_valid = IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let page_limit = usize::from(limit).min(MAX_BLOCKS_PER_RESPONSE);
    let blocks = match state
        .storage
        .get_record_commitment_block_range(from_height, page_limit)
        .await
    {
        Ok(blocks) => blocks,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to read verified block range");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error");
        }
    };
    let (tip_height, tip_hash) = state.storage.record_commitment_chain_tip().await;
    let page_tip = blocks
        .last()
        .map_or(from_height.saturating_sub(1), |block| block.header.height);
    let has_more = page_tip < tip_height;
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode block range response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    debug!(
        blocks = match &response {
            MemChainMessage::RecordBlockRangeResponseV1 { blocks, .. } => blocks.len(),
            _ => 0,
        },
        has_more, tip_height, "[MEMCHAIN_BLOCK] Served authenticated commitment range"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

fn protocol_error(status: StatusCode, code: &'static str) -> Response {
    (status, axum::Json(serde_json::json!({ "error": code }))).into_response()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    use aeronyx_core::ledger::{RecordCommitmentBlockV1, GENESIS_PREV_HASH};
    use aeronyx_core::protocol::{NodeDescriptor, NodeDiscoveryMessage, SignedNodeDescriptor};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[test]
    fn peer_guard_rejects_replay_and_enforces_rate_limit() {
        let peer = [0x11; 32];
        let now = 1_700_000_000;
        let mut guard = PeerRequestGuard::default();
        assert!(guard.admit(peer, [0x01; 16], now));
        assert!(!guard.admit(peer, [0x01; 16], now));
        for value in 2..=MAX_REQUESTS_PER_PEER_PER_MINUTE {
            let mut request_id = [0u8; 16];
            request_id[..4].copy_from_slice(&value.to_le_bytes());
            assert!(guard.admit(peer, request_id, now));
        }
        assert!(!guard.admit(peer, [0xFF; 16], now));
        assert!(guard.admit(peer, [0xFF; 16], now + 61));
    }

    #[tokio::test]
    async fn authenticated_range_sync_converges_two_commitment_ledgers() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let destination = MemoryStorage::open(":memory:", None).unwrap();

        let first = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x11; 32], [0x22; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let second = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first.hash(),
            vec![[0x33; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();

        let peer_store = Arc::new(PeerStore::new());
        let descriptor = NodeDescriptor::new(
            requester_identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now.saturating_add(600),
            "memchain-sync-test",
        );
        let descriptor = SignedNodeDescriptor::sign(descriptor, &requester_identity).unwrap();
        let import = peer_store.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
            now,
        );
        assert_eq!(import.inserted, 1);

        let request_id = [0xA7; 16];
        let requester = requester_identity.public_key_bytes();
        let signing_bytes = record_block_range_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            1,
            MAX_BLOCKS_PER_RESPONSE as u16,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: MAX_BLOCKS_PER_RESPONSE as u16,
            request_id,
            requester,
            request_timestamp: now,
            signature: requester_identity.sign(&signing_bytes),
        })
        .unwrap();
        let router = build_memchain_peer_router(
            Arc::clone(&source),
            peer_store,
            Arc::clone(&responder_identity),
        );
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-range")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(body.first().copied(), Some(MEMCHAIN_MAGIC));
        let response = decode_memchain(&body[1..]).unwrap();
        let MemChainMessage::RecordBlockRangeResponseV1 {
            request_id: response_request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } = response
        else {
            panic!("expected record block range response");
        };
        assert_eq!(response_request_id, request_id);
        assert_eq!(responder, responder_identity.public_key_bytes());
        assert!(!has_more);
        assert_eq!(tip_height, 2);
        assert_eq!(tip_hash, second.hash());
        let response_signing_bytes = record_block_range_response_signing_bytes(
            &response_request_id,
            &responder,
            response_timestamp,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        IdentityPublicKey::from_bytes(&responder)
            .unwrap()
            .verify(&response_signing_bytes, &signature)
            .unwrap();

        for block in &blocks {
            destination
                .append_record_commitment_block(block, Some(&responder))
                .await
                .unwrap();
        }
        assert_eq!(blocks, vec![first, second]);
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            source.record_commitment_chain_tip().await
        );
        let status = destination.record_commitment_chain_status().await;
        assert_eq!(status.block_count, 2);
        assert_eq!(status.commitment_count, 3);
    }
}
