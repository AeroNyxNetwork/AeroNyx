// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer.rs
// ============================================================================
//! # Inter-Node Encrypted Chat Relay API
//!
//! ## Creation Reason
//! Phase 9 connects AeroNyx node discovery to real encrypted message movement.
//! Discovery tells a node which peers advertise `NodeCapability::ChatRelay`;
//! this module exposes the receiving side for those peers.
//!
//! ## Main Functionality
//! - `POST /api/chat/peer/relay`: accepts a signed `ChatEnvelope` from another
//!   AeroNyx node
//! - `POST /api/chat/peer/blind-relay`: accepts a signed `BlindRelayEnvelope`
//!   and forwards only opaque encrypted bytes toward `next_hop`
//! - Verifies the envelope signature before doing any delivery or storage
//! - Delivers to locally online receiver devices when possible
//! - Falls back to the existing SQLite pending queue when the receiver is
//!   offline or all local routes are stale
//!
//! ## Dependencies
//! - aeronyx-core/src/protocol/chat.rs: `ChatEnvelope`, `BlindRelayEnvelope`,
//!   and bounded envelope encoding
//! - aeronyx-core/src/protocol/memchain.rs: wraps envelope for client delivery
//! - aeronyx-server/src/services/chat_relay.rs: pending queue and dedup logic
//! - aeronyx-server/src/services/peer_store.rs: verified node descriptors for
//!   next-hop routing
//! - aeronyx-server/src/services/session.rs: active receiver sessions
//! - aeronyx-transport/src/udp.rs: encrypted client packet send path
//!
//! ## Main Logical Flow
//! 1. Peer node posts an already end-to-end encrypted `ChatEnvelope`
//! 2. This node checks size and sender signature
//! 3. Duplicate message IDs are ignored idempotently
//! 4. Online receiver sessions get the envelope through the existing encrypted
//!    client transport
//! 5. Offline receivers keep the existing pending-message fallback
//! 6. Blind relay requests verify the previous-hop signature, decrement TTL,
//!    re-sign with this node key, and POST to the verified `next_hop`
//!
//! ## Important Note for Next Developer
//! - Never decrypt, inspect, log, store, or report ciphertext contents.
//! - Do not add client public IPs, destination domains, DNS contents, URLs,
//!   browsing history, voucher secrets, private keys, or wallet-level traffic
//!   analytics to this endpoint.
//! - The endpoint is node-to-node plumbing only. Client wire format remains
//!   `MemChainMessage::ChatRelay(ChatEnvelope)`.
//! - Blind relay keeps the relay invariant: route_id / next_hop / ttl /
//!   encrypted_blob / timestamp / signature are handled as routing metadata;
//!   encrypted_blob stays opaque and must not be parsed.
//!
//! ## Last Modified
//! v0.5.0-BlindRelayRouteHealth - Feed next-hop success/failure back into PeerStore scoring
//! v0.4.0-BlindRelayBackpressure - Added blind relay in-flight pressure gate
//! v0.3.0-BlindRelayEndpoint - Added node-to-node opaque blind relay endpoint
//! v0.2.0-PeerRelayHealth - Record inbound peer relay health counters
//! v0.1.0-DiscoveryPhase9 - Initial inter-node encrypted chat relay endpoint
// ============================================================================

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use aeronyx_core::crypto::transport::{
    DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD,
};
use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::chat::{
    encode_blind_relay_envelope, encode_envelope, BlindRelayEnvelope, ChatEnvelope,
};
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::DataPacket;
use aeronyx_transport::traits::Transport;
use aeronyx_transport::UdpTransport;
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::services::peer_store::PeerStore;
use crate::services::{ChatRelayService, Session, SessionManager};

// ============================================
// Constants
// ============================================

/// Maximum bincode-encoded envelope bytes accepted from another node.
///
/// This mirrors the protocol decode limit and protects the JSON endpoint from
/// carrying huge opaque payloads. Encrypted files should use blob storage, not
/// the peer envelope relay path.
const MAX_PEER_CHAT_ENVELOPE_BYTES: usize = 128 * 1024;

/// Maximum concurrent blind relay requests handled by this process.
///
/// Blind relay is intentionally opaque and can carry large encrypted blobs, so
/// it needs a hard in-flight cap before future multi-hop routing increases the
/// possible fanout. This is local backpressure only; callers should retry with
/// jitter at the transport/client layer.
const MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS: usize = 64;

// ============================================
// State / Request / Response Types
// ============================================

#[derive(Clone)]
struct ChatPeerState {
    chat_relay: Option<Arc<ChatRelayService>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
    blind_relay_in_flight: Arc<AtomicUsize>,
}

/// Node-to-node encrypted envelope relay request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerChatRelayRequest {
    /// End-to-end encrypted, sender-signed chat envelope.
    pub envelope: ChatEnvelope,
}

/// Node-to-node encrypted envelope relay response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerChatRelayResponse {
    /// Whether this node accepted the envelope as valid relay work.
    pub accepted: bool,
    /// Whether the message id was already seen on this node.
    pub duplicate: bool,
    /// Number of local online receiver sessions reached.
    pub delivered_online: usize,
    /// Whether the envelope was stored in the local pending queue.
    pub stored_pending: bool,
}

/// Node-to-node blind relay request.
///
/// `previous_hop_node_id` is transport/auth context for signature
/// verification. It is intentionally outside `BlindRelayEnvelope` so the
/// envelope itself remains the minimal route metadata set documented in
/// aeronyx-core. Do not add user ids, receiver wallet ids, domains, URLs, DNS
/// contents, or payload-derived information here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerBlindRelayRequest {
    /// Opaque encrypted relay envelope. `encrypted_blob` must not be parsed.
    pub envelope: BlindRelayEnvelope,
    /// Ed25519 node id that signed this hop.
    pub previous_hop_node_id: [u8; 32],
}

/// Node-to-node blind relay response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerBlindRelayResponse {
    /// Whether this node accepted the request as valid relay work.
    pub accepted: bool,
    /// Whether this node is the requested next hop.
    pub terminal: bool,
    /// Whether this node forwarded the opaque envelope to another node.
    pub forwarded: bool,
    /// Remaining TTL observed or forwarded by this node.
    pub ttl_remaining: u8,
    /// Privacy-safe coarse result bucket for nodeboard/audits.
    pub reason: Option<String>,
}

#[derive(Debug, thiserror::Error)]
enum ChatPeerRelayError {
    #[error("chat relay disabled")]
    RelayUnavailable,

    #[error("invalid envelope signature")]
    InvalidSignature,

    #[error("envelope too large: {size} bytes")]
    EnvelopeTooLarge { size: usize },

    #[error("envelope serialization failed")]
    Serialization,

    #[error("pending store failed")]
    StoreFailed,
}

#[derive(Debug, thiserror::Error)]
enum BlindRelayError {
    #[error("invalid previous hop public key")]
    InvalidPreviousHop,

    #[error("invalid blind envelope signature")]
    InvalidSignature,

    #[error("blind envelope too large")]
    EnvelopeTooLarge,

    #[error("ttl exhausted")]
    TtlExhausted,

    #[error("next hop not found")]
    NoRoute,

    #[error("next hop endpoint missing or invalid")]
    InvalidEndpoint,

    #[error("blind relay forward failed")]
    ForwardFailed,
}

impl BlindRelayError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidPreviousHop
            | Self::InvalidSignature
            | Self::EnvelopeTooLarge
            | Self::TtlExhausted => StatusCode::BAD_REQUEST,
            Self::NoRoute | Self::InvalidEndpoint => StatusCode::BAD_GATEWAY,
            Self::ForwardFailed => StatusCode::BAD_GATEWAY,
        }
    }

    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::InvalidPreviousHop => "invalid_previous_hop",
            Self::InvalidSignature => "invalid_signature",
            Self::EnvelopeTooLarge => "envelope_too_large",
            Self::TtlExhausted => "ttl_exhausted",
            Self::NoRoute => "no_route",
            Self::InvalidEndpoint => "invalid_endpoint",
            Self::ForwardFailed => "forward_failed",
        }
    }
}

impl ChatPeerRelayError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::RelayUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::InvalidSignature | Self::EnvelopeTooLarge { .. } | Self::Serialization => {
                StatusCode::BAD_REQUEST
            }
            Self::StoreFailed => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::RelayUnavailable => "relay_unavailable",
            Self::InvalidSignature => "invalid_signature",
            Self::EnvelopeTooLarge { .. } => "envelope_too_large",
            Self::Serialization => "envelope_serialization_failed",
            Self::StoreFailed => "store_pending_failed",
        }
    }
}

// ============================================
// Router
// ============================================

/// Builds node-to-node encrypted chat relay routes.
pub fn build_chat_peer_router(
    chat_relay: Option<Arc<ChatRelayService>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
) -> Router {
    let state = ChatPeerState {
        chat_relay,
        sessions,
        udp,
        peer_store,
        node_identity,
        http_client,
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
    };

    Router::new()
        .route("/api/chat/peer/relay", post(peer_relay_handler))
        .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
        .with_state(state)
}

// ============================================
// Handlers
// ============================================

async fn peer_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerChatRelayRequest>,
) -> impl IntoResponse {
    match process_peer_relay(state, request.envelope).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (
            error.status_code(),
            Json(PeerChatRelayResponse {
                accepted: false,
                duplicate: false,
                delivered_online: 0,
                stored_pending: false,
            }),
        )
            .into_response(),
    }
}

async fn peer_blind_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerBlindRelayRequest>,
) -> impl IntoResponse {
    let Some(_in_flight) = BlindRelayInFlightGuard::try_acquire(&state) else {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "backpressure");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(PeerBlindRelayResponse {
                accepted: false,
                terminal: false,
                forwarded: false,
                ttl_remaining: 0,
                reason: Some("backpressure".to_string()),
            }),
        )
            .into_response();
    };

    match process_peer_blind_relay(state, request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (
            error.status_code(),
            Json(PeerBlindRelayResponse {
                accepted: false,
                terminal: false,
                forwarded: false,
                ttl_remaining: 0,
                reason: Some(error.reason_bucket().to_string()),
            }),
        )
            .into_response(),
    }
}

struct BlindRelayInFlightGuard {
    counter: Arc<AtomicUsize>,
}

impl BlindRelayInFlightGuard {
    fn try_acquire(state: &ChatPeerState) -> Option<Self> {
        let counter = Arc::clone(&state.blind_relay_in_flight);
        let mut current = counter.load(Ordering::Relaxed);
        loop {
            if current >= MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(Self { counter }),
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for BlindRelayInFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn process_peer_relay(
    state: ChatPeerState,
    envelope: ChatEnvelope,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let now = now_secs();
    if let Err(error) = validate_peer_envelope(&envelope) {
        if let Some(relay) = state.chat_relay.as_ref() {
            relay.record_peer_relay_inbound_rejected(now, error.reason_bucket());
        }
        return Err(error);
    }

    let Some(relay) = state.chat_relay else {
        return Err(ChatPeerRelayError::RelayUnavailable);
    };

    if relay.is_online_duplicate(&envelope.message_id) {
        debug!(
            id = %hex::encode(envelope.message_id),
            "[CHAT_PEER] Duplicate peer envelope ignored"
        );
        relay.record_peer_relay_inbound_accepted(now, true, 0, false);
        return Ok(PeerChatRelayResponse {
            accepted: true,
            duplicate: true,
            delivered_online: 0,
            stored_pending: false,
        });
    }

    let target_sessions = state.sessions.get_all_by_wallet(&envelope.receiver);
    let mut delivered_online = 0usize;

    for session in target_sessions {
        if send_envelope_to_session(&envelope, &session, &state.udp).await {
            delivered_online += 1;
        }
    }

    let stored_pending = if delivered_online == 0 {
        relay.store_pending(&envelope).map_err(|error| {
            warn!(
                receiver = %hex::encode(&envelope.receiver[..4]),
                error = %error,
                "[CHAT_PEER] Failed to store peer envelope for offline receiver"
            );
            relay.record_peer_relay_inbound_rejected(now, "store_pending_failed");
            ChatPeerRelayError::StoreFailed
        })?;
        true
    } else {
        false
    };

    relay.record_peer_relay_inbound_accepted(now, false, delivered_online, stored_pending);

    Ok(PeerChatRelayResponse {
        accepted: true,
        duplicate: false,
        delivered_online,
        stored_pending,
    })
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn process_peer_blind_relay(
    state: ChatPeerState,
    request: PeerBlindRelayRequest,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let now = now_secs();
    let envelope = request.envelope;

    validate_blind_relay_envelope(&envelope, &request.previous_hop_node_id).map_err(|error| {
        state
            .peer_store
            .record_blind_relay_rejected(now, error.reason_bucket());
        error
    })?;

    let self_node_id = state.node_identity.public_key_bytes();
    if envelope.next_hop == self_node_id {
        state.peer_store.record_blind_relay_terminal(
            now,
            envelope.ttl,
            envelope.encrypted_blob.len(),
        );
        return Ok(PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: envelope.ttl,
            reason: Some("terminal_next_hop".to_string()),
        });
    }

    if !envelope.can_forward() {
        state
            .peer_store
            .record_blind_relay_rejected(now, "ttl_exhausted");
        return Err(BlindRelayError::TtlExhausted);
    }

    let next_hop = envelope.next_hop;
    let descriptor = state.peer_store.get_valid(&next_hop, now).ok_or_else(|| {
        state
            .peer_store
            .record_blind_relay_rejected(now, "no_route");
        BlindRelayError::NoRoute
    })?;

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            state
                .peer_store
                .record_route_forward_failure(&next_hop, now, "missing_endpoint");
            state
                .peer_store
                .record_blind_relay_rejected(now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        state
            .peer_store
            .record_route_forward_failure(&next_hop, now, "invalid_endpoint");
        state
            .peer_store
            .record_blind_relay_rejected(now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    let forwarded_envelope = envelope
        .decremented_ttl()
        .ok_or(BlindRelayError::TtlExhausted)?
        .sign_with(state.node_identity.as_ref());
    let ttl_remaining = forwarded_envelope.ttl;

    let response = state
        .http_client
        .post(&url)
        .json(&PeerBlindRelayRequest {
            envelope: forwarded_envelope,
            previous_hop_node_id: self_node_id,
        })
        .send()
        .await
        .map_err(|error| {
            debug!(
                reason = %classify_reqwest_error("blind_relay_request", &error),
                "[BLIND_RELAY] Next-hop forward failed"
            );
            state
                .peer_store
                .record_route_forward_failure(&next_hop, now, "request_failed");
            state
                .peer_store
                .record_blind_relay_rejected(now, "request_failed");
            BlindRelayError::ForwardFailed
        })?;

    if !response.status().is_success() {
        let reason = format!("http_{}", response.status().as_u16());
        debug!(
            status = %response.status(),
            "[BLIND_RELAY] Next-hop returned non-success"
        );
        state
            .peer_store
            .record_route_forward_failure(&next_hop, now, reason.clone());
        state.peer_store.record_blind_relay_rejected(now, reason);
        return Err(BlindRelayError::ForwardFailed);
    }

    state
        .peer_store
        .record_route_forward_success(&next_hop, now);
    state
        .peer_store
        .record_blind_relay_forwarded(now, ttl_remaining);

    Ok(PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining,
        reason: Some("forwarded".to_string()),
    })
}

fn validate_blind_relay_envelope(
    envelope: &BlindRelayEnvelope,
    previous_hop_node_id: &[u8; 32],
) -> Result<(), BlindRelayError> {
    let previous_hop = IdentityPublicKey::from_bytes(previous_hop_node_id)
        .map_err(|_| BlindRelayError::InvalidPreviousHop)?;
    envelope
        .verify_signature_from(&previous_hop)
        .map_err(|_| BlindRelayError::InvalidSignature)?;
    encode_blind_relay_envelope(envelope).map_err(|_| BlindRelayError::EnvelopeTooLarge)?;
    Ok(())
}

fn blind_peer_relay_url(endpoint: &str) -> Option<String> {
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.is_empty() {
        return None;
    }
    let base = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else {
        format!("http://{endpoint}")
    };
    Some(format!("{base}/api/chat/peer/blind-relay"))
}

fn classify_reqwest_error(phase: &str, error: &reqwest::Error) -> String {
    if error.is_timeout() {
        return format!("{phase}_timeout");
    }
    if error.is_connect() {
        return format!("{phase}_connect");
    }
    if error.is_status() {
        if let Some(status) = error.status() {
            return format!("{phase}_http_{}", status.as_u16());
        }
        return format!("{phase}_http_status");
    }
    if error.is_decode() {
        return format!("{phase}_decode");
    }
    if error.is_body() {
        return format!("{phase}_body");
    }
    if error.is_request() {
        return format!("{phase}_request");
    }
    format!("{phase}_unknown")
}

fn validate_peer_envelope(envelope: &ChatEnvelope) -> Result<(), ChatPeerRelayError> {
    envelope
        .verify_signature()
        .map_err(|_| ChatPeerRelayError::InvalidSignature)?;

    let encoded = encode_envelope(envelope).map_err(|_| ChatPeerRelayError::Serialization)?;
    if encoded.len() > MAX_PEER_CHAT_ENVELOPE_BYTES {
        return Err(ChatPeerRelayError::EnvelopeTooLarge {
            size: encoded.len(),
        });
    }

    Ok(())
}

async fn send_envelope_to_session(
    envelope: &ChatEnvelope,
    session: &Arc<Session>,
    udp: &Arc<UdpTransport>,
) -> bool {
    let msg = MemChainMessage::ChatRelay(envelope.clone());
    let plaintext = match encode_memchain(&msg) {
        Ok(plaintext) => plaintext,
        Err(error) => {
            warn!(error = %error, "[CHAT_PEER] Failed to encode client relay message");
            return false;
        }
    };

    let crypto = DefaultTransportCrypto::new();
    let counter = session.next_tx_counter();
    let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];
    let len = match crypto.encrypt(
        &session.session_key,
        counter,
        session.id.as_bytes(),
        &plaintext,
        &mut encrypted,
    ) {
        Ok(len) => len,
        Err(error) => {
            warn!(error = %error, "[CHAT_PEER] Failed to encrypt client relay message");
            return false;
        }
    };
    encrypted.truncate(len);

    let packet = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
    let bytes = encode_data_packet(&packet).to_vec();
    udp.send(&bytes, &session.client_endpoint).await.is_ok()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::ChatContentType;
    use aeronyx_transport::UdpTransport;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    use crate::config::ChatRelayConfig;

    fn signed_envelope() -> ChatEnvelope {
        let kp = IdentityKeyPair::generate();
        let mut envelope = ChatEnvelope {
            message_id: [0x11u8; 16],
            sender: kp.public_key_bytes(),
            receiver: [0x22u8; 32],
            timestamp: 1_800_000_000,
            ciphertext: b"opaque encrypted payload".to_vec(),
            nonce: [0x33u8; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        envelope.signature = kp.sign(&envelope.sign_data());
        envelope
    }

    fn test_chat_config(path: String) -> ChatRelayConfig {
        ChatRelayConfig {
            enabled: true,
            db_path: path,
            ..ChatRelayConfig::default()
        }
    }

    #[test]
    fn validate_peer_envelope_rejects_tampered_ciphertext() {
        let mut envelope = signed_envelope();
        envelope.ciphertext.push(0x44);

        assert!(matches!(
            validate_peer_envelope(&envelope),
            Err(ChatPeerRelayError::InvalidSignature)
        ));
    }

    #[tokio::test]
    async fn peer_relay_endpoint_stores_offline_receiver_message() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("aeronyx-chat-peer-{unique}.db"));
        let relay = Arc::new(
            ChatRelayService::new(
                test_chat_config(path.to_string_lossy().to_string()),
                [7u8; 32],
            )
            .unwrap(),
        );
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let http_client = Arc::new(reqwest::Client::new());
        let envelope = signed_envelope();
        let receiver = envelope.receiver;

        let app = build_chat_peer_router(
            Some(Arc::clone(&relay)),
            sessions,
            udp,
            peer_store,
            node_identity,
            http_client,
        );
        let body = serde_json::to_vec(&PeerChatRelayRequest { envelope }).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("pending message should be readable");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_endpoint_terminal_accepts_opaque_blob_without_parsing() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let http_client = Arc::new(reqwest::Client::new());
        let opaque_blob = br#"{"looks_like":"json","must_not_be_parsed":true}"#.to_vec();
        let envelope = BlindRelayEnvelope {
            route_id: [0x41u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: opaque_blob,
            timestamp: 1_800_000_001,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let app = build_chat_peer_router(
            None,
            sessions,
            udp,
            Arc::clone(&peer_store),
            node_identity,
            http_client,
        );
        let body = serde_json::to_vec(&PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
        })
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();

        assert!(parsed.accepted);
        assert!(parsed.terminal);
        assert!(!parsed.forwarded);
        assert_eq!(parsed.ttl_remaining, 2);
        let blind_stats = peer_store.status(1_800_000_010).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 0);
        assert!(peer_store
            .recent_audit_events()
            .iter()
            .any(|event| event.action == "blind_relay_terminal"));
    }

    #[tokio::test]
    async fn blind_relay_in_flight_guard_enforces_backpressure_limit() {
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store,
            node_identity: Arc::new(IdentityKeyPair::generate()),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS)),
        };

        assert!(BlindRelayInFlightGuard::try_acquire(&state).is_none());
    }
}
