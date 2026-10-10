// ============================================
// File: crates/aeronyx-server/src/server/tests.rs
// ============================================
//! # Server tests and shared fixtures
//!
//! Owns the `server` test module: shared fixtures and log-capture helpers
//! plus the topic children under `server/tests/`.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `server.rs`; bodies unchanged.

use super::data_plane_runtime::{
    client_hello_wire_version_is_supported, handshake_rejection_class, log_handshake_rejection,
    HandshakeRejectionClass,
};
use crate::services::memchain::{
    CustodyAuditWitnessPolicyReadiness, CustodyAuditWitnessReceiptPolicyEvidence,
    CustodyAuditWitnessReceiptReadinessSnapshot, CustodyAuditWitnessReceiptVaultAudit,
};

// [DISCOVERY-GOSSIP-RUNTIME 2026-09-25 by Codex] Keep existing server
// compatibility tests against the extracted runtime projections.
use super::discovery_gossip_runtime::{
    DirectoryProofGossipOutcome, DirectoryProofGossipPeerState, DirectoryProofGossipResult,
    DiscoveryGossipExecution, DiscoveryGossipFailure, DiscoveryGossipFailureKind,
    DiscoveryGossipPhase, DiscoveryGossipRoundAccumulator, DiscoveryGossipSampleRequest,
    DiscoveryPeerGossipReport, DiscoveryPeerIdentityHints,
};

use super::heartbeat_status::peer_store_heartbeat_status_value;
use super::memchain_storage_gate::MemChainDispatchGateError;
use super::peer_http::MEMCHAIN_SYNC_HTTP_PROFILE;
use super::{
    await_commitment_tip_announcement_or_newer, commitment_coordinator_lease_degraded_retry_delay,
    commitment_coordinator_lease_production_valid_for, commitment_follower_success_retry_delay,
    commitment_witness_startup_decision, custody_witness_auto_renewal_due,
    custody_witness_readiness_decision, custody_witness_renewal_status,
    custody_witness_renewal_warning_window_secs, custody_witness_runtime_audit_interval_secs,
    custody_witness_runtime_failure, data_plane_receive_failure_action,
    discovery_heartbeat_status_value, memchain_index_rejection_reason,
    open_endpoint_attestation_inbox, open_endpoint_evidence_store, prefix_to_netmask,
    required_runtime_supervisor_channel_closed, retry_required_data_plane_receive,
    take_pre_ready_runtime_failure, unix_now_secs, CommitmentCoordinatorLeaseRound,
    CommitmentFollowerRoundOutcome, CommitmentSyncTaskLivenessGuard,
    CommitmentTipAnnouncementWaitOutcome, CommitmentWitnessStartupBlockReason,
    CommitmentWitnessStartupDecision, CriticalRuntimeFailure, CustodyWitnessAuditEvidence,
    CustodyWitnessReadinessBlockReason, CustodyWitnessRenewalLogAction,
    CustodyWitnessRenewalLogState, CustodyWitnessRenewalRetryAction,
    CustodyWitnessRenewalRetrySchedule, CustodyWitnessRenewalRetryState,
    CustodyWitnessRenewalStatus, CustodyWitnessRuntimeTelemetry, DataPlaneReceiveFailureAction,
    DirectoryChainStore, MemChainStorageRequirement, PeerHttpClients, PeerStoreCacheDocument,
    PeerStoreCachePersistOutcome, PeerStoreVerifiedClientDeliveryAnchor,
    PeerStoreVerifiedClientDeliveryAnchorState, PeerStoreVerifiedClientDeliveryCacheEvidence,
    PeerStoreVerifiedClientDeliveryExternalWitnessDecision, RequiredApiListenerExit,
    RuntimeTaskRegistry, RuntimeTaskShutdownOutcome, RuntimeTaskShutdownReport, Server,
    SystemdNotifier, BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS,
    BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS, BLIND_RELAY_STARTUP_WARMUP_MAX_CANDIDATES,
    COORDINATOR_LEASE_PRODUCTION_SAFETY_SECS, DATA_PLANE_RECV_FAILURE_LIMIT,
    DIRECTORY_OPERATOR_HTTP_PROFILE, DIRECTORY_SYNC_HTTP_PROFILE, HTTP_TOO_EARLY_STATUS_CODE,
    ROUTEABILITY_CACHE_EVIDENCE_LEGACY_SCHEMA_VERSION, ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION,
    ROUTE_DOMAIN_CERTIFICATE_CACHE_SCHEMA_VERSION, ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION,
    THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION, TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION,
    VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT,
    VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT,
    VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION,
    VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION,
};
use crate::api::chat_peer::{
    build_chat_peer_router, PeerBlindRelayRequest, PeerBlindRelayResponse, PeerChatRelayReceiptV2,
    PeerChatRelayRequest, PeerChatRelayRequestV2, PeerChatRelayRequestV3, PeerChatRelayResponse,
    PeerChatRelayResponseV2,
};
use crate::api::directory_replica_sync::{
    DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS, DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS,
};
use crate::api::memchain_peer::{
    announce_current_record_commitment_tip_for_test, build_memchain_peer_router,
    CommitmentReconciliationOutcome,
};
use crate::api::{
    decode_bounded_json_response, read_bounded_http_response, BoundedHttpResponseError,
    PEER_ACK_RESPONSE_MAX_BYTES,
};
use crate::config_chat_relay::ChatRelayConfig;
use crate::error::{RuntimeTaskJoinFailureKind, ServerError};
use crate::services::chat_relay::{
    VerifiedSubmitAdmission, VerifiedSubmitCacheLookup, VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS,
};
use crate::services::chat_relay_anonymous_mailbox_source::ExactAnonymousMailboxTargetPin;
use aeronyx_core::crypto::handshake::{
    create_client_hello, verify_server_hello, DefaultHandshakeCrypto, HandshakeCrypto,
};
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto};
use aeronyx_core::crypto::{EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};
use aeronyx_core::ledger::{RecordCommitmentBlockV1, GENESIS_PREV_HASH};
use aeronyx_core::protocol::anonymous_mailbox::{
    encode_anonymous_mailbox_terminal_frame, AnonymousMailboxLeaseCreateV1,
    AnonymousMailboxTerminalFrameV1, AnonymousMailboxTicketIssueV1,
};
use aeronyx_core::protocol::auth::TIMESTAMP_WINDOW_SECS;
use aeronyx_core::protocol::chat::{
    encode_envelope, BlindRelayDeliveryReceipt, ChatContentType, ChatEnvelope,
};
use aeronyx_core::protocol::discovery::{
    AnonymousMailboxWorkPolicyError, DirectoryCommitmentBlockV1, DirectoryDescriptorCommitmentV1,
    DirectoryDescriptorInclusionProofV1, RouteDomainAttestationCertificateV1,
    RouteDomainAttestationV1,
};
use aeronyx_core::protocol::memchain::{
    ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1, MemChainMessage,
    CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1, CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1,
    CHAT_VERIFIED_SUBMIT_REJECTED_V1,
};
use aeronyx_core::protocol::messages::CLIENT_HELLO_SIZE;
use aeronyx_core::protocol::onion::is_onion_blob;
use aeronyx_core::protocol::{
    MessageType, NodeBootstrapSnapshot, NodeCapability, NodeCapacity, NodeDescriptor,
    NodeDiscoveryMessage, NodeProtocolFeature, OnionRoutePurpose, SignedNodeDescriptor,
    PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2,
};
use aeronyx_transport::{Transport, UdpTransport};
use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{header::CONTENT_LENGTH, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tracing_subscriber::{fmt::MakeWriter, prelude::*};

#[cfg(target_os = "linux")]
use std::os::unix::net::UnixDatagram;

use crate::api::discovery::{DiscoveryLocalCapabilityStatus, GossipResponse};
use crate::config::{DiscoveryConfig, MemChainConfig, MemChainMode, ServerConfig};
use crate::services::chat_relay_mailbox::{
    AnonymousMailboxCustodyRepository, AnonymousMailboxStoreError,
    AnonymousMailboxTicketIssueOutcome,
};

// [SERVER-DECOMPOSITION-PHASE0 2026-09-14 by Codex] Keep the first
// behavior-preserving extraction limited to required-listener tests.
mod anonymous_mailbox;
mod capability_endpoint;
mod chat_pull;
mod chat_relay;
mod commitment_custody;
mod discovery_gossip;
mod handshake_http;
mod memchain;
mod peer_cache;
mod remaining;
mod runtime_supervision;
mod session_vpn;
mod startup_runtime;
mod verified_submit;

use crate::services::memchain::MemoryStorage;
use crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound;
use crate::services::{
    AofWriter, ChatRelayService, DirectoryReplicaGossipAnnouncement, DirectoryReplicaStore,
    DirectoryReplicaSyncRuntime, HandshakeLimiter, MemPool, PeerStore, PeerStoreImportReport,
    SessionManager,
};
use crate::voucher_verifier::VoucherVerifier;
use tokio::sync::Mutex as TokioMutex;

#[derive(Clone, Default)]
struct CapturedServerLogs(Arc<Mutex<Vec<u8>>>);

struct CapturedServerLogWriter(Arc<Mutex<Vec<u8>>>);

impl Write for CapturedServerLogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("captured server log mutex")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'writer> MakeWriter<'writer> for CapturedServerLogs {
    type Writer = CapturedServerLogWriter;

    fn make_writer(&'writer self) -> Self::Writer {
        CapturedServerLogWriter(Arc::clone(&self.0))
    }
}

fn capture_server_info_logs(operation: impl FnOnce()) -> String {
    let captured = CapturedServerLogs::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .without_time()
        .with_writer(captured.clone())
        .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
            // [SERVER-LOG-TEST-TARGET 2026-09-25 by Codex] The handshake
            // rejection emitter lives in the extracted data-plane module.
            metadata.level() <= &tracing::Level::INFO
                && (metadata.target().ends_with("server")
                    || metadata.target().ends_with("server::data_plane_runtime"))
        }));
    let subscriber = tracing_subscriber::registry().with(layer);
    crate::with_scoped_test_subscriber(subscriber, operation);
    let bytes = captured
        .0
        .lock()
        .expect("captured server log mutex")
        .clone();
    String::from_utf8(bytes).expect("captured server logs are UTF-8")
}

fn capture_core_handshake_debug_logs(operation: impl FnOnce()) -> String {
    let captured = CapturedServerLogs::default();
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .without_time()
        .with_writer(captured.clone())
        .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
            metadata.level() <= &tracing::Level::DEBUG
                && metadata.target().ends_with("crypto::handshake")
        }));
    let subscriber = tracing_subscriber::registry().with(layer);
    crate::with_scoped_test_subscriber(subscriber, operation);
    let bytes = captured
        .0
        .lock()
        .expect("captured core handshake log mutex")
        .clone();
    String::from_utf8(bytes).expect("captured core handshake logs are UTF-8")
}

fn test_peer_http_client() -> Arc<reqwest::Client> {
    // [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Exercise the same
    // proxy-free construction boundary as production; `Client::new()` may
    // consult host proxy state and is not valid for isolated node tests.
    Arc::new(
        super::privacy_safe_peer_http_client_builder()
            .build()
            .unwrap(),
    )
}

fn test_chat_relay_config(db_path: &std::path::Path) -> ChatRelayConfig {
    let mut config = ChatRelayConfig::default();
    config.enabled = true;
    config.db_path = db_path.to_string_lossy().into_owned();
    config
}

fn test_chat_relay_service(
    db_path: &std::path::Path,
    node_secret: [u8; 32],
) -> Arc<ChatRelayService> {
    Arc::new(
        ChatRelayService::new(test_chat_relay_config(db_path), node_secret)
            .expect("initialize test chat relay service"),
    )
}

const ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW: u64 = 1_800_020_000;

fn test_anonymous_mailbox_custody_server(enabled: bool, db_path: &std::path::Path) -> Server {
    let mut config = ServerConfig::default();
    config.memchain.chat_relay.anonymous_mailbox.enabled = enabled;
    config.memchain.chat_relay.anonymous_mailbox.db_path = db_path.to_string_lossy().into_owned();
    config
        .memchain
        .chat_relay
        .anonymous_mailbox
        .ticket_issue_work_bits = 1;
    Server::new(
        config,
        IdentityKeyPair::from_bytes(&[0xc1; 32]).expect("cleanup test identity"),
        None,
    )
}

fn test_anonymous_mailbox_ticket_issue(
    server: &Server,
    expires_at: u64,
) -> AnonymousMailboxTicketIssueV1 {
    let depositor = IdentityKeyPair::from_bytes(&[0xc2; 32]).expect("depositor");
    let reader = IdentityKeyPair::from_bytes(&[0xc3; 32]).expect("reader");
    let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
        &[0xc4; 32],
        &depositor.public_key_bytes(),
        &reader.public_key_bytes(),
        2,
        32,
        ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
        ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW + 1_000,
    );
    for nonce in 0..u64::MAX {
        let request = AnonymousMailboxTicketIssueV1::new(
            [0xc5; 16],
            [0xc6; 16],
            server.identity.public_key_bytes(),
            claims,
            ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
            expires_at,
            nonce,
        )
        .expect("ticket issue request");
        if request.proof_digest().expect("proof digest")[0] & 0x80 == 0 {
            return request;
        }
    }
    unreachable!("one-bit work proof is reachable")
}

// [ANONYMOUS-MAILBOX-SOURCE-STARTUP-ACCEPTANCE 2026-09-05 by Codex]
// These fixtures call the production coordinator initializer directly, but
// own only a synthetic temporary path and an empty in-memory peer view.
// They neither start Server::run nor contact a peer.
fn test_anonymous_mailbox_source_server(enabled: bool, db_path: &std::path::Path) -> Server {
    let mut config = ServerConfig::default();
    config.memchain.chat_relay.anonymous_mailbox_source.enabled = enabled;
    config.memchain.chat_relay.anonymous_mailbox_source.db_path =
        db_path.to_string_lossy().into_owned();
    config
        .memchain
        .chat_relay
        .anonymous_mailbox_source
        .terminal_retention_secs = 10;
    Server::new(
        config,
        IdentityKeyPair::from_bytes(&[0xd1; 32]).expect("source cleanup identity"),
        None,
    )
}

fn test_anonymous_mailbox_source_descriptor(
    target: &IdentityKeyPair,
    now: u64,
) -> SignedNodeDescriptor {
    let mut descriptor = NodeDescriptor::new(
        target.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now + 1_000,
        "source-cleanup-target",
    )
    .with_x25519_kem(target.x25519_public_key_bytes())
    .with_protocol_features([
        NodeProtocolFeature::AnonymousMailboxV1,
        NodeProtocolFeature::OnionReplyV1,
        NodeProtocolFeature::BlindRelaySuccessReceiptV1,
        NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
    ]);
    descriptor.public_endpoint = Some("https://1.1.1.1:443".into());
    descriptor.capabilities = vec![NodeCapability::ChatRelay];
    SignedNodeDescriptor::sign(descriptor, target).expect("source cleanup descriptor")
}

fn test_anonymous_mailbox_source_terminal_frame(
    target: &IdentityKeyPair,
    request_byte: u8,
) -> Vec<u8> {
    let request = AnonymousMailboxTicketIssueV1::new(
        [request_byte; 16],
        [request_byte.wrapping_add(1); 16],
        target.public_key_bytes(),
        [request_byte.wrapping_add(2); 32],
        ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
        ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW + 30,
        u64::from(request_byte),
    )
    .expect("source cleanup request");
    encode_anonymous_mailbox_terminal_frame(&AnonymousMailboxTerminalFrameV1::TicketIssue(request))
        .expect("source cleanup terminal frame")
}

fn test_verified_submit_request(
    sender: &IdentityKeyPair,
    request_id: [u8; 16],
    message_id: [u8; 16],
    request_timestamp: u64,
    marker: u8,
) -> ChatRelayVerifiedSubmitRequestV1 {
    let mut envelope = ChatEnvelope {
        message_id,
        sender: sender.public_key_bytes(),
        receiver: [marker.wrapping_add(1); 32],
        timestamp: request_timestamp,
        ciphertext: vec![marker; 32],
        nonce: [marker.wrapping_add(2); 24],
        content_type: ChatContentType::Text,
        signature: [0_u8; 64],
    };
    envelope.signature = sender.sign(&envelope.sign_data());
    ChatRelayVerifiedSubmitRequestV1::signed(request_id, envelope, request_timestamp, sender)
        .expect("sign test verified submit request")
}

fn test_verified_submit_session(
    sender: &IdentityKeyPair,
    marker: u8,
) -> Arc<crate::services::Session> {
    Arc::new(crate::services::Session::new(
        aeronyx_common::types::SessionId::generate(),
        sender.public_key(),
        aeronyx_core::crypto::SessionKey::from_bytes([marker; 32]),
        Ipv4Addr::new(100, 64, 0, marker.max(1)),
        format!("127.0.0.1:{}", 10_000_u16 + u16::from(marker))
            .parse()
            .expect("parse test verified submit endpoint"),
    ))
}

fn seed_completed_verified_submit(
    relay: &ChatRelayService,
    request: &ChatRelayVerifiedSubmitRequestV1,
) -> ChatRelayVerifiedSubmitResponseV1 {
    assert_eq!(
        relay
            .reserve_verified_submit(request)
            .expect("reserve completed verified submit fixture"),
        VerifiedSubmitAdmission::Reserved
    );
    let response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
        request.request_id,
        request.envelope.message_id,
        false,
        true,
        None,
    );
    relay
        .remember_verified_submit_response(request, &response)
        .expect("persist completed verified submit fixture");
    response
}

fn assert_no_verified_submit_delivery_effects(relay: &ChatRelayService, sender: &IdentityKeyPair) {
    assert_eq!(
        relay
            .storage_usage()
            .expect("read verified submit storage usage")
            .pending_messages,
        0
    );
    assert!(relay
        .wallet_routes
        .lookup(&sender.public_key_bytes())
        .is_empty());
    assert_eq!(relay.peer_status().outbound_rounds, 0);
}

// [CHAT-PULL-ROUTE-AUTHORITY 2026-10-01 by Codex] Exercise operation
// authorization through the real dispatcher and encrypted UDP replies.
// Sessions are fixtures, not a claim to exercise the handshake here.
struct PullRouteFixture {
    relay: Arc<ChatRelayService>,
    wallet: IdentityKeyPair,
    node: IdentityKeyPair,
    session: Arc<crate::services::Session>,
    sessions: Arc<SessionManager>,
    client: Arc<UdpTransport>,
    server: Arc<UdpTransport>,
    _directory: tempfile::TempDir,
}

impl PullRouteFixture {
    async fn new(same_identity: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let relay = test_chat_relay_service(&directory.path().join("pull.sqlite3"), [0x67; 32]);
        let wallet = IdentityKeyPair::generate();
        let transport = if same_identity {
            wallet.clone()
        } else {
            IdentityKeyPair::generate()
        };
        let client = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let server = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let sessions = Arc::new(SessionManager::new(4, Duration::from_secs(60)));
        let session = sessions
            .create(
                aeronyx_common::types::SessionId::generate(),
                transport.public_key(),
                aeronyx_core::crypto::SessionKey::from_bytes([0x68; 32]),
                Ipv4Addr::new(100, 64, 0, 86),
                client.local_addr().unwrap(),
            )
            .unwrap();
        Self {
            relay,
            wallet,
            node: IdentityKeyPair::generate(),
            session,
            sessions,
            client,
            server,
            _directory: directory,
        }
    }

    fn pull(&self, v2: bool) -> MemChainMessage {
        use aeronyx_core::protocol::auth::{
            signed_message_digest, DOMAIN_CHAT_PULL, DOMAIN_CHAT_PULL_V2,
        };
        let wallet = self.wallet.public_key_bytes();
        let now = unix_now_secs();
        let after = 0u64.to_le_bytes();
        let limit = 1u32.to_le_bytes();
        let timestamp = now.to_le_bytes();
        if v2 {
            let cursor_len = 0u16.to_le_bytes();
            let signature = self.wallet.sign(&signed_message_digest(
                DOMAIN_CHAT_PULL_V2,
                &[&wallet, &after, &cursor_len, &[], &limit, &timestamp],
            ));
            MemChainMessage::ChatPullV2 {
                wallet,
                after_timestamp: 0,
                cursor: Vec::new(),
                limit: 1,
                request_timestamp: now,
                signature,
            }
        } else {
            let cursor = [0; 16];
            let signature = self.wallet.sign(&signed_message_digest(
                DOMAIN_CHAT_PULL,
                &[&wallet, &after, &cursor, &limit, &timestamp],
            ));
            MemChainMessage::ChatPull {
                wallet,
                after_timestamp: 0,
                cursor,
                limit: 1,
                request_timestamp: now,
                signature,
            }
        }
    }

    async fn dispatch(&self, message: MemChainMessage) {
        let mut config = MemChainConfig::default();
        config.mode = MemChainMode::Off;
        Server::handle_memchain_message(
            message,
            None,
            None,
            &None,
            &None,
            &config,
            "unused",
            &self.session,
            &self.server,
            &DefaultTransportCrypto::new(),
            &self.sessions,
            &Some(Arc::clone(&self.relay)),
            &Arc::new(PeerStore::new()),
            &self.node.public_key_bytes(),
            &self.node,
            None,
        )
        .await;
    }

    async fn receive_pull(&self, v2: bool) -> Vec<ChatEnvelope> {
        let mut datagram = vec![0; 65_535];
        let (len, _) =
            tokio::time::timeout(Duration::from_secs(2), self.client.recv(&mut datagram))
                .await
                .expect("bounded pull response")
                .unwrap();
        let packet = aeronyx_core::protocol::codec::decode_data_packet(&datagram[..len]).unwrap();
        assert_eq!(packet.session_id, *self.session.id.as_bytes());
        let mut clear = vec![0; packet.encrypted_payload.len()];
        let len = DefaultTransportCrypto::new()
            .decrypt(
                &self.session.session_key,
                packet.counter,
                self.session.id.as_bytes(),
                &packet.encrypted_payload,
                &mut clear,
            )
            .unwrap();
        let response = aeronyx_core::protocol::memchain::decode_memchain(&clear[1..len]).unwrap();
        match response {
            MemChainMessage::ChatPullResponse { envelopes, .. } if !v2 => envelopes,
            MemChainMessage::ChatPullResponseV2 { envelopes, .. } if v2 => envelopes,
            _ => panic!("unexpected pull response kind"),
        }
    }
}

async fn assert_cross_identity_pull_route_authority(v2: bool, existing: bool) {
    let fixture = PullRouteFixture::new(false).await;
    let wallet = fixture.wallet.public_key_bytes();
    let sender = IdentityKeyPair::generate();
    let mut envelope =
        test_verified_submit_request(&sender, [0x71; 16], [0x72; 16], unix_now_secs(), 0x73)
            .envelope;
    envelope.receiver = wallet;
    envelope.signature = sender.sign(&envelope.sign_data());
    fixture.relay.store_pending(&envelope).unwrap();
    if existing {
        // An existing delegated route must not have its endpoint rewritten
        // by a portable Pull signature. announce also refreshes its TTL.
        assert!(fixture.relay.wallet_routes.announce(
            &wallet,
            fixture.session.id.clone(),
            "127.0.0.1:9".parse().unwrap()
        ));
    }
    let before = fixture.relay.wallet_routes.lookup(&wallet);
    fixture.dispatch(fixture.pull(v2)).await;
    let received = fixture.receive_pull(v2).await;
    assert_eq!(
        received.len(),
        1,
        "cross-identity signed query remains available"
    );
    assert!(encode_envelope(&received[0]).unwrap() == encode_envelope(&envelope).unwrap());
    assert!(
        fixture.relay.wallet_routes.lookup(&wallet) == before,
        "portable Pull signature must not create or refresh a cross-identity route"
    );
}

/// Captures exact v3 ACK bodies and truncates the first one after custody.
///
/// [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] This test-only middleware
/// runs outside the real target router. `next.run` therefore completes
/// SQLite custody and replay-cache publication before the injected stream
/// error reaches the source node.
#[derive(Default)]
struct DirectRelayAckLossInjection {
    successful_ack_bodies: Mutex<Vec<Vec<u8>>>,
    successful_acks_seen: AtomicUsize,
}

async fn truncate_first_direct_relay_ack(
    State(injection): State<Arc<DirectRelayAckLossInjection>>,
    request: Request,
    next: Next,
) -> Response {
    let targets_v3 = request.uri().path() == "/api/chat/peer/relay-v3";
    let response = next.run(request).await;
    if !targets_v3 || !response.status().is_success() {
        return response;
    }

    let (mut parts, body) = response.into_parts();
    let body = to_bytes(body, PEER_ACK_RESPONSE_MAX_BYTES)
        .await
        .expect("real target ACK must remain bounded");
    injection
        .successful_ack_bodies
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(body.to_vec());
    let ack_index = injection
        .successful_acks_seen
        .fetch_add(1, AtomicOrdering::SeqCst);
    parts.headers.insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&body.len().to_string()).expect("bounded ACK length header"),
    );
    if ack_index > 0 {
        return Response::from_parts(parts, Body::from(body));
    }

    let prefix_len = body.len().saturating_div(2).max(1).min(body.len());
    let prefix = body.slice(..prefix_len);
    let truncated = futures::stream::iter(vec![
        Ok::<_, std::io::Error>(prefix),
        Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "injected custody ACK loss",
        )),
    ]);
    Response::from_parts(parts, Body::from_stream(truncated))
}

fn signed_test_chat_envelope(now: u64) -> ChatEnvelope {
    let sender = IdentityKeyPair::generate();
    let mut envelope = ChatEnvelope {
        message_id: [0x55; 16],
        sender: sender.public_key_bytes(),
        receiver: [0x66; 32],
        timestamp: now,
        ciphertext: b"opaque encrypted payload".to_vec(),
        nonce: [0x77; 24],
        content_type: ChatContentType::Text,
        signature: [0; 64],
    };
    envelope.signature = sender.sign(&envelope.sign_data());
    envelope
}

fn signed_chat_relay_peer_descriptor(
    endpoint: String,
    sequence: u64,
    expires_at: u64,
) -> SignedNodeDescriptor {
    signed_chat_relay_peer_descriptor_with_features(endpoint, sequence, expires_at, &[])
}

fn signed_chat_relay_peer_descriptor_with_features(
    endpoint: String,
    sequence: u64,
    expires_at: u64,
    features: &[NodeProtocolFeature],
) -> SignedNodeDescriptor {
    let peer_identity = IdentityKeyPair::generate();
    signed_chat_relay_peer_descriptor_for_identity(
        endpoint,
        sequence,
        expires_at,
        features,
        &peer_identity,
    )
}

fn signed_chat_relay_peer_descriptor_for_identity(
    endpoint: String,
    sequence: u64,
    expires_at: u64,
    features: &[NodeProtocolFeature],
    peer_identity: &IdentityKeyPair,
) -> SignedNodeDescriptor {
    // [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Receipt tests need the
    // signed descriptor and HTTP ACK to share one explicit target identity.
    let mut descriptor = NodeDescriptor::new(
        peer_identity.public_key_bytes(),
        sequence,
        sequence,
        expires_at,
        "test-peer",
    )
    .with_protocol_features(features.iter().copied());
    descriptor.public_endpoint = Some(endpoint);
    descriptor.capabilities = vec![NodeCapability::ChatRelay];
    descriptor.capacity = NodeCapacity {
        max_sessions: 32,
        max_bps: None,
        max_pps: None,
    };
    SignedNodeDescriptor::sign(descriptor, peer_identity).unwrap()
}

fn signed_probe_peer_descriptor(
    endpoint: String,
    sequence: u64,
    issued_at: u64,
    expires_at: u64,
    capabilities: Vec<NodeCapability>,
    kem_public: [u8; 32],
) -> SignedNodeDescriptor {
    let peer_identity = IdentityKeyPair::generate();
    let mut descriptor = NodeDescriptor::new(
        peer_identity.public_key_bytes(),
        sequence,
        issued_at,
        expires_at,
        "test-onion-peer",
    )
    .with_x25519_kem(kem_public);
    descriptor.public_endpoint = Some(endpoint);
    descriptor.capabilities = capabilities;
    descriptor.capacity = NodeCapacity {
        max_sessions: 32,
        max_bps: None,
        max_pps: None,
    };
    SignedNodeDescriptor::sign(descriptor, &peer_identity).unwrap()
}

const DIRECT_RELAY_RESTART_DRILL_STAGE_ENV: &str = "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_STAGE";
const DIRECT_RELAY_RESTART_DRILL_DB_ENV: &str = "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_DB";
const DIRECT_RELAY_RESTART_DRILL_V3_ENV: &str = "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_V3";
const DIRECT_RELAY_RESTART_DRILL_V2_ENV: &str = "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_V2";
// [ARCH-SPLIT-VERIFY 2026-10-02 by Codex] Match the worker's extracted test module.
const DIRECT_RELAY_RESTART_DRILL_WORKER: &str =
    "server::tests::chat_relay::target_bound_v3_restart_drill_subprocess_worker";
const DIRECT_RELAY_RESTART_DRILL_CRASH_EXIT_CODE: i32 = 73;

async fn run_direct_relay_restart_drill_child(
    stage: &str,
    db_path: &std::path::Path,
    v3_endpoint: Option<&str>,
    v2_endpoint: Option<&str>,
) -> std::process::Output {
    // [DIRECT-RELAY-CRASH-DRILL 2026-08-15 by Codex] Re-execute this exact
    // test binary so each phase owns a genuinely fresh process and no
    // process-local relay state can cross the restart boundary. Reuse the
    // production child isolation policy so the drill cannot inherit
    // systemd readiness, watchdog, or socket-activation authority.
    let executable = std::env::current_exe().expect("resolve restart drill test binary");
    let mut child = crate::isolated_child_command(executable);
    child
        .arg(DIRECT_RELAY_RESTART_DRILL_WORKER)
        .arg("--exact")
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        // [CHILD-SELECTION-GUARD 2026-10-02 by Codex] Freeze parsed output.
        .arg("--format=pretty")
        .arg("--color=never")
        .env(DIRECT_RELAY_RESTART_DRILL_STAGE_ENV, stage)
        .env(DIRECT_RELAY_RESTART_DRILL_DB_ENV, db_path)
        .kill_on_drop(true);
    if let Some(endpoint) = v3_endpoint {
        child.env(DIRECT_RELAY_RESTART_DRILL_V3_ENV, endpoint);
    }
    if let Some(endpoint) = v2_endpoint {
        child.env(DIRECT_RELAY_RESTART_DRILL_V2_ENV, endpoint);
    }
    tokio::time::timeout(Duration::from_secs(20), child.output())
        .await
        .expect("restart drill child exceeded its bounded deadline")
        .expect("start restart drill child")
}

fn assert_restart_drill_child_crashed(output: &std::process::Output) {
    assert_eq!(
        output.status.code(),
        Some(DIRECT_RELAY_RESTART_DRILL_CRASH_EXIT_CODE),
        "seed worker missed the intentional crash boundary with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_restart_drill_child_succeeded(stage: &str, output: &std::process::Output) {
    assert!(
        output.status.success() && restart_drill_selected_one_worker(&output.stdout),
        "{stage} worker rejected durable restart invariants with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// [CHILD-SELECTION-GUARD 2026-10-02 by Codex] An exit-zero libtest
// process may have selected no tests. Check the named result and exact
// counts on NORMAL completion only; crash phases deliberately lack them.
// This bounds parsing, not the pre-existing child output capture.
fn restart_drill_selected_one_worker(bytes: &[u8]) -> bool {
    if bytes.len() > 64 * 1024 {
        return false;
    }
    let Ok(output) = std::str::from_utf8(bytes) else {
        return false;
    };
    if output
        .lines()
        .filter(|line| *line == "running 1 test")
        .count()
        != 1
    {
        return false;
    }
    let expected = format!("test {} ... ok", DIRECT_RELAY_RESTART_DRILL_WORKER);
    let mut results = output
        .lines()
        .filter(|line| line.starts_with("test ") && !line.starts_with("test result:"));
    if results.next() != Some(expected.as_str()) || results.next().is_some() {
        return false;
    }
    let mut summaries = output
        .lines()
        .filter(|line| line.starts_with("test result:"));
    let Some(summary) = summaries.next() else {
        return false;
    };
    if summaries.next().is_some() {
        return false;
    }
    let Some(tail) =
        summary.strip_prefix("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; ")
    else {
        return false;
    };
    let Some((filtered, elapsed)) = tail.split_once(" filtered out; finished in ") else {
        return false;
    };
    filtered.parse::<u64>().is_ok()
        && elapsed
            .strip_suffix('s')
            .and_then(|value| value.parse::<f64>().ok())
            .is_some_and(|value| value.is_finite() && value >= 0.0)
}

#[test]
fn successful_child_selection_requires_one_named_test() {
    let valid = format!(
        "\nrunning 1 test\ntest {} ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2170 filtered out; finished in 0.01s\n",
        DIRECT_RELAY_RESTART_DRILL_WORKER,
    );
    assert!(restart_drill_selected_one_worker(valid.as_bytes()));
    assert!(restart_drill_selected_one_worker(
        valid.replace("2170 filtered", "0 filtered").as_bytes()
    ));
    for invalid in [
        valid.replace(DIRECT_RELAY_RESTART_DRILL_WORKER, "wrong::worker"),
        valid.replace("1 passed", "0 passed"),
        valid.replace("1 passed", "11 passed"),
        valid.replace("0 failed", "1 failed"),
        valid.replace("0 ignored", "1 ignored"),
        valid.replace("running 1 test", "running 0 tests"),
        valid.replace(" ... ok", " ... ignored"),
        valid.replace("test result: ok.", "test result: FAILED."),
        valid.replace("0.01s", "NaNs"),
        valid.replace("2170 filtered", "not-a-count filtered"),
        format!("{valid}{valid}"),
        "running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 2171 filtered out; finished in 0.00s\n".into(),
        "Usage: test-binary [OPTIONS]\n".into(),
        format!("{}: test\n1 test, 0 benchmarks\n", DIRECT_RELAY_RESTART_DRILL_WORKER),
        format!("running 1 test\ntest {} ... ok\n", DIRECT_RELAY_RESTART_DRILL_WORKER),
    ] {
        assert!(!restart_drill_selected_one_worker(invalid.as_bytes()));
    }
    assert!(!restart_drill_selected_one_worker(&[0xff]));
    assert!(!restart_drill_selected_one_worker(&vec![
        b' ';
        64 * 1024 + 1
    ]));
}

#[tokio::test]
async fn successful_child_selection_rejects_real_zero_test_exit() {
    // [CHILD-SELECTION-GUARD 2026-10-02 by Codex] Calibrate the gate
    // with an actual exit-zero child that runs no worker, not --list.
    let mut child = crate::isolated_child_command(
        std::env::current_exe().expect("resolve child selection test binary"),
    );
    child
        .arg("__aeronyx_intentionally_missing_worker_selection_guard__")
        .arg("--exact")
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .arg("--format=pretty")
        .arg("--color=never")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(20), child.output())
        .await
        .expect("zero-test child exceeded bounded deadline")
        .expect("start zero-test child");
    assert!(output.status.success(), "negative fixture must exit zero");
    assert!(String::from_utf8_lossy(&output.stdout).contains("running 0 tests"));
    assert!(std::panic::catch_unwind(|| {
        assert_restart_drill_child_succeeded("known zero-test negative", &output);
    })
    .is_err());
}

fn open_direct_relay_restart_drill_circuit(relay: &ChatRelayService, first_failure_at: u64) {
    for offset in 0..3 {
        let observed_at = first_failure_at.saturating_add(offset);
        let permit = relay
            .begin_direct_peer_delivery(observed_at)
            .expect("closed circuit should admit failure seed");
        let allows_more =
            relay.complete_direct_peer_delivery(observed_at, permit, false, false, true);
        assert_eq!(allows_more, offset < 2);
    }
}

fn open_direct_relay_restart_drill_circuit_at_recovery(relay: &ChatRelayService, recovery_at: u64) {
    let cooldown = relay
        .peer_status()
        .direct_peer_retry
        .circuit
        .cooldown_seconds;
    let first_failure_at = recovery_at.saturating_sub(cooldown.saturating_add(2));
    open_direct_relay_restart_drill_circuit(relay, first_failure_at);
}

fn terminate_direct_relay_restart_drill_process() -> ! {
    // `process::exit` intentionally skips Rust destructors while preserving
    // committed SQLite transactions. This is a bounded crash injection,
    // not a graceful service recreation.
    std::process::exit(DIRECT_RELAY_RESTART_DRILL_CRASH_EXIT_CODE);
}

fn seed_direct_relay_restart_drill_crash(relay: &ChatRelayService) -> ! {
    let now = unix_now_secs();
    relay
        .store_pending(&signed_test_chat_envelope(now))
        .expect("persist encrypted custody before crash");
    open_direct_relay_restart_drill_circuit(relay, now.saturating_sub(2));
    let status = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(status.state, "open");
    assert!(status.restart_protected);

    // [DIRECT-RELAY-CRASH-DRILL 2026-08-15 by Codex] `process::exit`
    // crosses a real process boundary after both safety writes commit.
    terminate_direct_relay_restart_drill_process();
}

fn seed_direct_relay_half_open_crash(relay: &ChatRelayService) -> ! {
    let now = unix_now_secs();
    open_direct_relay_restart_drill_circuit_at_recovery(relay, now);

    let permit = relay
        .begin_direct_peer_delivery(now)
        .expect("expired cooldown should commit one half-open lease");
    assert!(permit.is_half_open());
    let circuit = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(circuit.state, "half_open");
    assert_eq!(circuit.half_open_attempted_total, 1);
    assert_eq!(circuit.half_open_failed_total, 0);
    assert!(circuit.restart_protected);

    // [DIRECT-RELAY-HALF-OPEN-CRASH 2026-08-15 by Codex] Deliberately
    // abandon the durable lease without cancellation or completion.
    terminate_direct_relay_restart_drill_process();
}

fn seed_direct_relay_half_open_progress_crash(relay: &ChatRelayService) -> ! {
    let now = unix_now_secs();
    open_direct_relay_restart_drill_circuit_at_recovery(relay, now);

    let permit = relay
        .begin_direct_peer_delivery(now)
        .expect("expired cooldown should commit the first recovery probe");
    assert!(permit.is_half_open());
    assert!(relay.complete_direct_peer_delivery(now, permit, false, true, false));
    let circuit = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(circuit.state, "half_open");
    assert_eq!(circuit.opened_total, 1);
    assert_eq!(circuit.half_open_consecutive_successes, 1);
    assert_eq!(circuit.half_open_attempted_total, 1);
    assert_eq!(circuit.half_open_succeeded_total, 1);
    assert_eq!(circuit.half_open_failed_total, 0);
    assert_eq!(circuit.recovered_total, 0);
    assert!(circuit.restart_protected);

    // [DIRECT-RELAY-HALF-OPEN-PROGRESS 2026-08-15 by Codex] Terminate only
    // after the successful completion transition has reached SQLite.
    terminate_direct_relay_restart_drill_process();
}

fn seed_direct_relay_checkpoint_table_loss_crash(
    relay: &ChatRelayService,
    db_path: &std::path::Path,
) -> ! {
    let circuit = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(circuit.state, "closed");
    assert!(circuit.restart_protected);

    let conn = rusqlite::Connection::open(db_path).expect("open relay database for crash drill");
    conn.execute("DROP TABLE relay_direct_peer_circuit_checkpoint", [])
        .expect("remove installed checkpoint table before crash");
    drop(conn);

    // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] Exit without
    // destructors after SQLite commits the destructive schema change.
    terminate_direct_relay_restart_drill_process();
}

fn verify_missing_checkpoint_table_restart(db_path: &std::path::Path) {
    assert!(matches!(
        ChatRelayService::new(test_chat_relay_config(db_path), [0x79; 32]),
        Err(
            crate::services::chat_relay::ChatRelayError::CorruptStoredData {
                field: "direct_peer_circuit_checkpoint_table"
            }
        )
    ));
}

async fn verify_direct_relay_restart_drill(relay: &ChatRelayService) {
    let circuit = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(circuit.state, "open");
    assert!(circuit.restart_protected);
    assert!(circuit.checkpoint_loaded_at.is_some());
    assert!(circuit.checkpoint_persisted_at.is_some());

    let receiver = [0x66; 32];
    let (pending, has_more) = relay
        .pull_pending(&receiver, 0, &[0; 16], 2)
        .expect("recover encrypted mailbox after restart");
    assert_eq!(pending.len(), 1);
    assert!(!has_more);
    let recovered_envelope = pending[0].envelope.clone();
    relay
        .store_pending(&recovered_envelope)
        .expect("exact post-restart retry remains idempotent");
    let (after_retry, has_more) = relay
        .pull_pending(&receiver, 0, &[0; 16], 2)
        .expect("read idempotent mailbox after restart");
    assert_eq!(after_retry.len(), 1);
    assert!(!has_more);

    let v3_endpoint =
        std::env::var(DIRECT_RELAY_RESTART_DRILL_V3_ENV).expect("restart drill v3 endpoint");
    let v2_endpoint =
        std::env::var(DIRECT_RELAY_RESTART_DRILL_V2_ENV).expect("restart drill v2 endpoint");
    let now = unix_now_secs();
    let v3_descriptor = signed_chat_relay_peer_descriptor_for_identity(
        v3_endpoint,
        now.saturating_sub(1),
        now.saturating_add(300),
        &[NodeProtocolFeature::DirectPeerRelayTargetBindingV3],
        &IdentityKeyPair::generate(),
    );
    let v2_descriptor = signed_chat_relay_peer_descriptor_for_identity(
        v2_endpoint,
        now.saturating_sub(1),
        now.saturating_add(300),
        &[NodeProtocolFeature::DirectPeerRelayAuthV2],
        &IdentityKeyPair::generate(),
    );
    let v3_node_id = v3_descriptor.node_id();
    let v2_node_id = v2_descriptor.node_id();
    let peer_store = PeerStore::new();
    peer_store
        .upsert_verified(v3_descriptor, now)
        .expect("install restart drill v3 descriptor");
    peer_store
        .upsert_verified(v2_descriptor, now)
        .expect("install restart drill v2 descriptor");
    peer_store.record_route_forward_success(&v3_node_id, now.saturating_sub(1));
    peer_store.record_route_forward_success(&v2_node_id, now.saturating_sub(1));

    let client = test_peer_http_client();
    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(client.as_ref()),
        Some(relay),
        &peer_store,
        &IdentityKeyPair::generate(),
        &signed_test_chat_envelope(now),
    )
    .await;
    assert_eq!(accepted, 0);
    let post_attempt = relay.peer_status();
    assert_eq!(post_attempt.direct_peer_retry.circuit.state, "open");
    assert_eq!(post_attempt.last_outbound_attempted, 0);
    assert_eq!(
        post_attempt.last_outbound_failure_reason.as_deref(),
        Some("peer_relay_circuit_open")
    );

    assert_eq!(
        relay
            .ack_messages(&[recovered_envelope.message_id], &receiver)
            .expect("ack recovered encrypted custody"),
        1
    );
    let (after_ack, has_more) = relay
        .pull_pending(&receiver, 0, &[0; 16], 1)
        .expect("verify recovered custody deletion");
    assert!(after_ack.is_empty());
    assert!(!has_more);
}

fn verify_interrupted_half_open_restart(relay: &ChatRelayService) {
    let now = unix_now_secs();
    let circuit = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(circuit.state, "open");
    assert!(circuit.restart_protected);
    assert!(circuit.checkpoint_loaded_at.is_some());
    assert!(circuit.checkpoint_persisted_at.is_some());
    assert_eq!(circuit.opened_total, 2);
    assert_eq!(circuit.half_open_attempted_total, 1);
    assert_eq!(circuit.half_open_succeeded_total, 0);
    assert_eq!(circuit.half_open_failed_total, 1);
    let remaining = circuit
        .open_remaining_seconds
        .expect("interrupted probe must restart a bounded cooldown");
    assert!(remaining > 0);
    assert!(remaining <= circuit.cooldown_seconds);

    let blocked_before = circuit.blocked_total;
    assert!(relay.begin_direct_peer_delivery(now).is_none());
    let blocked = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(blocked.state, "open");
    assert_eq!(blocked.blocked_total, blocked_before.saturating_add(1));
    assert_eq!(blocked.half_open_failed_total, 1);
}

fn resume_half_open_progress_after_restart(relay: &ChatRelayService) {
    let now = unix_now_secs();
    let restored = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(restored.state, "half_open");
    assert_eq!(restored.opened_total, 1);
    assert_eq!(restored.half_open_consecutive_successes, 1);
    assert_eq!(restored.half_open_attempted_total, 1);
    assert_eq!(restored.half_open_succeeded_total, 1);
    assert_eq!(restored.half_open_failed_total, 0);
    assert_eq!(restored.recovered_total, 0);
    assert!(restored.restart_protected);
    assert!(restored.checkpoint_loaded_at.is_some());
    assert!(restored.checkpoint_persisted_at.is_some());

    let permit = relay
        .begin_direct_peer_delivery(now)
        .expect("restored progress should admit exactly one serial probe");
    assert!(permit.is_half_open());
    assert!(relay.complete_direct_peer_delivery(now, permit, false, true, false));
    let closed = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(closed.state, "closed");
    assert_eq!(closed.opened_total, 1);
    assert_eq!(closed.half_open_consecutive_successes, 0);
    assert_eq!(closed.half_open_attempted_total, 2);
    assert_eq!(closed.half_open_succeeded_total, 2);
    assert_eq!(closed.half_open_failed_total, 0);
    assert_eq!(closed.recovered_total, 1);
    assert!(closed.restart_protected);
}

fn verify_closed_circuit_after_restart(relay: &ChatRelayService) {
    let closed = relay.peer_status().direct_peer_retry.circuit;
    assert_eq!(closed.state, "closed");
    assert_eq!(closed.opened_total, 1);
    assert_eq!(closed.half_open_consecutive_successes, 0);
    assert_eq!(closed.half_open_attempted_total, 2);
    assert_eq!(closed.half_open_succeeded_total, 2);
    assert_eq!(closed.half_open_failed_total, 0);
    assert_eq!(closed.recovered_total, 1);
    assert!(closed.open_remaining_seconds.is_none());
    assert!(closed.restart_protected);
    assert!(closed.checkpoint_loaded_at.is_some());
    assert!(closed.checkpoint_persisted_at.is_some());
}

fn directory_gossip_announcement(now: u64) -> DirectoryReplicaGossipAnnouncement {
    let producer = IdentityKeyPair::generate();
    let descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:9".to_string(), now, now + 300);
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
    let descriptor_hash = commitment.descriptor_hash;
    let block =
        DirectoryCommitmentBlockV1::new_signed(1, now, [0u8; 32], vec![commitment], &producer)
            .unwrap();
    let block_hash = block.hash();
    let proof =
        DirectoryDescriptorInclusionProofV1::from_block_at(&block, &descriptor, now).unwrap();
    DirectoryReplicaGossipAnnouncement {
        producer: producer.public_key_bytes(),
        block_hash,
        descriptor_hash,
        proof,
    }
}

fn directory_gossip_announcements_for_receiver(
    now: u64,
    receiver_producer: [u8; 32],
) -> [DirectoryReplicaGossipAnnouncement; 3] {
    // [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] Keep peer-aware
    // fallback test fixtures explicit without obscuring the test flow.
    let mut receiver_announcement = directory_gossip_announcement(now);
    receiver_announcement.producer = receiver_producer;
    let mut first_alternate = directory_gossip_announcement(now);
    first_alternate.producer = [0x92; 32];
    let mut second_alternate = directory_gossip_announcement(now);
    second_alternate.producer = [0x93; 32];
    [receiver_announcement, first_alternate, second_alternate]
}

fn gossip_execution<'a>(
    client: &'a reqwest::Client,
    peer_store: &'a PeerStore,
    directory_announcements: &'a [DirectoryReplicaGossipAnnouncement],
    now: u64,
    peer_timeout: Duration,
) -> DiscoveryGossipExecution<'a> {
    DiscoveryGossipExecution {
        client,
        peer_store,
        directory_announcements,
        peer_identity_hints: None,
        now,
        snapshot_limit: 8,
        peer_timeout,
    }
}

async fn spawn_legacy_gossip_mock(
    delay: Duration,
    calls: Arc<AtomicUsize>,
) -> (String, tokio::task::JoinHandle<()>) {
    // [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] Shared mock keeps
    // concurrency tests focused on scheduling rather than route boilerplate.
    let app = Router::new().route(
        "/api/discovery/gossip",
        post(move |Json(_message): Json<NodeDiscoveryMessage>| {
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, AtomicOrdering::SeqCst);
                tokio::time::sleep(delay).await;
                Json(GossipResponse {
                    applied: PeerStoreImportReport::empty(),
                    response: None,
                })
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, task)
}
