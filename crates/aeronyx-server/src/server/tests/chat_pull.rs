// File: crates/aeronyx-server/src/server/tests/chat_pull.rs
// Purpose: Pull authority, direct-relay custody, and UDP handshake acceptance.
// Dependencies: parent test helpers, production dispatcher/router, core crypto,
// ephemeral loopback HTTP/UDP, and private temporary SQLite repositories.
// Flow: signed sender ingress -> direct-v3 -> reopen -> signed PullV2/ACK;
// portable handler composition adds V2 handshake and inbound AEAD/replay;
// a separate non-Linux fixture also exercises the production UDP dispatcher.
// Boundary: sender ingress remains preauthenticated; orderly in-process reopen,
// NOT Server::run, discovery gossip, process crash, failover, or Linux TUN proof.
// [CHAT-CUSTODY-LIFECYCLE 2026-10-02 by Codex] Add composed acceptance without
// changing production behavior or seeding target custody through storage APIs.
// Last Modified: 2026-10-02. Originally split from server.rs `mod tests`.
use super::*;

#[tokio::test]
async fn chat_pull_route_authority_v1_cannot_create() {
    assert_cross_identity_pull_route_authority(false, false).await;
}

#[tokio::test]
async fn chat_pull_route_authority_v2_cannot_create() {
    assert_cross_identity_pull_route_authority(true, false).await;
}

#[tokio::test]
async fn chat_pull_route_authority_v1_cannot_refresh() {
    assert_cross_identity_pull_route_authority(false, true).await;
}

#[tokio::test]
async fn chat_pull_route_authority_v2_cannot_refresh() {
    assert_cross_identity_pull_route_authority(true, true).await;
}

#[tokio::test]
async fn chat_pull_route_authority_same_identity_and_invalid_signature() {
    for v2 in [false, true] {
        let fixture = PullRouteFixture::new(true).await;
        let wallet = fixture.wallet.public_key_bytes();
        let mut invalid = fixture.pull(v2);
        match &mut invalid {
            MemChainMessage::ChatPull { signature, .. }
            | MemChainMessage::ChatPullV2 { signature, .. } => signature[0] ^= 1,
            _ => unreachable!(),
        }
        fixture.dispatch(invalid).await;
        assert!(fixture.relay.wallet_routes.lookup(&wallet).is_empty());
        fixture.dispatch(fixture.pull(v2)).await;
        assert!(fixture.receive_pull(v2).await.is_empty());
        assert!(
            fixture.relay.wallet_routes.lookup(&wallet)
                == vec![(fixture.session.id.clone(), fixture.session.endpoint())]
        );
    }
}

#[tokio::test]
async fn chat_pull_route_authority_session_bound_delegation_still_works() {
    use aeronyx_core::protocol::auth::{
        signed_message_digest, DOMAIN_DEVICE_REGISTER, DOMAIN_WALLET_PRESENCE,
    };
    let fixture = PullRouteFixture::new(false).await;
    let wallet = fixture.wallet.public_key_bytes();
    let timestamp = unix_now_secs();
    let ts = timestamp.to_le_bytes();
    let device_id = [0x74; 16];
    let signature = fixture.wallet.sign(&signed_message_digest(
        DOMAIN_DEVICE_REGISTER,
        &[fixture.session.id.as_bytes(), &device_id, &wallet, &ts],
    ));
    fixture
        .dispatch(MemChainMessage::DeviceRegister {
            device_id,
            device_name: String::new(),
            wallet_pubkey: wallet,
            timestamp,
            signature,
        })
        .await;
    assert!(
        fixture.relay.wallet_routes.lookup(&wallet)
            == vec![(fixture.session.id.clone(), fixture.session.endpoint())]
    );
    assert!(fixture
        .relay
        .wallet_routes
        .remove_route(&wallet, &fixture.session.id));
    let signature = fixture.wallet.sign(&signed_message_digest(
        DOMAIN_WALLET_PRESENCE,
        &[fixture.session.id.as_bytes(), &wallet, &ts],
    ));
    fixture
        .dispatch(MemChainMessage::WalletPresence {
            wallet_pubkey: wallet,
            timestamp,
            signature,
        })
        .await;
    assert!(
        fixture.relay.wallet_routes.lookup(&wallet)
            == vec![(fixture.session.id.clone(), fixture.session.endpoint())]
    );
}

// [CHAT-CUSTODY-LIFECYCLE 2026-10-02 by Codex] Keep fixture state private and
// without Debug: test failures must not dump identities, keys, or ciphertext.
struct CustodyAcceptanceNode {
    relay: Arc<ChatRelayService>,
    identity: IdentityKeyPair,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peers: Arc<PeerStore>,
}

impl CustodyAcceptanceNode {
    async fn open(path: &std::path::Path, secret: [u8; 32], identity: IdentityKeyPair) -> Self {
        Self {
            relay: test_chat_relay_service(path, secret),
            identity,
            sessions: Arc::new(SessionManager::new(8, Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peers: Arc::new(PeerStore::new()),
        }
    }

    async fn reopen(self, path: &std::path::Path, secret: [u8; 32]) -> Self {
        let Self {
            relay,
            identity,
            sessions,
            udp,
            peers,
        } = self;
        // The router must already be stopped and joined. Prove this is an
        // actual connection close/reopen, not a second handle to the same store.
        let previous = Arc::downgrade(&relay);
        drop((sessions, udp, peers, relay));
        assert!(
            previous.upgrade().is_none(),
            "target repository still owned"
        );
        Self::open(path, secret, identity).await
    }

    async fn client(&self, identity: IdentityKeyPair) -> CustodyAcceptanceClient {
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let session = self
            .sessions
            .create(
                aeronyx_common::types::SessionId::generate(),
                identity.public_key(),
                aeronyx_core::crypto::SessionKey::from_bytes([0x91; 32]),
                Ipv4Addr::new(100, 64, 0, 91),
                udp.local_addr().unwrap(),
            )
            .unwrap();
        CustodyAcceptanceClient {
            identity,
            session,
            udp,
        }
    }

    async fn dispatch(
        &self,
        client: &CustodyAcceptanceClient,
        message: MemChainMessage,
        http: Option<&reqwest::Client>,
    ) {
        // Canonical protocol frames enter the real post-auth dispatcher. This
        // deliberately does not claim to exercise inbound transport or handshake.
        let bytes = aeronyx_core::protocol::encode_memchain(&message).unwrap();
        let message = aeronyx_core::protocol::decode_memchain(&bytes[1..]).unwrap();
        let mut config = MemChainConfig::default();
        config.mode = MemChainMode::Off;
        Server::handle_memchain_message(
            message,
            None,
            None,
            &None,
            &None,
            &config,
            "unused-chat-only",
            &client.session,
            &self.udp,
            &DefaultTransportCrypto::new(),
            &self.sessions,
            &Some(Arc::clone(&self.relay)),
            &self.peers,
            &self.identity.public_key_bytes(),
            &self.identity,
            http,
        )
        .await;
    }
}

struct CustodyAcceptanceClient {
    identity: IdentityKeyPair,
    session: Arc<crate::services::Session>,
    udp: Arc<UdpTransport>,
}

impl CustodyAcceptanceClient {
    async fn pull(&self, node: &CustodyAcceptanceNode) -> Vec<ChatEnvelope> {
        use aeronyx_core::protocol::auth::{signed_message_digest, DOMAIN_CHAT_PULL_V2};
        let wallet = self.identity.public_key_bytes();
        let now = unix_now_secs();
        let signature = self.identity.sign(&signed_message_digest(
            DOMAIN_CHAT_PULL_V2,
            &[
                &wallet,
                &0u64.to_le_bytes(),
                &0u16.to_le_bytes(),
                &[],
                &1u32.to_le_bytes(),
                &now.to_le_bytes(),
            ],
        ));
        node.dispatch(
            self,
            MemChainMessage::ChatPullV2 {
                wallet,
                after_timestamp: 0,
                cursor: Vec::new(),
                limit: 1,
                request_timestamp: now,
                signature,
            },
            None,
        )
        .await;
        let mut datagram = vec![0; 65_535];
        let (len, _) = tokio::time::timeout(Duration::from_secs(2), self.udp.recv(&mut datagram))
            .await
            .expect("bounded custody pull reply")
            .unwrap();
        let packet = aeronyx_core::protocol::codec::decode_data_packet(&datagram[..len]).unwrap();
        assert!(packet.session_id == *self.session.id.as_bytes());
        let mut clear = vec![0; packet.encrypted_payload.len()];
        let len = DefaultTransportCrypto::new()
            .decrypt(
                &self.session.session_key,
                packet.counter,
                self.session.id.as_bytes(),
                &packet.encrypted_payload,
                &mut clear,
            )
            .expect("authenticated custody reply");
        assert_eq!(clear[0], aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC);
        let MemChainMessage::ChatPullResponseV2 {
            envelopes,
            has_more,
            ..
        } = aeronyx_core::protocol::decode_memchain(&clear[1..len]).unwrap()
        else {
            panic!("unexpected custody reply kind")
        };
        assert!(!has_more);
        envelopes
    }

    fn ack(&self, message_id: [u8; 16]) -> MemChainMessage {
        use aeronyx_core::protocol::auth::{signed_message_digest, DOMAIN_CHAT_ACK};
        use sha2::{Digest, Sha256};
        let wallet = self.identity.public_key_bytes();
        let now = unix_now_secs();
        let ids_hash: [u8; 32] = Sha256::digest(message_id).into();
        let signature = self.identity.sign(&signed_message_digest(
            DOMAIN_CHAT_ACK,
            &[&wallet, &now.to_le_bytes(), &ids_hash],
        ));
        MemChainMessage::ChatAck {
            message_ids: vec![message_id],
            wallet,
            ack_timestamp: now,
            signature,
        }
    }
}

// [CHAT-CUSTODY-LIFECYCLE 2026-10-02 by Codex] Observe the real target's ACK,
// never replace it. The source must use v3 and verify a request-bound receipt.
struct CustodyHttpEvidence {
    target: [u8; 32],
    request_commitment: [u8; 32],
    v3_calls: AtomicUsize,
    legacy_calls: AtomicUsize,
    valid_receipts: AtomicUsize,
}

async fn observe_custody_http(
    State(evidence): State<Arc<CustodyHttpEvidence>>,
    request: Request,
    next: Next,
) -> Response {
    let v3 = request.uri().path() == "/api/chat/peer/relay-v3";
    if v3 {
        evidence.v3_calls.fetch_add(1, AtomicOrdering::SeqCst);
    } else if matches!(
        request.uri().path(),
        "/api/chat/peer/relay" | "/api/chat/peer/relay-v2"
    ) {
        evidence.legacy_calls.fetch_add(1, AtomicOrdering::SeqCst);
    }
    let response = next.run(request).await;
    if !v3 {
        return response;
    }
    assert!(response.status().is_success(), "target rejected custody");
    let (parts, body) = response.into_parts();
    let body = to_bytes(body, PEER_ACK_RESPONSE_MAX_BYTES).await.unwrap();
    let ack: PeerChatRelayResponseV2 = serde_json::from_slice(&body).unwrap();
    assert!(ack.relay.accepted && ack.relay.stored_pending);
    assert!(
        ack.receipt
            .as_ref()
            .expect("signed custody receipt")
            .verify_expected_commitment(
                &evidence.request_commitment,
                &evidence.target,
                unix_now_secs()
            )
            .is_ok(),
        "custody receipt did not bind request and target"
    );
    evidence.valid_receipts.fetch_add(1, AtomicOrdering::SeqCst);
    Response::from_parts(parts, Body::from(body))
}

// Drop aborts only this test's task on assertion failure. The success path
// explicitly signals graceful shutdown and joins before closing the SQLite owner.
struct CustodyTestRouter {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl CustodyTestRouter {
    async fn close(mut self) {
        self.stop
            .take()
            .unwrap()
            .send(())
            .expect("test router alive");
        tokio::time::timeout(Duration::from_secs(5), &mut self.task)
            .await
            .expect("test router graceful shutdown deadline")
            .expect("test router task")
            .expect("test router shutdown");
    }
}

impl Drop for CustodyTestRouter {
    fn drop(&mut self) {
        self.task.abort();
    }
}

enum CustodyAcceptanceCase {
    Ack,
    UnackedReopen,
    InvalidAck,
    // [CHAT-PORTABLE-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] No TUN or
    // platform gate: compose production handshake and packet services directly.
    PortableHandshake,
    // [CHAT-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] Linux's production UDP
    // task requires a real TUN; this fixture never creates privileged devices.
    #[cfg(not(target_os = "linux"))]
    TransportHandshake,
}

// [CHAT-CUSTODY-LIFECYCLE 2026-10-02 by Codex] One bounded protocol fixture
// composes production boundaries; no direct target store_pending seed is used.
async fn run_custody_acceptance(case: CustodyAcceptanceCase) {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.sqlite3");
    let target_path = directory.path().join("target.sqlite3");
    let source =
        CustodyAcceptanceNode::open(&source_path, [0x92; 32], IdentityKeyPair::generate()).await;
    let mut target =
        CustodyAcceptanceNode::open(&target_path, [0x93; 32], IdentityKeyPair::generate()).await;
    let sender = source.client(IdentityKeyPair::generate()).await;
    let receiver_identity = IdentityKeyPair::generate();
    let (sender_e2e, sender_kem) = sender
        .identity
        .e2e_handshake(&receiver_identity.x25519_public_key_bytes());
    let (receiver_e2e, _) = receiver_identity.e2e_handshake(&sender_kem);
    let plaintext = b"ephemeral custody acceptance payload";
    let nonce = [0x94; 24];
    let mut envelope = ChatEnvelope {
        message_id: [0x95; 16],
        sender: sender.identity.public_key_bytes(),
        receiver: receiver_identity.public_key_bytes(),
        timestamp: unix_now_secs(),
        ciphertext: sender_e2e.encrypt_raw(plaintext, &nonce).unwrap(),
        nonce,
        content_type: ChatContentType::Text,
        signature: [0; 64],
    };
    envelope.signature = sender.identity.sign(&envelope.sign_data());
    let target_id = target.identity.public_key_bytes();
    let commitment = PeerChatRelayRequestV3::sign(envelope.clone(), target_id, &source.identity)
        .unwrap()
        .request_commitment()
        .unwrap();
    let evidence = Arc::new(CustodyHttpEvidence {
        target: target_id,
        request_commitment: commitment,
        v3_calls: AtomicUsize::new(0),
        legacy_calls: AtomicUsize::new(0),
        valid_receipts: AtomicUsize::new(0),
    });
    let http = test_peer_http_client();
    let router = build_chat_peer_router(
        Some(Arc::clone(&target.relay)),
        Arc::clone(&target.sessions),
        Arc::clone(&target.udp),
        Arc::clone(&target.peers),
        Arc::new(target.identity.clone()),
        Arc::clone(&http),
        None,
    )
    .layer(middleware::from_fn_with_state(
        Arc::clone(&evidence),
        observe_custody_http,
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let router = CustodyTestRouter {
        stop: Some(stop),
        task: tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        }),
    };
    let now = unix_now_secs();
    source
        .peers
        .upsert_verified(
            signed_chat_relay_peer_descriptor_for_identity(
                endpoint,
                now.saturating_sub(1),
                now + 300,
                &[
                    NodeProtocolFeature::DirectPeerRelayAuthV2,
                    NodeProtocolFeature::DirectPeerRelayReceiptV2,
                    NodeProtocolFeature::DirectPeerRelayTargetBindingV3,
                ],
                &target.identity,
            ),
            now,
        )
        .unwrap();
    // Route health is explicitly SEEDED, not discovery/probe acceptance. No
    // onion receipt capability is seeded, so its zero-attempt preflight permits
    // the production compatibility direct-v3 path to the sole eligible target.
    source.peers.record_route_forward_success(&target_id, now);
    assert!(source.peers.is_routeable_now(&target_id, now));
    source
        .dispatch(
            &sender,
            MemChainMessage::ChatRelay(envelope.clone()),
            Some(http.as_ref()),
        )
        .await;
    assert_eq!(evidence.v3_calls.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(evidence.legacy_calls.load(AtomicOrdering::SeqCst), 0);
    assert_eq!(evidence.valid_receipts.load(AtomicOrdering::SeqCst), 1);
    let source_status = source.relay.peer_status();
    assert_eq!(source_status.outbound_accepted_total, 1);
    assert_eq!(source_status.outbound_failed_total, 0);
    assert_eq!(target.relay.peer_status().inbound_delivered_online_total, 0);
    assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 1);
    // There has been no receiver session on T. Local source fallback custody
    // is allowed; this is one target copy, not a global exactly-once assertion.
    router.close().await;
    target = target.reopen(&target_path, [0x93; 32]).await;
    // [CHAT-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] The receiver in this
    // branch has no injected Session and never calls the dispatcher directly.
    // [CHAT-PORTABLE-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] Both ingress
    // variants share the wire client and the complete durable custody lifecycle.
    let ingress = match case {
        CustodyAcceptanceCase::PortableHandshake => {
            Some(transport_acceptance::Ingress::PortableHandlers)
        }
        #[cfg(not(target_os = "linux"))]
        CustodyAcceptanceCase::TransportHandshake => {
            Some(transport_acceptance::Ingress::ProductionUdp)
        }
        _ => None,
    };
    if let Some(ingress) = ingress {
        transport_acceptance::run(
            target,
            &target_path,
            receiver_identity,
            &envelope,
            &receiver_e2e,
            plaintext,
            ingress,
        )
        .await;
        return;
    }
    let mut receiver = target.client(receiver_identity.clone()).await;
    assert_custody_envelope(
        &receiver.pull(&target).await,
        &envelope,
        &receiver_e2e,
        plaintext,
    );
    match case {
        CustodyAcceptanceCase::Ack => {}
        CustodyAcceptanceCase::PortableHandshake => {
            unreachable!("handled before session injection")
        }
        #[cfg(not(target_os = "linux"))]
        CustodyAcceptanceCase::TransportHandshake => {
            unreachable!("handled before session injection")
        }
        CustodyAcceptanceCase::UnackedReopen => {
            drop(receiver);
            target = target.reopen(&target_path, [0x93; 32]).await;
            receiver = target.client(receiver_identity.clone()).await;
            assert_custody_envelope(
                &receiver.pull(&target).await,
                &envelope,
                &receiver_e2e,
                plaintext,
            );
        }
        CustodyAcceptanceCase::InvalidAck => {
            let mut forged = receiver.ack(envelope.message_id);
            if let MemChainMessage::ChatAck { signature, .. } = &mut forged {
                signature[0] ^= 1;
            }
            target.dispatch(&receiver, forged, None).await;
            assert_custody_envelope(
                &receiver.pull(&target).await,
                &envelope,
                &receiver_e2e,
                plaintext,
            );
            let other = target.client(IdentityKeyPair::generate()).await;
            target
                .dispatch(&other, other.ack(envelope.message_id), None)
                .await;
            assert_custody_envelope(
                &receiver.pull(&target).await,
                &envelope,
                &receiver_e2e,
                plaintext,
            );
            drop(other);
        }
    }
    // ACK has no success response. Only an authenticated pull after a genuine
    // repository close/reopen proves durable retirement of this target copy.
    target
        .dispatch(&receiver, receiver.ack(envelope.message_id), None)
        .await;
    drop(receiver);
    target = target.reopen(&target_path, [0x93; 32]).await;
    let receiver = target.client(receiver_identity).await;
    assert!(receiver.pull(&target).await.is_empty());
    assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 0);
}

fn assert_custody_envelope(
    envelopes: &[ChatEnvelope],
    expected: &ChatEnvelope,
    e2e: &aeronyx_core::crypto::E2eSession,
    plaintext: &[u8],
) {
    assert_eq!(envelopes.len(), 1);
    let actual = &envelopes[0];
    assert!(actual.verify_signature().is_ok());
    assert!(
        aeronyx_core::protocol::encode_memchain(&MemChainMessage::ChatRelay(actual.clone()))
            .unwrap()
            == aeronyx_core::protocol::encode_memchain(&MemChainMessage::ChatRelay(
                expected.clone()
            ))
            .unwrap()
    );
    assert!(e2e.decrypt_raw(&actual.ciphertext, &actual.nonce).unwrap() == plaintext);
}

#[tokio::test]
async fn chat_custody_ingress_v3_pull_ack_survives_reopen() {
    tokio::time::timeout(
        Duration::from_secs(25),
        run_custody_acceptance(CustodyAcceptanceCase::Ack),
    )
    .await
    .expect("bounded custody lifecycle");
}

#[tokio::test]
async fn chat_custody_unacked_item_redelivers_after_reopen() {
    tokio::time::timeout(
        Duration::from_secs(25),
        run_custody_acceptance(CustodyAcceptanceCase::UnackedReopen),
    )
    .await
    .expect("bounded unacknowledged lifecycle");
}

#[tokio::test]
async fn chat_custody_invalid_acks_preserve_item_until_valid_ack() {
    tokio::time::timeout(
        Duration::from_secs(25),
        run_custody_acceptance(CustodyAcceptanceCase::InvalidAck),
    )
    .await
    .expect("bounded invalid ACK lifecycle");
}

// [CHAT-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] Private composition of the
// actual authentication services, not a substitute packet handler or preauthenticated
// receiver. Server::new only creates in-memory shutdown/telemetry state here;
// run/startup, management, API/discovery listeners, and TUN are never started.
mod transport_acceptance {
    use super::*;
    use crate::handlers::packet::DecryptedPayload;
    use crate::handlers::PacketHandler;
    #[cfg(not(target_os = "linux"))]
    use crate::management::reporter::SessionEventSender;
    use crate::services::traffic_tracker::TrafficTracker;
    use crate::services::{DenyList, HandshakeService, NodePolicyRuntime};
    use aeronyx_core::crypto::handshake::{
        create_client_hello_v2, derive_client_session_keys_v2, verify_server_hello_v2,
    };
    use aeronyx_core::crypto::kdf::SessionKeys;
    use aeronyx_core::crypto::EphemeralKeyPair;
    use aeronyx_core::protocol::codec::{
        decode_client_hello, decode_data_packet, decode_server_hello, encode_client_hello,
        encode_data_packet, encode_server_hello, ProtocolCodec,
    };
    use aeronyx_core::protocol::{DataPacket, MessageType, PROTOCOL_VERSION_V2};

    // [CHAT-PORTABLE-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] Explicitly
    // distinguish a portable service composition from the full UDP dispatcher.
    #[derive(Clone, Copy)]
    pub(super) enum Ingress {
        PortableHandlers,
        #[cfg(not(target_os = "linux"))]
        ProductionUdp,
    }

    // Own every spawned UDP task; explicit bounded join is the success path,
    // while Drop also prevents an assertion failure from leaking a listener.
    struct UdpRuntime {
        server: Server,
        packet_handler: Arc<PacketHandler>,
        udp: Arc<UdpTransport>,
        task: tokio::task::JoinHandle<()>,
    }

    impl UdpRuntime {
        fn start(node: &mut CustodyAcceptanceNode, ingress: Ingress) -> Self {
            assert_eq!(node.sessions.count(), 0);
            let mut config = ServerConfig::default();
            config.memchain.mode = MemChainMode::Off;
            let server = Server::new(config, node.identity.clone(), None);
            let (ip_pool, sessions, routing) = server.init_services().unwrap();
            node.sessions = Arc::clone(&sessions);
            let traffic = Arc::new(TrafficTracker::new());
            let policy = Arc::new(NodePolicyRuntime::default());
            let packet_handler = Arc::new(PacketHandler::new(
                Arc::clone(&sessions),
                Arc::clone(&routing),
                Arc::clone(&traffic),
                Arc::new(AtomicU64::new(0)),
                Arc::clone(&policy),
            ));
            let handshake = Arc::new(HandshakeService::new(
                node.identity.clone(),
                ip_pool,
                Arc::clone(&sessions),
                Arc::clone(&routing),
                Arc::new(DenyList::new()),
                policy,
            ));
            let task = match ingress {
                Ingress::PortableHandlers => {
                    // [CHAT-PORTABLE-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex]
                    // Test-only loop, NOT spawn_udp_task coverage: no limiter,
                    // voucher policy, TUN, or session-eviction lifecycle claim.
                    // Every receiver session still originates in HandshakeService;
                    // only PacketHandler-authenticated MemChain reaches dispatch.
                    let mut shutdown_rx = server.shutdown_tx.subscribe();
                    let udp = Arc::clone(&node.udp);
                    let handler = Arc::clone(&packet_handler);
                    let relay = Some(Arc::clone(&node.relay));
                    let peers = Arc::clone(&node.peers);
                    let identity = node.identity.clone();
                    let config = server.config.memchain.clone();
                    tokio::spawn(async move {
                        let mut buffer = vec![0; 65_535];
                        let crypto = DefaultTransportCrypto::new();
                        loop {
                            let (len, source) = tokio::select! {
                                _ = shutdown_rx.recv() => break,
                                received = udp.recv(&mut buffer) => {
                                    received.expect("portable fixture UDP receive")
                                }
                            };
                            let bytes = &buffer[..len];
                            match ProtocolCodec::classify_datagram(bytes) {
                                MessageType::ClientHello => {
                                    let hello = decode_client_hello(bytes).unwrap();
                                    assert_eq!(hello.version, PROTOCOL_VERSION_V2);
                                    assert!(&encode_client_hello(&hello)[..] == bytes);
                                    let result = handshake
                                        .process(&hello, &[], source.addr)
                                        .expect("production handshake admits receiver");
                                    assert_eq!(
                                        result.session.protocol_version,
                                        PROTOCOL_VERSION_V2
                                    );
                                    udp.send(&encode_server_hello(&result.response), &source.addr)
                                        .await
                                        .unwrap();
                                }
                                MessageType::Data => {
                                    let Ok((session, payload)) =
                                        handler.handle_udp_packet(bytes, source.addr)
                                    else {
                                        // Real drop counters prove AEAD/replay rejection.
                                        continue;
                                    };
                                    let DecryptedPayload::MemChain(message) = payload else {
                                        panic!("expected authenticated MemChain payload")
                                    };
                                    Server::handle_memchain_message(
                                        message,
                                        None,
                                        None,
                                        &None,
                                        &None,
                                        &config,
                                        "unused-chat-only",
                                        &session,
                                        &udp,
                                        &crypto,
                                        &sessions,
                                        &relay,
                                        &peers,
                                        &identity.public_key_bytes(),
                                        &identity,
                                        None,
                                    )
                                    .await;
                                }
                                _ => panic!("unexpected portable fixture datagram"),
                            }
                        }
                    })
                }
                #[cfg(not(target_os = "linux"))]
                Ingress::ProductionUdp => {
                    // Empty signed extension takes the production Missing-voucher
                    // compatibility path without HTTP. Even unexpected lookup is
                    // confined to loopback, never the configured production issuer.
                    let voucher = Arc::new(VoucherVerifier::with_issuer_keys_url(
                        "http://127.0.0.1:9/unused-test-issuer".to_owned(),
                    ));
                    server.spawn_udp_task(
                        Arc::clone(&node.udp),
                        handshake,
                        Arc::clone(&packet_handler),
                        voucher,
                        sessions,
                        SessionEventSender::disabled(),
                        None,
                        None,
                        None,
                        None,
                        server.config.memchain.clone(),
                        hex::encode(node.identity.public_key_bytes()),
                        Some(Arc::clone(&node.relay)),
                        routing,
                        Arc::clone(&node.peers),
                        test_peer_http_client(),
                        traffic,
                    )
                }
            };
            Self {
                server,
                packet_handler,
                udp: Arc::clone(&node.udp),
                task,
            }
        }

        async fn expect_drops(&self, decrypt_failed: u64, replay: u64) {
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    let drops = self.packet_handler.runtime_status().drop_reasons;
                    if drops.decrypt_failed == decrypt_failed && drops.replay == replay {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("UDP handler observed the expected authentication/replay rejection");
        }

        async fn close(mut self) {
            self.server.shutdown.store(true, AtomicOrdering::SeqCst);
            self.server
                .shutdown_tx
                .send(())
                .expect("UDP task subscribed");
            tokio::time::timeout(Duration::from_secs(5), &mut self.task)
                .await
                .expect("UDP shutdown deadline")
                .expect("UDP task joined");
            Server::shutdown_udp_transport(&self.udp).await;
        }
    }

    impl Drop for UdpRuntime {
        fn drop(&mut self) {
            self.server.shutdown.store(true, AtomicOrdering::SeqCst);
            let _ = self.server.shutdown_tx.send(());
            self.task.abort();
        }
    }

    // No Debug and no reference to server Session/keys: all transport material
    // below comes from verified wire hellos and the client's own ephemeral DH.
    struct WireClient {
        identity: IdentityKeyPair,
        session_id: [u8; 16],
        keys: SessionKeys,
        udp: UdpTransport,
        target: std::net::SocketAddr,
        tx_counter: u64,
        rx_counter: Option<u64>,
    }

    impl WireClient {
        async fn connect(node: &CustodyAcceptanceNode, identity: IdentityKeyPair) -> Self {
            let udp = UdpTransport::bind("127.0.0.1:0").await.unwrap();
            let target = node.udp.local_addr().unwrap();
            let ephemeral = EphemeralKeyPair::generate();
            let hello = create_client_hello_v2(&identity, ephemeral.public_key_bytes(), &[]);
            udp.send(&encode_client_hello(&hello), &target)
                .await
                .unwrap();
            let mut response = vec![0; 65_535];
            let (len, source) =
                tokio::time::timeout(Duration::from_secs(3), udp.recv(&mut response))
                    .await
                    .expect("bounded real ServerHello")
                    .unwrap();
            assert_eq!(source.addr, target);
            let response = decode_server_hello(&response[..len]).unwrap();
            assert_eq!(response.version, PROTOCOL_VERSION_V2);
            verify_server_hello_v2(&response, &hello, Some(&node.identity.public_key_bytes()))
                .expect("ServerHello bound to client hello and pinned node");
            let shared = ephemeral.exchange(&response.server_ephemeral_key);
            let keys = derive_client_session_keys_v2(&shared, &hello, &[], &response).unwrap();
            assert!(keys.c2s != keys.s2c, "direction keys must be distinct");
            assert_eq!(node.sessions.count(), 1, "handshake created the session");
            Self {
                identity,
                session_id: response.session_id,
                keys,
                udp,
                target,
                tx_counter: 0,
                rx_counter: None,
            }
        }

        fn packet(&mut self, message: &MemChainMessage) -> Vec<u8> {
            self.tx_counter = self.tx_counter.checked_add(1).unwrap();
            let clear = aeronyx_core::protocol::encode_memchain(message).unwrap();
            let mut sealed = vec![0; clear.len() + 16];
            let len = DefaultTransportCrypto::new()
                .encrypt(
                    &self.keys.c2s,
                    self.tx_counter,
                    &self.session_id,
                    &clear,
                    &mut sealed,
                )
                .unwrap();
            sealed.truncate(len);
            encode_data_packet(&DataPacket::new(self.session_id, self.tx_counter, sealed)).to_vec()
        }

        async fn send(&self, packet: &[u8]) {
            self.udp.send(packet, &self.target).await.unwrap();
        }

        async fn pull(&mut self) -> (Vec<ChatEnvelope>, Vec<u8>) {
            use aeronyx_core::protocol::auth::{signed_message_digest, DOMAIN_CHAT_PULL_V2};
            let wallet = self.identity.public_key_bytes();
            let now = unix_now_secs();
            let signature = self.identity.sign(&signed_message_digest(
                DOMAIN_CHAT_PULL_V2,
                &[
                    &wallet,
                    &0u64.to_le_bytes(),
                    &0u16.to_le_bytes(),
                    &[],
                    &1u32.to_le_bytes(),
                    &now.to_le_bytes(),
                ],
            ));
            let packet = self.packet(&MemChainMessage::ChatPullV2 {
                wallet,
                after_timestamp: 0,
                cursor: Vec::new(),
                limit: 1,
                request_timestamp: now,
                signature,
            });
            self.send(&packet).await;
            let mut datagram = vec![0; 65_535];
            let (len, source) =
                tokio::time::timeout(Duration::from_secs(3), self.udp.recv(&mut datagram))
                    .await
                    .expect("bounded encrypted PullV2 response")
                    .unwrap();
            assert_eq!(source.addr, self.target);
            let response = decode_data_packet(&datagram[..len]).unwrap();
            assert!(response.session_id == self.session_id);
            assert!(self
                .rx_counter
                .is_none_or(|previous| response.counter > previous));
            let mut clear = vec![0; response.encrypted_payload.len()];
            let len = DefaultTransportCrypto::new()
                .decrypt(
                    &self.keys.s2c,
                    response.counter,
                    &self.session_id,
                    &response.encrypted_payload,
                    &mut clear,
                )
                .expect("response authenticated under negotiated s2c key");
            self.rx_counter = Some(response.counter);
            assert_eq!(clear[0], aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC);
            let MemChainMessage::ChatPullResponseV2 {
                envelopes,
                has_more,
                ..
            } = aeronyx_core::protocol::decode_memchain(&clear[1..len]).unwrap()
            else {
                panic!("expected PullV2 response")
            };
            assert!(!has_more);
            (envelopes, packet)
        }

        fn ack_packet(&mut self, message_id: [u8; 16]) -> Vec<u8> {
            use aeronyx_core::protocol::auth::{signed_message_digest, DOMAIN_CHAT_ACK};
            use sha2::{Digest, Sha256};
            let wallet = self.identity.public_key_bytes();
            let now = unix_now_secs();
            let ids_hash: [u8; 32] = Sha256::digest(message_id).into();
            let signature = self.identity.sign(&signed_message_digest(
                DOMAIN_CHAT_ACK,
                &[&wallet, &now.to_le_bytes(), &ids_hash],
            ));
            self.packet(&MemChainMessage::ChatAck {
                message_ids: vec![message_id],
                wallet,
                ack_timestamp: now,
                signature,
            })
        }
    }

    pub(super) async fn run(
        mut target: CustodyAcceptanceNode,
        path: &std::path::Path,
        receiver: IdentityKeyPair,
        envelope: &ChatEnvelope,
        e2e: &aeronyx_core::crypto::E2eSession,
        plaintext: &[u8],
        ingress: Ingress,
    ) {
        let runtime = UdpRuntime::start(&mut target, ingress);
        let mut client = WireClient::connect(&target, receiver.clone()).await;
        let (items, replay_packet) = client.pull().await;
        assert_custody_envelope(&items, envelope, e2e, plaintext);

        // A valid owner ACK inside an INVALID transport packet must never
        // reach custody deletion. Observe the real handler, not a timeout alone.
        let mut tampered = client.ack_packet(envelope.message_id);
        *tampered.last_mut().unwrap() ^= 1;
        client.send(&tampered).await;
        runtime.expect_drops(1, 0).await;
        assert_custody_envelope(&client.pull().await.0, envelope, e2e, plaintext);

        client.send(&replay_packet).await;
        runtime.expect_drops(1, 1).await;
        assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 1);
        let ack = client.ack_packet(envelope.message_id);
        client.send(&ack).await;
        // Ordered processing of the subsequent Pull provides an authenticated
        // completion barrier for ACK, which has no response of its own.
        assert!(client.pull().await.0.is_empty());
        assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 0);
        drop(client);
        runtime.close().await;
        target = target.reopen(path, [0x93; 32]).await;

        let runtime = UdpRuntime::start(&mut target, ingress);
        let mut client = WireClient::connect(&target, receiver).await;
        assert!(client.pull().await.0.is_empty());
        assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 0);
        drop(client);
        runtime.close().await;
    }

    // [CHAT-PORTABLE-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] Compiles and
    // runs without a TUN on every supported platform; execution evidence is
    // limited to the actual host running this test, not an unrun Linux runner.
    #[tokio::test]
    async fn chat_custody_portable_handshake_pull_ack_survives_reopen() {
        tokio::time::timeout(
            Duration::from_secs(35),
            run_custody_acceptance(CustodyAcceptanceCase::PortableHandshake),
        )
        .await
        .expect("bounded portable handshake/custody/reopen lifecycle");
    }

    #[cfg(not(target_os = "linux"))]
    #[tokio::test]
    async fn chat_custody_real_handshake_pull_ack_survives_reopen() {
        tokio::time::timeout(
            Duration::from_secs(35),
            run_custody_acceptance(CustodyAcceptanceCase::TransportHandshake),
        )
        .await
        .expect("bounded handshake/custody/reopen lifecycle");
    }
}
