// File: crates/aeronyx-server/src/server/tests/chat_pull.rs
// Purpose: Pull route authority and direct-relay custody lifecycle acceptance.
// Dependencies: parent test helpers, production dispatcher/router, core crypto,
// ephemeral loopback HTTP/UDP, and private temporary SQLite repositories.
// Flow: signed sender ingress -> direct-v3 -> reopen -> signed PullV2/ACK.
// Boundary: preauthenticated sessions, in-process tasks, orderly DB reopen;
// NOT handshake, Server::run, discovery gossip, process crash, or failover proof.
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
    let mut receiver = target.client(receiver_identity.clone()).await;
    assert_custody_envelope(
        &receiver.pull(&target).await,
        &envelope,
        &receiver_e2e,
        plaintext,
    );
    match case {
        CustodyAcceptanceCase::Ack => {}
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
