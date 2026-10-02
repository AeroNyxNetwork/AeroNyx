// File: crates/aeronyx-server/src/server/tests/chat_pull.rs
// Purpose: Pull authority, direct-relay custody, and UDP handshake acceptance.
// Dependencies: parent test helpers, production dispatcher/router, core crypto,
// ephemeral loopback HTTP/UDP, and private temporary SQLite repositories.
// Flow: signed sender ingress -> direct-v3 -> reopen -> signed PullV1/V2/ACK;
// portable handler composition adds V2 handshake and inbound AEAD/replay;
// a separate non-Linux fixture also exercises the production UDP dispatcher.
// Boundary: legacy cases retain preauthenticated sender ingress; the dual case
// uses real handshake/packet services at both ends. Orderly in-process reopen,
// NOT Server::run, discovery gossip, process crash, failover, or Linux TUN proof.
// [CHAT-CUSTODY-LIFECYCLE 2026-10-02 by Codex] Add composed acceptance without
// changing production behavior or seeding target custody through storage APIs.
// [CHAT-V1-COALESCING 2026-10-03 by Codex] Cover byte-target prefixes,
// single-envelope compatibility and 51-item multi-page custody acceptance.
// Last Modified: 2026-10-03. Originally split from server.rs `mod tests`.
use super::*;
// [CHAT-V1-COALESCING 2026-10-03 by Codex] Import the production AEAD size explicitly.
use aeronyx_core::crypto::transport::ENCRYPTION_OVERHEAD;

// [CHAT-V1-COALESCING 2026-10-03 by Codex] Codec-only synthetic envelopes:
// these do not claim valid signatures or replace real-ingress custody fixtures.
fn coalescing_envelope(id: u8, ciphertext_len: usize) -> ChatEnvelope {
    ChatEnvelope {
        message_id: [id; 16],
        sender: [0; 32],
        receiver: [0; 32],
        timestamp: 0,
        ciphertext: vec![0; ciphertext_len],
        nonce: [0; 24],
        content_type: ChatContentType::Text,
        signature: [0; 64],
    }
}

fn coalescing_datagram_size(response: &MemChainMessage) -> usize {
    let plaintext = aeronyx_core::protocol::encode_memchain(response).unwrap();
    // Exercise the actual DataPacket codec as well as the application encoder.
    let packet = aeronyx_core::protocol::DataPacket::new(
        [0; 16],
        0,
        vec![0; plaintext.len() + ENCRYPTION_OVERHEAD],
    );
    aeronyx_core::protocol::codec::encode_data_packet(&packet).len()
}

#[test]
fn chat_pull_v1_coalescing_exact_boundary_and_plus_one() {
    // 54 + 2 * (188 + 385) = 1200 full UDP payload bytes.
    let exact = Server::coalesce_legacy_chat_pull(
        vec![coalescing_envelope(1, 385), coalescing_envelope(2, 385)],
        false,
    )
    .unwrap();
    assert_eq!(
        coalescing_datagram_size(&exact),
        Server::LEGACY_CHAT_PULL_COALESCING_TARGET
    );
    assert!(
        matches!(&exact, MemChainMessage::ChatPullResponse { envelopes, has_more: false } if envelopes.len() == 2)
    );
    let overflow = Server::coalesce_legacy_chat_pull(
        vec![
            coalescing_envelope(1, 385),
            coalescing_envelope(2, 386),
            coalescing_envelope(3, 1),
        ],
        false,
    )
    .unwrap();
    assert_eq!(coalescing_datagram_size(&overflow), 627);
    assert!(
        matches!(&overflow, MemChainMessage::ChatPullResponse { envelopes, has_more: true }
        if envelopes.len() == 1 && envelopes[0].message_id == [1; 16])
    );
}

#[test]
fn chat_pull_v1_coalescing_single_over_target_stays_whole() {
    let first = coalescing_envelope(1, 2048);
    let expected =
        aeronyx_core::protocol::encode_memchain(&MemChainMessage::ChatRelay(first.clone()))
            .unwrap();
    for has_tail in [false, true] {
        let mut input = vec![first.clone()];
        if has_tail {
            input.push(coalescing_envelope(2, 1));
        }
        let response = Server::coalesce_legacy_chat_pull(input, false).unwrap();
        assert_eq!(coalescing_datagram_size(&response), 2290);
        let MemChainMessage::ChatPullResponse {
            envelopes,
            has_more,
        } = response
        else {
            panic!("expected V1 response")
        };
        assert_eq!(has_more, has_tail);
        assert_eq!(envelopes.len(), 1);
        assert!(
            aeronyx_core::protocol::encode_memchain(&MemChainMessage::ChatRelay(
                envelopes[0].clone()
            ))
            .unwrap()
                == expected
        );
    }
}

#[test]
fn chat_pull_v1_coalescing_preserves_existing_more_flag() {
    for envelopes in [Vec::new(), vec![coalescing_envelope(1, 1)]] {
        let response = Server::coalesce_legacy_chat_pull(envelopes, true).unwrap();
        assert!(coalescing_datagram_size(&response) <= Server::LEGACY_CHAT_PULL_COALESCING_TARGET);
        assert!(matches!(
            response,
            MemChainMessage::ChatPullResponse { has_more: true, .. }
        ));
    }
    let empty = Server::coalesce_legacy_chat_pull(Vec::new(), false).unwrap();
    assert_eq!(coalescing_datagram_size(&empty), 54);
    assert!(matches!(
        empty,
        MemChainMessage::ChatPullResponse {
            has_more: false,
            ..
        }
    ));
}

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
    // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] One expected commitment
    // per admitted fixture envelope; do not replace the real target receipt.
    request_commitments: Vec<[u8; 32]>,
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
    let receipt = ack.receipt.as_ref().expect("signed custody receipt");
    assert!(
        evidence.request_commitments.iter().any(|commitment| receipt
            .verify_expected_commitment(commitment, &evidence.target, unix_now_secs())
            .is_ok()),
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
    // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] Both client-facing
    // ends authenticate on the wire; peer custody still uses the real HTTP API.
    DualPortableHandshake,
    // [CHAT-V1-COALESCING 2026-10-03 by Codex] V2 transport carrying
    // independently assembled legacy V1 frames across byte-target-sized pages.
    ManualLegacyPages,
    // [CHAT-V1-COALESCING 2026-10-03 by Codex] Deterministic socket failure
    // leaves durable custody available after reopen and before an explicit ACK.
    LegacyWriteFailure,
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
    let mut source =
        CustodyAcceptanceNode::open(&source_path, [0x92; 32], IdentityKeyPair::generate()).await;
    let mut target =
        CustodyAcceptanceNode::open(&target_path, [0x93; 32], IdentityKeyPair::generate()).await;
    // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] Envelope construction
    // needs an identity, not an injected session. Only legacy cases use client().
    let sender_identity = IdentityKeyPair::generate();
    let receiver_identity = IdentityKeyPair::generate();
    let (sender_e2e, sender_kem) =
        sender_identity.e2e_handshake(&receiver_identity.x25519_public_key_bytes());
    let (receiver_e2e, _) = receiver_identity.e2e_handshake(&sender_kem);
    let plaintext = b"ephemeral custody acceptance payload";
    let nonce = [0x94; 24];
    let mut envelope = ChatEnvelope {
        message_id: [0x95; 16],
        sender: sender_identity.public_key_bytes(),
        receiver: receiver_identity.public_key_bytes(),
        timestamp: unix_now_secs(),
        ciphertext: sender_e2e.encrypt_raw(plaintext, &nonce).unwrap(),
        nonce,
        content_type: ChatContentType::Text,
        signature: [0; 64],
    };
    envelope.signature = sender_identity.sign(&envelope.sign_data());
    // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] Deliberately reverse
    // timestamp order relative to IDs and share seconds across adjacent IDs.
    // Each message has its own nonce; every target row enters through real HTTP.
    let envelopes = if matches!(case, CustodyAcceptanceCase::ManualLegacyPages) {
        (1u8..=51)
            .map(|index| {
                let mut item = envelope.clone();
                item.message_id = [index; 16];
                item.timestamp = envelope
                    .timestamp
                    .checked_sub(u64::from((index - 1) / 2))
                    .unwrap();
                item.nonce = [index; 24];
                item.ciphertext = sender_e2e.encrypt_raw(plaintext, &item.nonce).unwrap();
                item.signature = sender_identity.sign(&item.sign_data());
                item
            })
            .collect::<Vec<_>>()
    } else {
        vec![envelope.clone()]
    };
    let target_id = target.identity.public_key_bytes();
    let commitments = envelopes
        .iter()
        .map(|item| {
            PeerChatRelayRequestV3::sign(item.clone(), target_id, &source.identity)
                .unwrap()
                .request_commitment()
                .unwrap()
        })
        .collect();
    let evidence = Arc::new(CustodyHttpEvidence {
        target: target_id,
        request_commitments: commitments,
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
    // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] No direct dispatch or
    // SessionManager::create on the dual case's sender or receiver path.
    if matches!(case, CustodyAcceptanceCase::ManualLegacyPages) {
        transport_acceptance::submit_legacy_pages_from_wire(
            &mut source,
            sender_identity,
            &envelopes,
            Arc::clone(&http),
        )
        .await;
    } else if matches!(case, CustodyAcceptanceCase::DualPortableHandshake) {
        transport_acceptance::submit_from_wire(
            &mut source,
            sender_identity,
            &envelope,
            Arc::clone(&http),
            &evidence,
            &target,
        )
        .await;
    } else {
        let sender = source.client(sender_identity).await;
        source
            .dispatch(
                &sender,
                MemChainMessage::ChatRelay(envelope.clone()),
                Some(http.as_ref()),
            )
            .await;
    }
    assert_eq!(
        evidence.v3_calls.load(AtomicOrdering::SeqCst),
        envelopes.len()
    );
    assert_eq!(evidence.legacy_calls.load(AtomicOrdering::SeqCst), 0);
    assert_eq!(
        evidence.valid_receipts.load(AtomicOrdering::SeqCst),
        envelopes.len()
    );
    let expected_pending = u64::try_from(envelopes.len()).unwrap();
    let source_status = source.relay.peer_status();
    assert_eq!(source_status.outbound_accepted_total, expected_pending);
    assert_eq!(source_status.outbound_failed_total, 0);
    assert_eq!(target.relay.peer_status().inbound_delivered_online_total, 0);
    assert_eq!(
        target.relay.storage_usage().unwrap().pending_messages,
        expected_pending
    );
    // There has been no receiver session on T. Local source fallback custody
    // is allowed; this is one target copy, not a global exactly-once assertion.
    router.close().await;
    target = target.reopen(&target_path, [0x93; 32]).await;
    if matches!(case, CustodyAcceptanceCase::ManualLegacyPages) {
        transport_acceptance::run_legacy_pages(
            target,
            &target_path,
            receiver_identity,
            &envelopes,
            &receiver_e2e,
            plaintext,
        )
        .await;
        return;
    }
    // [CHAT-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] The receiver in this
    // branch has no injected Session and never calls the dispatcher directly.
    // [CHAT-PORTABLE-TRANSPORT-ACCEPTANCE 2026-10-02 by Codex] Both ingress
    // variants share the wire client and the complete durable custody lifecycle.
    let ingress = match case {
        CustodyAcceptanceCase::PortableHandshake | CustodyAcceptanceCase::DualPortableHandshake => {
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
        CustodyAcceptanceCase::LegacyWriteFailure => {
            // [CHAT-V1-COALESCING 2026-10-03 by Codex] Shutdown forces the
            // real send to return false without sysctl changes or UDP loss bets.
            target.udp.shutdown().await.unwrap();
            let response =
                Server::coalesce_legacy_chat_pull(vec![envelope.clone()], false).unwrap();
            assert!(
                !Server::send_to_session(
                    &response,
                    &receiver.session,
                    &target.udp,
                    &DefaultTransportCrypto::new(),
                )
                .await
            );
            let frame = transport_acceptance::legacy_pull_frame(
                &receiver.identity,
                0,
                [0; 16],
                unix_now_secs(),
                b"AeroNyx-ChatPull-v1",
            );
            let request = aeronyx_core::protocol::decode_memchain(&frame[1..]).unwrap();
            target.dispatch(&receiver, request, None).await;
            assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 1);
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
        CustodyAcceptanceCase::ManualLegacyPages => {
            unreachable!("handled before session injection")
        }
        CustodyAcceptanceCase::PortableHandshake | CustodyAcceptanceCase::DualPortableHandshake => {
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

// [CHAT-V1-COALESCING 2026-10-03 by Codex] Existing custody lifecycle still
// requires explicit owner ACK after a failed V1 write and successful reopen.
#[tokio::test]
async fn chat_custody_failed_v1_write_preserves_item_until_ack() {
    run_custody_acceptance(CustodyAcceptanceCase::LegacyWriteFailure).await;
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
        // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] Test-only portable
        // completion evidence, not a protocol ACK or a production metric.
        completed_dispatches: Arc<AtomicUsize>,
        udp: Arc<UdpTransport>,
        task: tokio::task::JoinHandle<()>,
    }

    impl UdpRuntime {
        fn start(node: &mut CustodyAcceptanceNode, ingress: Ingress) -> Self {
            Self::start_with_http(node, ingress, None)
        }

        // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] Inject only the
        // existing loopback peer client for the sender's portable composition;
        // receiver-only callers preserve their no-outbound-HTTP boundary.
        fn start_with_http(
            node: &mut CustodyAcceptanceNode,
            ingress: Ingress,
            http: Option<Arc<reqwest::Client>>,
        ) -> Self {
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
            let completed_dispatches = Arc::new(AtomicUsize::new(0));
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
                    let completed = Arc::clone(&completed_dispatches);
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
                                        http.as_deref(),
                                    )
                                    .await;
                                    // Observe completion only after the real
                                    // authenticated dispatcher returns. A lost
                                    // negative-control datagram cannot pass.
                                    completed.fetch_add(1, AtomicOrdering::SeqCst);
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
                completed_dispatches,
                udp: Arc::clone(&node.udp),
                task,
            }
        }

        // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] A bounded local
        // barrier complements authenticated Pull without assuming UDP delivery.
        async fn expect_completed(&self, count: usize) {
            tokio::time::timeout(Duration::from_secs(5), async {
                while self.completed_dispatches.load(AtomicOrdering::SeqCst) != count {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("portable authenticated dispatch completion deadline");
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
            let clear = aeronyx_core::protocol::encode_memchain(message).unwrap();
            self.packet_from_plaintext(&clear)
        }

        // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] Share negotiated
        // transport without forcing independently built frames through serde.
        fn packet_from_plaintext(&mut self, clear: &[u8]) -> Vec<u8> {
            self.tx_counter = self.tx_counter.checked_add(1).unwrap();
            let mut sealed = vec![0; clear.len() + 16];
            let len = DefaultTransportCrypto::new()
                .encrypt(
                    &self.keys.c2s,
                    self.tx_counter,
                    &self.session_id,
                    clear,
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
            let clear = self.receive_plaintext().await;
            let MemChainMessage::ChatPullResponseV2 {
                envelopes,
                has_more,
                ..
            } = aeronyx_core::protocol::decode_memchain(&clear[1..]).unwrap()
            else {
                panic!("expected PullV2 response")
            };
            assert!(!has_more);
            (envelopes, packet)
        }

        // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] Reuse exact s2c,
        // session and replay checks for both application response versions.
        async fn receive_plaintext(&mut self) -> Vec<u8> {
            let mut datagram = vec![0; 65_535];
            let (len, source) =
                tokio::time::timeout(Duration::from_secs(3), self.udp.recv(&mut datagram))
                    .await
                    .expect("bounded encrypted pull response")
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
            clear.truncate(len);
            assert_eq!(clear[0], aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC);
            clear
        }

        // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] Independent V1
        // application bytes inside the negotiated V2 transport session.
        async fn legacy_pull(&mut self, after: u64, cursor: [u8; 16]) -> (Vec<ChatEnvelope>, bool) {
            let frame = legacy_pull_frame(
                &self.identity,
                after,
                cursor,
                unix_now_secs(),
                b"AeroNyx-ChatPull-v1",
            );
            let packet = self.packet_from_plaintext(&frame);
            self.send(&packet).await;
            let clear = self.receive_plaintext().await;
            // [CHAT-V1-COALESCING 2026-10-03 by Codex] These real-wire
            // fixture items are small; every full response must fit the target.
            let packet_bytes = clear.len()
                + ENCRYPTION_OVERHEAD
                + aeronyx_core::protocol::messages::DATA_PACKET_HEADER_SIZE;
            assert!(packet_bytes <= Server::LEGACY_CHAT_PULL_COALESCING_TARGET);
            let page = parse_legacy_response(&clear).expect("bounded canonical V1 response");
            // Exercise parser fail-closed bounds against real response bytes.
            assert!(parse_legacy_response(&clear[..clear.len() - 1]).is_none());
            let mut invalid = clear.clone();
            invalid.extend_from_slice(&[0]);
            assert!(parse_legacy_response(&invalid).is_none());
            invalid = clear;
            invalid[5..13].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(parse_legacy_response(&invalid).is_none());
            page
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

    // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] Freeze the client V1
    // layout independently of serde and of the server's signing helper.
    pub(super) fn legacy_pull_frame(
        identity: &IdentityKeyPair,
        after: u64,
        cursor: [u8; 16],
        timestamp: u64,
        domain: &[u8],
    ) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        let wallet = identity.public_key_bytes();
        let mut fields = Vec::new();
        fields.extend_from_slice(&wallet);
        fields.extend_from_slice(&after.to_le_bytes());
        fields.extend_from_slice(&cursor);
        fields.extend_from_slice(&50u32.to_le_bytes());
        fields.extend_from_slice(&timestamp.to_le_bytes());
        let mut transcript = Sha256::new();
        transcript.update(domain);
        transcript.update(&fields);
        let signature = identity.sign(&transcript.finalize());
        let mut frame = vec![0xAE, 12, 0, 0, 0];
        frame.extend_from_slice(&fields);
        frame.extend_from_slice(&signature);
        assert_eq!(frame.len(), 137);
        assert!(
            frame
                == aeronyx_core::protocol::encode_memchain(&MemChainMessage::ChatPull {
                    wallet,
                    after_timestamp: after,
                    cursor,
                    limit: 50,
                    request_timestamp: timestamp,
                    signature,
                })
                .unwrap()
        );
        frame
    }

    // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] ACK signs the ordered
    // ID digest, not its wire count. No production codec constructs this frame.
    fn legacy_ack_frame(identity: &IdentityKeyPair, ids: &[[u8; 16]]) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        assert!(!ids.is_empty() && ids.len() <= 100);
        let wallet = identity.public_key_bytes();
        let timestamp = unix_now_secs();
        let mut ordered = Vec::new();
        for id in ids {
            ordered.extend_from_slice(id);
        }
        let mut transcript = Sha256::new();
        transcript.update(b"AeroNyx-ChatAck-v1");
        transcript.update(wallet);
        transcript.update(timestamp.to_le_bytes());
        transcript.update(Sha256::digest(&ordered));
        let signature = identity.sign(&transcript.finalize());
        let mut frame = vec![0xAE, 14, 0, 0, 0];
        frame.extend_from_slice(&u64::try_from(ids.len()).unwrap().to_le_bytes());
        frame.extend_from_slice(&ordered);
        frame.extend_from_slice(&wallet);
        frame.extend_from_slice(&timestamp.to_le_bytes());
        frame.extend_from_slice(&signature);
        assert_eq!(frame.len(), 117 + 16 * ids.len());
        assert!(
            frame
                == aeronyx_core::protocol::encode_memchain(&MemChainMessage::ChatAck {
                    message_ids: ids.to_vec(),
                    wallet,
                    ack_timestamp: timestamp,
                    signature,
                })
                .unwrap()
        );
        frame
    }

    // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] Bounded manual reader:
    // no allocation from an unvalidated wire length. Canonical bool/trailing
    // policy is deliberately stricter than the legacy Dart decoder.
    fn legacy_take<'a>(rest: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
        let (field, remaining) = rest.split_at_checked(len)?;
        *rest = remaining;
        Some(field)
    }

    fn legacy_array<const N: usize>(rest: &mut &[u8]) -> Option<[u8; N]> {
        legacy_take(rest, N)?.try_into().ok()
    }

    fn parse_legacy_response(frame: &[u8]) -> Option<(Vec<ChatEnvelope>, bool)> {
        use aeronyx_core::protocol::chat::ChatContentType;
        if frame.len() > 65_535 {
            return None;
        }
        let mut rest = frame;
        if legacy_take(&mut rest, 5)? != [0xAE, 13, 0, 0, 0] {
            return None;
        }
        let count = usize::try_from(u64::from_le_bytes(legacy_array(&mut rest)?)).ok()?;
        if count > 500 || count > rest.len() / 188 {
            return None;
        }
        let mut envelopes = Vec::with_capacity(count);
        for _ in 0..count {
            let message_id = legacy_array(&mut rest)?;
            let sender = legacy_array(&mut rest)?;
            let receiver = legacy_array(&mut rest)?;
            let timestamp = u64::from_le_bytes(legacy_array(&mut rest)?);
            let len = usize::try_from(u64::from_le_bytes(legacy_array(&mut rest)?)).ok()?;
            let ciphertext = legacy_take(&mut rest, len)?.to_vec();
            let nonce = legacy_array(&mut rest)?;
            let content_type = match u32::from_le_bytes(legacy_array(&mut rest)?) {
                0 => ChatContentType::Text,
                1 => ChatContentType::Media,
                2 => ChatContentType::System,
                _ => return None,
            };
            let signature = legacy_array(&mut rest)?;
            envelopes.push(ChatEnvelope {
                message_id,
                sender,
                receiver,
                timestamp,
                ciphertext,
                nonce,
                content_type,
                signature,
            });
        }
        let has_more = match legacy_take(&mut rest, 1)?[0] {
            0 => false,
            1 => true,
            _ => return None,
        };
        rest.is_empty().then_some((envelopes, has_more))
    }

    // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] All 51 items traverse
    // real sender handshake/AEAD admission and the target's HTTP custody path.
    pub(super) async fn submit_legacy_pages_from_wire(
        source: &mut CustodyAcceptanceNode,
        sender: IdentityKeyPair,
        envelopes: &[ChatEnvelope],
        http: Arc<reqwest::Client>,
    ) {
        let runtime = UdpRuntime::start_with_http(source, Ingress::PortableHandlers, Some(http));
        let mut client = WireClient::connect(source, sender).await;
        for (index, envelope) in envelopes.iter().enumerate() {
            let packet = client.packet(&MemChainMessage::ChatRelay(envelope.clone()));
            client.send(&packet).await;
            runtime.expect_completed(index + 1).await;
        }
        let (items, more) = client.legacy_pull(0, [0; 16]).await;
        assert!(items.is_empty() && !more);
        runtime.expect_completed(envelopes.len() + 1).await;
        drop(client);
        runtime.close().await;
    }

    // [CHAT-MANUAL-V1-ACCEPTANCE 2026-10-02 by Codex] Fixed timestamp floor
    // across all pages, including deletion of the previous cursor row by ACK.
    pub(super) async fn run_legacy_pages(
        mut target: CustodyAcceptanceNode,
        path: &std::path::Path,
        receiver: IdentityKeyPair,
        expected: &[ChatEnvelope],
        e2e: &aeronyx_core::crypto::E2eSession,
        plaintext: &[u8],
    ) {
        assert_eq!(expected.len(), 51);
        assert_eq!(expected[0].timestamp, expected[1].timestamp);
        assert!(expected[0].timestamp > expected[50].timestamp);
        assert!(expected
            .windows(2)
            .all(|pair| pair[0].message_id < pair[1].message_id));
        let after = expected[50].timestamp.checked_sub(1).unwrap();
        let runtime = UdpRuntime::start(&mut target, Ingress::PortableHandlers);
        let mut client = WireClient::connect(&target, receiver.clone()).await;
        let now = unix_now_secs();
        let wrong_domain =
            legacy_pull_frame(&receiver, after, [0; 16], now, b"AeroNyx-ChatPull-v2");
        let mut changed_cursor =
            legacy_pull_frame(&receiver, after, [0; 16], now, b"AeroNyx-ChatPull-v1");
        changed_cursor[45] ^= 1;
        let expired = legacy_pull_frame(
            &receiver,
            after,
            [0; 16],
            now.checked_sub(120).unwrap(),
            b"AeroNyx-ChatPull-v1",
        );
        for (index, frame) in [wrong_domain, changed_cursor, expired].iter().enumerate() {
            let packet = client.packet_from_plaintext(frame);
            client.send(&packet).await;
            // Completion is observed after the real dispatcher returns. A lost
            // packet alone cannot make the subsequent no-response check pass.
            runtime.expect_completed(index + 1).await;
            assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 51);
            assert!(target
                .relay
                .wallet_routes
                .lookup(&receiver.public_key_bytes())
                .is_empty());
            let mut datagram = vec![0; 65_535];
            assert!(tokio::time::timeout(
                Duration::from_millis(100),
                client.udp.recv(&mut datagram)
            )
            .await
            .is_err());
        }
        // [CHAT-V1-COALESCING 2026-10-03 by Codex] Limit50 remains an upper
        // bound. Prove an exact ordered union across byte-target-sized pages.
        let mut cursor = [0; 16];
        let mut received = 0;
        let mut pages = 0;
        let mut completed = 3;
        let mut unique = std::collections::BTreeSet::new();
        loop {
            assert!(pages < expected.len(), "bounded pagination progress");
            let (page, more) = client.legacy_pull(after, cursor).await;
            completed += 1;
            runtime.expect_completed(completed).await;
            // This fixture has no expiry notifications; no empty progress page.
            assert!(!page.is_empty() && page.len() <= 50);
            let end = received + page.len();
            assert!(end <= expected.len());
            for (actual, expected) in page.iter().zip(&expected[received..end]) {
                assert_custody_envelope(std::slice::from_ref(actual), expected, e2e, plaintext);
                assert!(unique.insert(actual.message_id), "duplicate across pages");
            }
            cursor = page.last().unwrap().message_id;
            received = end;
            pages += 1;
            assert_eq!(more, received < expected.len());
            // ACK only this fully verified page. The next cursor deliberately
            // references a deleted row; never advance to fetched-but-unsent IDs.
            let ids: Vec<_> = page.iter().map(|item| item.message_id).collect();
            let packet = client.packet_from_plaintext(&legacy_ack_frame(&receiver, &ids));
            client.send(&packet).await;
            completed += 1;
            runtime.expect_completed(completed).await;
            assert_eq!(
                target.relay.storage_usage().unwrap().pending_messages,
                u64::try_from(expected.len() - received).unwrap()
            );
            if !more {
                break;
            }
        }
        assert!(pages > 1);
        assert_eq!(received, 51);
        assert_eq!(unique.len(), 51);
        let (empty, more) = client.legacy_pull(after, cursor).await;
        runtime.expect_completed(completed + 1).await;
        assert!(empty.is_empty() && !more);
        assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 0);
        drop(client);
        runtime.close().await;
        target = target.reopen(path, [0x93; 32]).await;
        let runtime = UdpRuntime::start(&mut target, Ingress::PortableHandlers);
        let mut client = WireClient::connect(&target, receiver).await;
        let (empty, more) = client.legacy_pull(0, [0; 16]).await;
        runtime.expect_completed(1).await;
        assert!(empty.is_empty() && !more);
        assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 0);
        drop(client);
        runtime.close().await;
    }

    // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] Reuse the same wire
    // client for sender admission. Legacy ChatRelay has no sender success ACK:
    // neither UDP send nor our test-only completion counter proves durability.
    // The real request-bound target receipt, stored item and subsequent reopen do.
    pub(super) async fn submit_from_wire(
        source: &mut CustodyAcceptanceNode,
        sender: IdentityKeyPair,
        envelope: &ChatEnvelope,
        http: Arc<reqwest::Client>,
        evidence: &CustodyHttpEvidence,
        target: &CustodyAcceptanceNode,
    ) {
        let runtime = UdpRuntime::start_with_http(source, Ingress::PortableHandlers, Some(http));
        let mut client = WireClient::connect(source, sender).await;
        assert_eq!(target.sessions.count(), 0, "receiver remains offline");
        let assert_no_custody = || {
            assert_eq!(evidence.v3_calls.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(evidence.legacy_calls.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(evidence.valid_receipts.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(source.relay.peer_status().outbound_accepted_total, 0);
            assert_eq!(source.relay.storage_usage().unwrap().pending_messages, 0);
            assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 0);
        };
        assert_no_custody();

        let mut tampered = client.packet(&MemChainMessage::ChatRelay(envelope.clone()));
        *tampered.last_mut().unwrap() ^= 1;
        client.send(&tampered).await;
        runtime.expect_drops(1, 0).await;
        assert_eq!(runtime.completed_dispatches.load(AtomicOrdering::SeqCst), 0);
        assert_no_custody();

        // This envelope is correctly signed but not by the authenticated session
        // identity. Observe its completed dispatch before checking zero effects.
        let unrelated_identity = IdentityKeyPair::generate();
        let mut mismatched = envelope.clone();
        mismatched.sender = unrelated_identity.public_key_bytes();
        mismatched.signature = unrelated_identity.sign(&mismatched.sign_data());
        assert!(mismatched.verify_signature().is_ok());
        let mismatch = client.packet(&MemChainMessage::ChatRelay(mismatched));
        client.send(&mismatch).await;
        runtime.expect_completed(1).await;
        assert!(client.pull().await.0.is_empty());
        runtime.expect_completed(2).await;
        assert_no_custody();
        assert!(source
            .relay
            .wallet_routes
            .lookup(&unrelated_identity.public_key_bytes())
            .is_empty());

        let valid = client.packet(&MemChainMessage::ChatRelay(envelope.clone()));
        client.send(&valid).await;
        runtime.expect_completed(3).await;
        // Signed Pull queries the sender's own mailbox, never the receiver's.
        // An authenticated response is a serial barrier, not a custody receipt.
        assert!(client.pull().await.0.is_empty());
        runtime.expect_completed(4).await;
        assert_eq!(evidence.v3_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(evidence.valid_receipts.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(source.relay.storage_usage().unwrap().pending_messages, 1);
        assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 1);

        client.send(&valid).await;
        runtime.expect_drops(1, 1).await;
        assert_eq!(runtime.completed_dispatches.load(AtomicOrdering::SeqCst), 4);
        assert!(client.pull().await.0.is_empty());
        runtime.expect_completed(5).await;
        assert_eq!(evidence.v3_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(evidence.legacy_calls.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(evidence.valid_receipts.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(source.relay.peer_status().outbound_accepted_total, 1);
        assert_eq!(source.relay.peer_status().outbound_failed_total, 0);
        assert_eq!(source.relay.storage_usage().unwrap().pending_messages, 1);
        assert_eq!(target.relay.storage_usage().unwrap().pending_messages, 1);
        assert_eq!(target.sessions.count(), 0, "receiver remains offline");
        drop(client);
        runtime.close().await;
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

    // [CHAT-DUAL-HANDSHAKE-CUSTODY 2026-10-02 by Codex] Both client-facing
    // ends use portable production services, not a full Server::run/TUN fixture.
    #[tokio::test]
    async fn chat_custody_dual_handshake_sender_admission_pull_ack_survives_reopen() {
        tokio::time::timeout(
            Duration::from_secs(45),
            run_custody_acceptance(CustodyAcceptanceCase::DualPortableHandshake),
        )
        .await
        .expect("bounded dual handshake/custody/reopen lifecycle");
    }

    // [CHAT-V1-COALESCING 2026-10-03 by Codex] Final acceptance slice with
    // small coalesced pages; not a PMTU guarantee or Dart/native client test.
    #[tokio::test]
    async fn chat_custody_manual_v1_multiple_pages_ack_survives_reopen() {
        tokio::time::timeout(
            Duration::from_secs(90),
            run_custody_acceptance(CustodyAcceptanceCase::ManualLegacyPages),
        )
        .await
        .expect("bounded manual V1 51-item multi-page custody lifecycle");
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
