// ============================================
// File: crates/aeronyx-server/tests/mailbox_wire_interop.rs
// ============================================
//! Interop between `aeronyx-mailbox-wire` (the client's source, dalek 4,
//! strict verification) and the node (`aeronyx-core`, dalek 1).
//!
//! [MAILBOX-WIRE 2026-10-09 by Claude]
//! 1. Golden bytes: with fixed keys and inputs, every deterministic frame,
//!    signature and commitment is byte-identical on both sides.
//! 2. Full handler: the wire crate acts as a phone (fresh source key, no VPN)
//!    against the node's real `/api/chat/peer/blind-relay` handler with a
//!    real SQLite mailbox store, through ticket → lease → put → pull → ack →
//!    empty pull. Previously every mailbox test bypassed the HTTP handler.

use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

use aeronyx_core::crypto::IdentityKeyPair as CoreKey;
use aeronyx_core::protocol::anonymous_mailbox as core_mb;
use aeronyx_core::protocol::chat as core_chat;
use aeronyx_mailbox_wire::chat as wire_chat;
use aeronyx_mailbox_wire::crypto::IdentityKeyPair as WireKey;
use aeronyx_mailbox_wire::mailbox as wire_mb;
use aeronyx_mailbox_wire::onion::OnionHop;
use aeronyx_mailbox_wire::source::SourceExchange;
use aeronyx_server::api::chat_peer::build_chat_peer_router_with_anonymous_mailbox;
use aeronyx_server::config_chat_relay::{AnonymousMailboxStoreConfig, ChatRelayConfig};
use aeronyx_server::services::chat_relay::ChatRelayService;
use aeronyx_server::services::chat_relay_mailbox::{
    AnonymousMailboxCustodyRepository, SqliteAnonymousMailboxStore,
};
use aeronyx_server::services::onion_keys;
use aeronyx_server::services::peer_store::PeerStore;
use aeronyx_server::services::SessionManager;
use aeronyx_transport::UdpTransport;

const T0: u64 = 1_800_000_000;

fn keys(seed: u8) -> (CoreKey, WireKey) {
    (
        CoreKey::from_bytes(&[seed; 32]).unwrap(),
        WireKey::from_bytes(&[seed; 32]).unwrap(),
    )
}

#[test]
fn identities_and_x25519_derivation_match() {
    for seed in [1u8, 7, 0x42, 0xfe] {
        let (core, wire) = keys(seed);
        assert_eq!(core.public_key_bytes(), wire.public_key_bytes());
        assert_eq!(
            core.x25519_public_key_bytes(),
            wire.x25519_public_key_bytes()
        );
        let message = b"aeronyx-mailbox-wire golden";
        assert_eq!(core.sign(message), wire.sign(message));
    }
}

#[test]
fn deterministic_frames_are_byte_identical() {
    let (core_target, wire_target) = keys(0x31);
    let (core_deposit, wire_deposit) = keys(0x32);
    let (core_reader, wire_reader) = keys(0x33);
    let (core_source, wire_source) = keys(0x34);
    let mailbox_id = [0x41; 32];
    let (issued, expires) = (T0, T0 + 86_400);

    let claims = core_mb::AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
        &mailbox_id,
        &core_deposit.public_key_bytes(),
        &core_reader.public_key_bytes(),
        16,
        1 << 20,
        issued,
        expires,
    );
    assert_eq!(
        claims,
        wire_mb::AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &mailbox_id,
            &wire_deposit.public_key_bytes(),
            &wire_reader.public_key_bytes(),
            16,
            1 << 20,
            issued,
            expires,
        )
    );

    // Ticket issue request: transcript, commitment and PoW digest.
    let core_issue = core_mb::AnonymousMailboxTicketIssueV1::new(
        [1; 16],
        [2; 16],
        core_target.public_key_bytes(),
        claims,
        T0,
        T0 + 240,
        7,
    )
    .unwrap();
    let wire_issue = wire_mb::AnonymousMailboxTicketIssueV1::new(
        [1; 16],
        [2; 16],
        wire_target.public_key_bytes(),
        claims,
        T0,
        T0 + 240,
        7,
    )
    .unwrap();
    assert_eq!(
        core_issue.signing_bytes().unwrap(),
        wire_issue.signing_bytes().unwrap()
    );
    assert_eq!(
        core_issue.request_commitment().unwrap(),
        wire_issue.request_commitment().unwrap()
    );
    assert_eq!(
        core_issue.proof_digest().unwrap(),
        wire_issue.proof_digest().unwrap()
    );

    // Target-issued admission ticket.
    let core_ticket = core_mb::AnonymousMailboxAdmissionTicketV1::issue(
        [2; 16],
        claims,
        T0,
        T0 + 240,
        &core_target,
    )
    .unwrap();
    let wire_ticket = wire_mb::AnonymousMailboxAdmissionTicketV1::issue(
        [2; 16],
        claims,
        T0,
        T0 + 240,
        &wire_target,
    )
    .unwrap();
    assert_eq!(core_ticket.signature, wire_ticket.signature);

    // Lease create, put, pull, ack: full encoded terminal frames.
    let core_lease = core_mb::AnonymousMailboxLeaseCreateV1::new(
        mailbox_id,
        core_deposit.public_key_bytes(),
        16,
        1 << 20,
        issued,
        expires,
        core_ticket,
        &core_reader,
    )
    .unwrap();
    let wire_lease = wire_mb::AnonymousMailboxLeaseCreateV1::new(
        mailbox_id,
        wire_deposit.public_key_bytes(),
        16,
        1 << 20,
        issued,
        expires,
        wire_ticket,
        &wire_reader,
    )
    .unwrap();
    let core_put = core_mb::AnonymousMailboxPutV1::new(
        mailbox_id,
        [3; 16],
        vec![0xAB; 300],
        T0,
        T0 + 3600,
        &core_deposit,
    )
    .unwrap();
    let wire_put = wire_mb::AnonymousMailboxPutV1::new(
        mailbox_id,
        [3; 16],
        vec![0xAB; 300],
        T0,
        T0 + 3600,
        &wire_deposit,
    )
    .unwrap();
    let core_pull =
        core_mb::AnonymousMailboxPullOneV1::new(mailbox_id, [4; 16], vec![9; 12], T0, &core_reader)
            .unwrap();
    let wire_pull =
        wire_mb::AnonymousMailboxPullOneV1::new(mailbox_id, [4; 16], vec![9; 12], T0, &wire_reader)
            .unwrap();
    let core_ack = core_mb::AnonymousMailboxAckV1::new(
        mailbox_id,
        [5; 16],
        [3; 16],
        [6; 32],
        T0,
        &core_reader,
    )
    .unwrap();
    let wire_ack = wire_mb::AnonymousMailboxAckV1::new(
        mailbox_id,
        [5; 16],
        [3; 16],
        [6; 32],
        T0,
        &wire_reader,
    )
    .unwrap();
    let pairs = [
        (
            core_mb::encode_anonymous_mailbox_terminal_frame(
                &core_mb::AnonymousMailboxTerminalFrameV1::TicketIssue(core_issue),
            ),
            wire_mb::encode_anonymous_mailbox_terminal_frame(
                &wire_mb::AnonymousMailboxTerminalFrameV1::TicketIssue(wire_issue),
            ),
        ),
        (
            core_mb::encode_anonymous_mailbox_terminal_frame(
                &core_mb::AnonymousMailboxTerminalFrameV1::LeaseCreate(core_lease),
            ),
            wire_mb::encode_anonymous_mailbox_terminal_frame(
                &wire_mb::AnonymousMailboxTerminalFrameV1::LeaseCreate(wire_lease),
            ),
        ),
        (
            core_mb::encode_anonymous_mailbox_terminal_frame(
                &core_mb::AnonymousMailboxTerminalFrameV1::Put(core_put),
            ),
            wire_mb::encode_anonymous_mailbox_terminal_frame(
                &wire_mb::AnonymousMailboxTerminalFrameV1::Put(wire_put),
            ),
        ),
        (
            core_mb::encode_anonymous_mailbox_terminal_frame(
                &core_mb::AnonymousMailboxTerminalFrameV1::PullOne(core_pull),
            ),
            wire_mb::encode_anonymous_mailbox_terminal_frame(
                &wire_mb::AnonymousMailboxTerminalFrameV1::PullOne(wire_pull),
            ),
        ),
        (
            core_mb::encode_anonymous_mailbox_terminal_frame(
                &core_mb::AnonymousMailboxTerminalFrameV1::Ack(core_ack),
            ),
            wire_mb::encode_anonymous_mailbox_terminal_frame(
                &wire_mb::AnonymousMailboxTerminalFrameV1::Ack(wire_ack),
            ),
        ),
    ];
    for (index, (core_bytes, wire_bytes)) in pairs.into_iter().enumerate() {
        assert_eq!(
            core_bytes.unwrap(),
            wire_bytes.unwrap(),
            "frame {index} differs"
        );
    }

    // Route request and its MemChain framing (variant 40).
    let core_route = core_mb::AnonymousMailboxRouteRequestV1::signed(
        [8; 16],
        core_target.public_key_bytes(),
        vec![1, 2, 3],
        T0,
        &core_source,
    )
    .unwrap();
    let wire_route = wire_mb::AnonymousMailboxRouteRequestV1::signed(
        [8; 16],
        wire_target.public_key_bytes(),
        vec![1, 2, 3],
        T0,
        &wire_source,
    )
    .unwrap();
    let core_framed = aeronyx_core::protocol::memchain::encode_memchain(
        &aeronyx_core::protocol::memchain::MemChainMessage::AnonymousMailboxRouteV1(core_route),
    )
    .unwrap();
    assert_eq!(
        core_framed,
        aeronyx_mailbox_wire::source::encode_route_payload(&wire_route).unwrap()
    );

    // Signed terminal response.
    let core_response = core_mb::AnonymousMailboxTerminalResponseV1::signed(
        core_mb::AnonymousMailboxOperationV1::PullOne,
        [4; 16],
        [7; 32],
        core_mb::AnonymousMailboxOutcomeV1::Accepted,
        vec![5; 40],
        T0,
        &core_target,
    )
    .unwrap();
    let wire_response = wire_mb::AnonymousMailboxTerminalResponseV1::signed(
        wire_mb::AnonymousMailboxOperationV1::PullOne,
        [4; 16],
        [7; 32],
        wire_mb::AnonymousMailboxOutcomeV1::Accepted,
        vec![5; 40],
        T0,
        &wire_target,
    )
    .unwrap();
    assert_eq!(core_response.signature, wire_response.signature);
}

#[test]
fn chat_and_blind_relay_envelopes_are_byte_identical() {
    let (core_sender, wire_sender) = keys(0x51);
    let core_env = core_chat::ChatEnvelope {
        message_id: [1; 16],
        sender: core_sender.public_key_bytes(),
        receiver: [2; 32],
        timestamp: T0,
        ciphertext: vec![3; 77],
        nonce: [4; 24],
        content_type: core_chat::ChatContentType::System,
        signature: [0; 64],
    };
    let wire_env = wire_chat::ChatEnvelope {
        message_id: [1; 16],
        sender: wire_sender.public_key_bytes(),
        receiver: [2; 32],
        timestamp: T0,
        ciphertext: vec![3; 77],
        nonce: [4; 24],
        content_type: wire_chat::ChatContentType::System,
        signature: [0; 64],
    };
    assert_eq!(core_env.sign_data(), wire_env.sign_data());
    let core_signed = core_chat::ChatEnvelope {
        signature: core_sender.sign(&core_env.sign_data()),
        ..core_env
    };
    let wire_signed = wire_chat::ChatEnvelope {
        signature: wire_sender.sign(&wire_env.sign_data()),
        ..wire_env
    };
    let core_bytes = core_chat::encode_envelope(&core_signed).unwrap();
    assert_eq!(
        core_bytes,
        wire_chat::encode_envelope(&wire_signed).unwrap()
    );
    // Each side accepts the other's bytes.
    assert!(core_chat::decode_envelope_strict_verified(&core_bytes).is_ok());
    assert_eq!(
        wire_chat::decode_envelope_strict_verified(&core_bytes).unwrap(),
        wire_signed
    );

    let core_relay = core_chat::BlindRelayEnvelope {
        route_id: [5; 16],
        next_hop: [6; 32],
        ttl: 2,
        encrypted_blob: vec![7; 99],
        timestamp: T0,
        signature: [0; 64],
    }
    .sign_with(&core_sender);
    let wire_relay = wire_chat::BlindRelayEnvelope {
        route_id: [5; 16],
        next_hop: [6; 32],
        ttl: 2,
        encrypted_blob: vec![7; 99],
        timestamp: T0,
        signature: [0; 64],
    }
    .sign_with(&wire_sender);
    assert_eq!(core_relay.signing_data(), wire_relay.signing_data());
    assert_eq!(core_relay.signature, wire_relay.signature);
    // The JSON shape the node deserializes.
    assert_eq!(
        serde_json::to_value(&core_relay).unwrap(),
        serde_json::to_value(&wire_relay).unwrap()
    );
}

async fn post(app: &axum::Router, body: &[u8]) -> Vec<u8> {
    let request = Request::builder()
        .method("POST")
        .uri(aeronyx_mailbox_wire::source::BLIND_RELAY_PATH)
        .header("content-type", "application/json")
        .body(Body::from(body.to_vec()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap()
        .to_vec()
}

/// The store refuses symlinked paths (macOS temp dirs live behind
/// `/var -> /private/var`), so tests use a real directory, as the node's own
/// store tests do.
fn test_dir() -> tempfile::TempDir {
    std::fs::create_dir_all("target/test-temp").unwrap();
    tempfile::TempDir::new_in("target/test-temp").unwrap()
}

/// Blind relay replay protection is durable in the chat relay service, as on
/// every production node; without it the handler refuses relay work.
fn chat_relay(directory: &tempfile::TempDir) -> Arc<ChatRelayService> {
    let config = ChatRelayConfig {
        enabled: true,
        db_path: directory
            .path()
            .join("chat.db")
            .to_string_lossy()
            .into_owned(),
        ..ChatRelayConfig::default()
    };
    Arc::new(ChatRelayService::new(config, [7u8; 32]).unwrap())
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[tokio::test]
async fn wire_source_runs_the_full_lifecycle_through_the_real_handler() {
    let directory = test_dir();
    let node = CoreKey::generate();
    let config = AnonymousMailboxStoreConfig {
        enabled: true,
        db_path: directory
            .path()
            .join("mailbox.db")
            .to_string_lossy()
            .into_owned(),
        ticket_issue_work_bits: 4,
        ..AnonymousMailboxStoreConfig::default()
    };
    let store =
        SqliteAnonymousMailboxStore::open_with_ticket_issuer(config, node.clone(), [0x5A; 32])
            .unwrap();
    let store: Arc<dyn AnonymousMailboxCustodyRepository> = Arc::new(store);
    let app = build_chat_peer_router_with_anonymous_mailbox(
        Some(chat_relay(&directory)),
        Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        Arc::new(PeerStore::new()),
        Arc::new(node.clone()),
        Arc::new(reqwest::Client::new()),
        None,
        Some(store),
    );
    let target = node.public_key_bytes();
    let path = [OnionHop {
        node_id: target,
        kem_pub: onion_keys::current_public_key(),
    }];

    let reader = WireKey::generate();
    let depositor = WireKey::generate();
    let mailbox_id = [0x77; 32];
    let t = now();
    let (lease_issued, lease_expires) = (t, t + 3600);
    let claims = wire_mb::AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
        &mailbox_id,
        &depositor.public_key_bytes(),
        &reader.public_key_bytes(),
        16,
        1 << 20,
        lease_issued,
        lease_expires,
    );

    // 1. Ticket (proof of work against the store's 4 bits).
    let issue = (0..u64::MAX)
        .map(|nonce| {
            wire_mb::AnonymousMailboxTicketIssueV1::new(
                [1; 16],
                [2; 16],
                target,
                claims,
                t,
                t + 240,
                nonce,
            )
            .unwrap()
        })
        .find(|request| request.verify_for_target(&target, t, 4).is_ok())
        .unwrap();
    let exchange = SourceExchange::prepare(
        &path,
        wire_mb::encode_anonymous_mailbox_terminal_frame(
            &wire_mb::AnonymousMailboxTerminalFrameV1::TicketIssue(issue.clone()),
        )
        .unwrap(),
        t,
    )
    .unwrap();
    let reply = post(&app, exchange.body_json()).await;
    let wire_mb::AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response) =
        exchange.open(&reply).unwrap()
    else {
        panic!("ticket response frame");
    };
    response.verify_for_request(&issue, &target).unwrap();
    assert_eq!(
        response.outcome,
        wire_mb::AnonymousMailboxOutcomeV1::Accepted
    );
    let ticket = response.ticket.unwrap();

    // Generic exchange helper for the four signed operations.
    async fn exchange_op(
        app: &axum::Router,
        path: &[OnionHop],
        frame: wire_mb::AnonymousMailboxTerminalFrameV1,
        operation: wire_mb::AnonymousMailboxOperationV1,
        request_id: [u8; 16],
        commitment: [u8; 32],
        target: [u8; 32],
    ) -> wire_mb::AnonymousMailboxTerminalResponseV1 {
        let exchange = SourceExchange::prepare(
            path,
            wire_mb::encode_anonymous_mailbox_terminal_frame(&frame).unwrap(),
            now(),
        )
        .unwrap();
        let reply = post(app, exchange.body_json()).await;
        let response = match exchange.open(&reply).unwrap() {
            wire_mb::AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(r)
            | wire_mb::AnonymousMailboxTerminalFrameV1::PutResponse(r)
            | wire_mb::AnonymousMailboxTerminalFrameV1::PullOneResponse(r)
            | wire_mb::AnonymousMailboxTerminalFrameV1::AckResponse(r) => r,
            _ => panic!("unexpected frame"),
        };
        response
            .verify_for_request(operation, &request_id, &commitment, &target)
            .unwrap();
        assert_eq!(
            response.outcome,
            wire_mb::AnonymousMailboxOutcomeV1::Accepted
        );
        response
    }

    // 2. Lease.
    let lease = wire_mb::AnonymousMailboxLeaseCreateV1::new(
        mailbox_id,
        depositor.public_key_bytes(),
        16,
        1 << 20,
        lease_issued,
        lease_expires,
        ticket.clone(),
        &reader,
    )
    .unwrap();
    let commitment = lease.request_commitment().unwrap();
    exchange_op(
        &app,
        &path,
        wire_mb::AnonymousMailboxTerminalFrameV1::LeaseCreate(lease),
        wire_mb::AnonymousMailboxOperationV1::LeaseCreate,
        ticket.ticket_id,
        commitment,
        target,
    )
    .await;

    // 3. Put.
    let sealed = vec![0xC3; 777];
    let put = wire_mb::AnonymousMailboxPutV1::new(
        mailbox_id,
        [3; 16],
        sealed.clone(),
        now(),
        lease_expires,
        &depositor,
    )
    .unwrap();
    let commitment = put.request_commitment().unwrap();
    exchange_op(
        &app,
        &path,
        wire_mb::AnonymousMailboxTerminalFrameV1::Put(put),
        wire_mb::AnonymousMailboxOperationV1::Put,
        [3; 16],
        commitment,
        target,
    )
    .await;

    // 4. Pull: exactly the deposited bytes.
    let pull =
        wire_mb::AnonymousMailboxPullOneV1::new(mailbox_id, [4; 16], Vec::new(), now(), &reader)
            .unwrap();
    let commitment = pull.request_commitment().unwrap();
    let response = exchange_op(
        &app,
        &path,
        wire_mb::AnonymousMailboxTerminalFrameV1::PullOne(pull),
        wire_mb::AnonymousMailboxOperationV1::PullOne,
        [4; 16],
        commitment,
        target,
    )
    .await;
    let item = wire_mb::AnonymousMailboxPullResultV1::decode(&response.sealed_payload).unwrap();
    assert_eq!(item.item_id, [3; 16]);
    assert_eq!(item.sealed_item, sealed);

    // 5. Ack, then the mailbox is empty.
    let ack = wire_mb::AnonymousMailboxAckV1::new(
        mailbox_id,
        [5; 16],
        item.item_id,
        item.sealed_commitment,
        now(),
        &reader,
    )
    .unwrap();
    let commitment = ack.request_commitment().unwrap();
    exchange_op(
        &app,
        &path,
        wire_mb::AnonymousMailboxTerminalFrameV1::Ack(ack),
        wire_mb::AnonymousMailboxOperationV1::Ack,
        [5; 16],
        commitment,
        target,
    )
    .await;
    let pull =
        wire_mb::AnonymousMailboxPullOneV1::new(mailbox_id, [6; 16], Vec::new(), now(), &reader)
            .unwrap();
    let commitment = pull.request_commitment().unwrap();
    let response = exchange_op(
        &app,
        &path,
        wire_mb::AnonymousMailboxTerminalFrameV1::PullOne(pull),
        wire_mb::AnonymousMailboxOperationV1::PullOne,
        [6; 16],
        commitment,
        target,
    )
    .await;
    assert!(response.sealed_payload.is_empty());
}

#[tokio::test]
async fn a_reply_opens_once_and_only_for_its_exchange() {
    // Two exchanges to the same node: swapping replies must fail, and the
    // session is consumed by the first open (enforced by `open(self)`).
    let directory = test_dir();
    let node = CoreKey::generate();
    let config = AnonymousMailboxStoreConfig {
        enabled: true,
        db_path: directory
            .path()
            .join("mailbox.db")
            .to_string_lossy()
            .into_owned(),
        ticket_issue_work_bits: 4,
        ..AnonymousMailboxStoreConfig::default()
    };
    let store: Arc<dyn AnonymousMailboxCustodyRepository> = Arc::new(
        SqliteAnonymousMailboxStore::open_with_ticket_issuer(config, node.clone(), [0x5B; 32])
            .unwrap(),
    );
    let app = build_chat_peer_router_with_anonymous_mailbox(
        Some(chat_relay(&directory)),
        Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        Arc::new(PeerStore::new()),
        Arc::new(node.clone()),
        Arc::new(reqwest::Client::new()),
        None,
        Some(store),
    );
    let target = node.public_key_bytes();
    let path = [OnionHop {
        node_id: target,
        kem_pub: onion_keys::current_public_key(),
    }];
    let reader = WireKey::generate();
    let frame = |id: u8| {
        wire_mb::encode_anonymous_mailbox_terminal_frame(
            &wire_mb::AnonymousMailboxTerminalFrameV1::PullOne(
                wire_mb::AnonymousMailboxPullOneV1::new(
                    [id; 32],
                    [id; 16],
                    Vec::new(),
                    now(),
                    &reader,
                )
                .unwrap(),
            ),
        )
        .unwrap()
    };
    let first = SourceExchange::prepare(&path, frame(1), now()).unwrap();
    let second = SourceExchange::prepare(&path, frame(2), now()).unwrap();
    let first_reply = post(&app, first.body_json()).await;
    let second_reply = post(&app, second.body_json()).await;
    assert!(
        first.open(&second_reply).is_err(),
        "a reply must not open under another exchange"
    );
    assert!(second.open(&second_reply).is_ok());
    let _ = first_reply;
}
