// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn verified_submit_dispatches_and_responds_with_memchain_storage_off() {
    // [CHAT-DISPATCH-STORAGE-DECOUPLING 2026-09-02 by Codex] Exercise the
    // real handler and encrypted UDP response with no MemPool, AOF writer,
    // MemoryStorage, or vector index, matching the production canary mode.
    let directory = tempfile::tempdir().expect("chat dispatch directory");
    let relay =
        test_chat_relay_service(&directory.path().join("chat-dispatch.sqlite3"), [0x51; 32]);
    let relay_option = Some(Arc::clone(&relay));
    let sender = IdentityKeyPair::generate();
    let node_identity = IdentityKeyPair::generate();
    let request =
        test_verified_submit_request(&sender, [0x52; 16], [0x53; 16], unix_now_secs(), 0x54);
    let client_udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let server_udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let session = Arc::new(crate::services::Session::new(
        aeronyx_common::types::SessionId::generate(),
        sender.public_key(),
        aeronyx_core::crypto::SessionKey::from_bytes([0x55; 32]),
        Ipv4Addr::new(100, 64, 0, 85),
        client_udp.local_addr().expect("client UDP address"),
    ));
    let sessions = Arc::new(SessionManager::new(16, Duration::from_secs(60)));
    let peer_store = Arc::new(PeerStore::new());
    let mut config = MemChainConfig::default();
    config.mode = MemChainMode::Off;
    assert!(!config.is_enabled());
    let no_storage = None;
    let no_vector_index = None;
    let crypto = DefaultTransportCrypto::new();

    Server::handle_memchain_message(
        MemChainMessage::ChatRelayVerifiedSubmitV1(request.clone()),
        None,
        None,
        &no_storage,
        &no_vector_index,
        &config,
        "unused-while-storage-off",
        &session,
        &server_udp,
        &crypto,
        &sessions,
        &relay_option,
        &peer_store,
        &node_identity.public_key_bytes(),
        &node_identity,
        None,
    )
    .await;

    let mut datagram = vec![0_u8; 65_535];
    let (received, _) =
        tokio::time::timeout(Duration::from_secs(2), client_udp.recv(&mut datagram))
            .await
            .expect("verified submit response timeout")
            .expect("receive verified submit response");
    let packet = aeronyx_core::protocol::codec::decode_data_packet(&datagram[..received])
        .expect("decode encrypted response packet");
    assert_eq!(packet.session_id, *session.id.as_bytes());
    let mut plaintext = vec![0_u8; packet.encrypted_payload.len()];
    let plaintext_len = crypto
        .decrypt(
            &session.session_key,
            packet.counter,
            session.id.as_bytes(),
            &packet.encrypted_payload,
            &mut plaintext,
        )
        .expect("decrypt verified submit response");
    plaintext.truncate(plaintext_len);
    assert_eq!(
        plaintext.first().copied(),
        Some(aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC)
    );
    let response = aeronyx_core::protocol::memchain::decode_memchain(&plaintext[1..])
        .expect("decode verified submit response");
    let MemChainMessage::ChatRelayVerifiedSubmitResponseV1(response) = response else {
        panic!("unexpected response variant");
    };
    response
        .validate_for_request(&request)
        .expect("response must bind exact verified submit");
    assert_eq!(response.result, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1);
    assert_eq!(relay.peer_status().verified_submit.total, 1);
}

#[tokio::test]
async fn verified_submit_correlates_entry_retry_and_session_rejection() {
    // [CHAT-VERIFIED-SUBMIT-HANDLER-CORRELATION 2026-08-23 by Codex]
    // Exercise the real client handler, durable entry custody, authenticated
    // session binding, and aggregate telemetry without requiring a live
    // peer. Both non-onion outcomes must remain safe to correlate with the
    // exact request and must never attach terminal receipt bytes.
    let directory = tempfile::tempdir().expect("verified submit handler directory");
    let relay = test_chat_relay_service(
        &directory.path().join("verified-submit.sqlite3"),
        [0x71; 32],
    );
    let relay_option = Some(Arc::clone(&relay));
    let sender = IdentityKeyPair::generate();
    let unrelated = IdentityKeyPair::generate();
    let source_node = IdentityKeyPair::generate();
    let now = unix_now_secs();
    let mut envelope = ChatEnvelope {
        message_id: [0x72; 16],
        sender: sender.public_key_bytes(),
        receiver: [0x73; 32],
        timestamp: now,
        ciphertext: b"opaque verified submit payload".to_vec(),
        nonce: [0x74; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    envelope.signature = sender.sign(&envelope.sign_data());
    let request = ChatRelayVerifiedSubmitRequestV1::signed([0x75; 16], envelope, now, &sender)
        .expect("sign verified submit request");
    let session = Arc::new(crate::services::Session::new(
        aeronyx_common::types::SessionId::generate(),
        sender.public_key(),
        aeronyx_core::crypto::SessionKey::from_bytes([0x76; 32]),
        Ipv4Addr::new(100, 64, 0, 76),
        "127.0.0.1:1076".parse().unwrap(),
    ));
    let store = PeerStore::new();

    let retry_response = Server::handle_verified_chat_submit(
        request.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(retry_response.result, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1);
    assert!(retry_response.terminal_receipt.is_none());
    retry_response
        .validate_for_request(&request)
        .expect("entry retry response must correlate");

    let replay_response = Server::handle_verified_chat_submit(
        request.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(replay_response, retry_response);

    let mut conflicting_envelope = request.envelope.clone();
    conflicting_envelope.message_id = [0x78; 16];
    conflicting_envelope.nonce = [0x79; 24];
    conflicting_envelope.signature = sender.sign(&conflicting_envelope.sign_data());
    let conflicting_request = ChatRelayVerifiedSubmitRequestV1::signed(
        request.request_id,
        conflicting_envelope,
        now,
        &sender,
    )
    .expect("sign conflicting verified submit request");
    let conflict_response = Server::handle_verified_chat_submit(
        conflicting_request.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(conflict_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    conflict_response
        .validate_for_request(&conflicting_request)
        .expect("conflict response must correlate without leaking prior response");

    let unrelated_session = Arc::new(crate::services::Session::new(
        aeronyx_common::types::SessionId::generate(),
        unrelated.public_key(),
        aeronyx_core::crypto::SessionKey::from_bytes([0x77; 32]),
        Ipv4Addr::new(100, 64, 0, 77),
        "127.0.0.1:1077".parse().unwrap(),
    ));
    let rejected_response = Server::handle_verified_chat_submit(
        request.clone(),
        &unrelated_session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(rejected_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(rejected_response.terminal_receipt.is_none());
    rejected_response
        .validate_for_request(&request)
        .expect("rejected response must correlate");

    let status = relay.peer_status().verified_submit;
    assert_eq!(status.total, 4);
    assert_eq!(status.entry_retry_total, 2);
    assert_eq!(status.rejected_total, 2);
    assert_eq!(status.unknown_result_total, 0);
    assert_eq!(status.replayed_total, 1);
    assert_eq!(status.request_conflict_total, 1);
    assert_eq!(status.last_result.as_deref(), Some("rejected"));
}

#[tokio::test]
async fn verified_submit_pending_and_capacity_reject_before_side_effects() {
    // [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex] A
    // crash-left exact reservation and a different request at capacity
    // must both stop before wallet-route mutation, network selection, or
    // entry custody. This is the live-handler proof of the durable gate.
    let directory = tempfile::tempdir().expect("verified submit admission directory");
    let mut relay_config = ChatRelayConfig::default();
    relay_config.enabled = true;
    relay_config.dedup_lru_capacity = 1;
    relay_config.db_path = directory
        .path()
        .join("verified-submit-admission.sqlite3")
        .to_string_lossy()
        .into_owned();
    let relay = Arc::new(
        ChatRelayService::new(relay_config, [0x81; 32])
            .expect("initialize verified submit admission relay"),
    );
    let relay_option = Some(Arc::clone(&relay));
    let sender = IdentityKeyPair::generate();
    let source_node = IdentityKeyPair::generate();
    let now = unix_now_secs();
    let make_request = |request_id: [u8; 16], message_id: [u8; 16]| {
        let mut envelope = ChatEnvelope {
            message_id,
            sender: sender.public_key_bytes(),
            receiver: [0x82; 32],
            timestamp: now,
            ciphertext: b"opaque admission payload".to_vec(),
            nonce: [0x83; 24],
            content_type: ChatContentType::Text,
            signature: [0_u8; 64],
        };
        envelope.signature = sender.sign(&envelope.sign_data());
        ChatRelayVerifiedSubmitRequestV1::signed(request_id, envelope, now, &sender)
            .expect("sign verified submit admission request")
    };
    let pending_request = make_request([0x84; 16], [0x85; 16]);
    assert_eq!(
        relay
            .reserve_verified_submit(&pending_request)
            .expect("seed crash-left pending reservation"),
        VerifiedSubmitAdmission::Reserved
    );
    let session = Arc::new(crate::services::Session::new(
        aeronyx_common::types::SessionId::generate(),
        sender.public_key(),
        aeronyx_core::crypto::SessionKey::from_bytes([0x86; 32]),
        Ipv4Addr::new(100, 64, 0, 86),
        "127.0.0.1:1086".parse().unwrap(),
    ));
    let store = PeerStore::new();

    let pending_response = Server::handle_verified_chat_submit(
        pending_request.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(pending_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    pending_response
        .validate_for_request(&pending_request)
        .expect("pending rejection remains request-bound");

    let saturated_request = make_request([0x87; 16], [0x88; 16]);
    let saturated_response = Server::handle_verified_chat_submit(
        saturated_request.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(saturated_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    saturated_response
        .validate_for_request(&saturated_request)
        .expect("capacity rejection remains request-bound");

    assert_eq!(
        relay
            .storage_usage()
            .expect("read admission-gated storage usage")
            .pending_messages,
        0
    );
    assert!(relay
        .wallet_routes
        .lookup(&sender.public_key_bytes())
        .is_empty());
    assert_eq!(relay.peer_status().outbound_rounds, 0);
    let verified_status = relay.peer_status().verified_submit;
    assert_eq!(verified_status.pending_rejected_total, 1);
    assert_eq!(verified_status.capacity_rejected_total, 1);
    assert_eq!(verified_status.rejected_total, 2);
}

#[tokio::test]
async fn verified_submit_restart_recovers_entry_without_reselecting_onion_path() {
    // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Exercise the
    // real handler after a process crash. The replacement process repeats
    // only idempotent encrypted entry custody and persists an exact retry
    // response without wallet-route or onion-network side effects.
    let directory = tempfile::tempdir().expect("verified recovery directory");
    let db_path = directory.path().join("verified-recovery.sqlite3");
    let mut relay_config = ChatRelayConfig::default();
    relay_config.enabled = true;
    relay_config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0x89; 32];
    let sender = IdentityKeyPair::generate();
    let source_node = IdentityKeyPair::generate();
    let now = unix_now_secs();
    let mut envelope = ChatEnvelope {
        message_id: [0x8A; 16],
        sender: sender.public_key_bytes(),
        receiver: [0x8B; 32],
        timestamp: now,
        ciphertext: b"opaque restart recovery payload".to_vec(),
        nonce: [0x8C; 24],
        content_type: ChatContentType::Text,
        signature: [0_u8; 64],
    };
    envelope.signature = sender.sign(&envelope.sign_data());
    let request = ChatRelayVerifiedSubmitRequestV1::signed([0x8D; 16], envelope, now, &sender)
        .expect("sign restart recovery request");

    {
        let predecessor = ChatRelayService::new(relay_config.clone(), secret)
            .expect("initialize predecessor verified-submit relay");
        assert_eq!(
            predecessor
                .reserve_verified_submit(&request)
                .expect("reserve predecessor verified submit"),
            VerifiedSubmitAdmission::Reserved
        );
        predecessor
            .store_pending(&request.envelope)
            .expect("persist predecessor entry custody");
    }
    let aged_owner =
        i64::try_from(now.saturating_sub(VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS + 1))
            .expect("convert aged verified-submit owner timestamp");
    rusqlite::Connection::open(&db_path)
        .expect("open restart recovery database")
        .execute(
            "UPDATE relay_verified_submit_reservations
             SET reserved_at = ?1, owner_acquired_at = ?1",
            rusqlite::params![aged_owner],
        )
        .expect("age predecessor verified-submit owner lease");

    let relay = Arc::new(
        ChatRelayService::new(relay_config, secret)
            .expect("initialize replacement verified-submit relay"),
    );
    let relay_option = Some(Arc::clone(&relay));
    let session = Arc::new(crate::services::Session::new(
        aeronyx_common::types::SessionId::generate(),
        sender.public_key(),
        aeronyx_core::crypto::SessionKey::from_bytes([0x8E; 32]),
        Ipv4Addr::new(100, 64, 0, 142),
        "127.0.0.1:1142".parse().unwrap(),
    ));
    let store = PeerStore::new();

    let recovered_response = Server::handle_verified_chat_submit(
        request.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(
        recovered_response.result,
        CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1
    );
    assert!(recovered_response.terminal_receipt.is_none());
    recovered_response
        .validate_for_request(&request)
        .expect("recovered entry response remains request-bound");
    assert_eq!(
        relay
            .storage_usage()
            .expect("read recovered entry storage")
            .pending_messages,
        1
    );
    assert_eq!(relay.peer_status().outbound_rounds, 0);
    assert!(relay
        .wallet_routes
        .lookup(&sender.public_key_bytes())
        .is_empty());

    let replay_response = Server::handle_verified_chat_submit(
        request,
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(replay_response, recovered_response);
    assert_eq!(
        relay
            .storage_usage()
            .expect("read exact replay storage")
            .pending_messages,
        1
    );
    let verified_status = relay.peer_status().verified_submit;
    assert_eq!(verified_status.replayed_total, 1);
    assert_eq!(verified_status.entry_recovery.attempted_total, 1);
    assert_eq!(verified_status.entry_recovery.completed_total, 1);
    assert_eq!(verified_status.entry_recovery.failed_total, 0);
    assert_eq!(verified_status.entry_recovery.deferred_total, 0);
    assert_eq!(
        verified_status.entry_recovery.last_outcome.as_deref(),
        Some("completed")
    );
}

#[tokio::test]
async fn verified_submit_stale_completed_replay_in_process_and_after_restart() {
    // [CHAT-VERIFIED-SUBMIT-STALE-REPLAY 2026-09-02 by Codex] An exact
    // completed result remains the sole authority for a request whose
    // signed timestamp is one second outside the admission window. Both
    // memory and durable recovery paths must return the original result
    // without repeating route selection or entry custody.
    let directory = tempfile::tempdir().expect("stale replay directory");
    let db_path = directory.path().join("stale-replay.sqlite3");
    let secret = [0x91; 32];
    let sender = IdentityKeyPair::generate();
    let source_node = IdentityKeyPair::generate();
    // Keep the synthetic completion inside the store's real-clock startup
    // retention gate while driving the request clock deterministically.
    // An epoch-era fixture would be correctly removed when the repository
    // is reopened, masking the restart replay boundary under test.
    let completed_at = unix_now_secs();
    let completed_response_ttl_secs = TIMESTAMP_WINDOW_SECS * 2 + 1;
    let request = test_verified_submit_request(
        &sender,
        [0x92; 16],
        [0x93; 16],
        completed_at.saturating_sub(TIMESTAMP_WINDOW_SECS + 1),
        0x94,
    );
    let session = test_verified_submit_session(&sender, 0x95);
    let store = PeerStore::new();
    let expected = {
        let relay = test_chat_relay_service(&db_path, secret);
        let expected = seed_completed_verified_submit(&relay, &request);
        rusqlite::Connection::open(&db_path)
            .expect("open deterministic stale replay database")
            .execute(
                "UPDATE relay_verified_submit_responses SET completed_at = ?1",
                rusqlite::params![completed_at],
            )
            .expect("set deterministic completion time");
        let relay_option = Some(Arc::clone(&relay));
        let replayed = Server::handle_verified_chat_submit_with_clock(
            request.clone(),
            &session,
            &relay_option,
            &store,
            &source_node.public_key_bytes(),
            &source_node,
            None,
            || completed_at,
        )
        .await;
        assert_eq!(replayed, expected);
        assert_no_verified_submit_delivery_effects(&relay, &sender);
        expected
    };

    let restarted = test_chat_relay_service(&db_path, secret);
    let restarted_option = Some(Arc::clone(&restarted));
    let replayed = Server::handle_verified_chat_submit_with_clock(
        request,
        &session,
        &restarted_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
        || completed_at.saturating_add(completed_response_ttl_secs),
    )
    .await;
    assert_eq!(replayed, expected);
    assert_no_verified_submit_delivery_effects(&restarted, &sender);
}

#[tokio::test]
async fn verified_submit_stale_completed_replay_freshness_and_retention_boundaries() {
    let directory = tempfile::tempdir().expect("verified submit boundary directory");
    let sender = IdentityKeyPair::generate();
    let source_node = IdentityKeyPair::generate();
    let session = test_verified_submit_session(&sender, 0x96);
    let store = PeerStore::new();
    // [VERIFIED-SUBMIT-CLOCK 2026-09-13 by Codex] Keep every boundary in
    // this test on one clock snapshot so a wall-clock tick cannot change it.
    let now = unix_now_secs();

    // The signed freshness window is symmetric. A request exactly sixty
    // seconds in the future is admitted and may create entry custody.
    let fresh_relay =
        test_chat_relay_service(&directory.path().join("fresh-boundary.sqlite3"), [0x97; 32]);
    let fresh_request = test_verified_submit_request(
        &sender,
        [0x98; 16],
        [0x99; 16],
        now.saturating_add(TIMESTAMP_WINDOW_SECS),
        0x9A,
    );
    fresh_request
        .verify_authentication()
        .expect("sixty-second boundary remains fresh");
    let fresh_option = Some(Arc::clone(&fresh_relay));
    let fresh_response = Server::handle_verified_chat_submit_with_clock(
        fresh_request.clone(),
        &session,
        &fresh_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
        || now,
    )
    .await;
    assert_eq!(fresh_response.result, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1);
    fresh_response
        .validate_for_request(&fresh_request)
        .expect("fresh boundary response remains request-bound");
    assert_eq!(
        fresh_relay
            .storage_usage()
            .expect("read fresh boundary custody")
            .pending_messages,
        1
    );

    // A completed response is still live at the inclusive 121-second
    // durable boundary. Restart clears the process-local cache so this
    // exercises the SQLite result before any new admission or custody.
    let durable_path = directory.path().join("retention-boundary.sqlite3");
    let secret = [0x9B; 32];
    let stale_request = test_verified_submit_request(
        &sender,
        [0x9C; 16],
        [0x9D; 16],
        now.saturating_sub(TIMESTAMP_WINDOW_SECS + 1),
        0x9E,
    );
    let expected = {
        let relay = test_chat_relay_service(&durable_path, secret);
        seed_completed_verified_submit(&relay, &stale_request)
    };
    let restarted = test_chat_relay_service(&durable_path, secret);
    let retention_boundary = i64::try_from(now.saturating_sub(TIMESTAMP_WINDOW_SECS * 2 + 1))
        .expect("retention boundary fits SQLite");
    rusqlite::Connection::open(&durable_path)
        .expect("open retention boundary database")
        .execute(
            "UPDATE relay_verified_submit_responses SET completed_at = ?1",
            rusqlite::params![retention_boundary],
        )
        .expect("age completed response to retention boundary");
    let restarted_option = Some(Arc::clone(&restarted));
    let boundary_response = Server::handle_verified_chat_submit_with_clock(
        stale_request,
        &session,
        &restarted_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
        || now,
    )
    .await;
    assert_eq!(boundary_response, expected);
    assert_no_verified_submit_delivery_effects(&restarted, &sender);

    // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] An in-memory
    // cache entry must not extend the durable 121-second replay lifetime.
    let expired_path = directory.path().join("expired-memory-cache.sqlite3");
    let expired_relay = test_chat_relay_service(&expired_path, [0xB1; 32]);
    let expired_request =
        test_verified_submit_request(&sender, [0xB2; 16], [0xB3; 16], 1_000, 0xB4);
    let expired_expected = seed_completed_verified_submit(&expired_relay, &expired_request);
    rusqlite::Connection::open(&expired_path)
        .expect("open expired replay database")
        .execute(
            "UPDATE relay_verified_submit_responses SET completed_at = 1000",
            [],
        )
        .expect("set expired completion time");
    let observer = rusqlite::Connection::open(&expired_path).expect("open replay observer");
    let before_data_version: i64 = observer
        .query_row("PRAGMA data_version", [], |row| row.get(0))
        .expect("read replay data version");
    let expired_option = Some(Arc::clone(&expired_relay));
    let expired_response = Server::handle_verified_chat_submit_with_clock(
        expired_request.clone(),
        &session,
        &expired_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
        || 1_122,
    )
    .await;
    assert_eq!(expired_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(matches!(
        expired_relay
            .verified_submit_cache_lookup(&expired_request)
            .expect("memory cache remains independently populated"),
        VerifiedSubmitCacheLookup::Exact(response) if response == expired_expected
    ));
    assert_eq!(
        observer
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .expect("recheck replay data version"),
        before_data_version
    );
    assert_no_verified_submit_delivery_effects(&expired_relay, &sender);

    // A correctly signed request beyond the future window cannot use an
    // existing completion as replay authority.
    let future_path = directory.path().join("future-completion.sqlite3");
    let future_relay = test_chat_relay_service(&future_path, [0xB5; 32]);
    let future_request = test_verified_submit_request(&sender, [0xB6; 16], [0xB7; 16], 2_000, 0xB8);
    let future_expected = seed_completed_verified_submit(&future_relay, &future_request);
    let future_option = Some(Arc::clone(&future_relay));
    let future_response = Server::handle_verified_chat_submit_with_clock(
        future_request.clone(),
        &session,
        &future_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
        || 1_000,
    )
    .await;
    assert_eq!(future_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(matches!(
        future_relay
            .verified_submit_cache_lookup(&future_request)
            .expect("future rejection must preserve prior completion"),
        VerifiedSubmitCacheLookup::Exact(response) if response == future_expected
    ));
    assert_no_verified_submit_delivery_effects(&future_relay, &sender);
}

#[tokio::test]
async fn verified_submit_stale_completed_replay_rejects_noncompleted_states() {
    let directory = tempfile::tempdir().expect("stale rejection directory");
    let relay = test_chat_relay_service(
        &directory.path().join("stale-rejections.sqlite3"),
        [0xA1; 32],
    );
    let relay_option = Some(Arc::clone(&relay));
    let sender = IdentityKeyPair::generate();
    let source_node = IdentityKeyPair::generate();
    let session = test_verified_submit_session(&sender, 0xA2);
    let store = PeerStore::new();
    let stale_timestamp = unix_now_secs().saturating_sub(TIMESTAMP_WINDOW_SECS + 1);

    let miss = test_verified_submit_request(&sender, [0xA3; 16], [0xA4; 16], stale_timestamp, 0xA5);
    let miss_response = Server::handle_verified_chat_submit(
        miss.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(miss_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(matches!(
        relay
            .verified_submit_cache_lookup(&miss)
            .expect("recheck stale miss"),
        VerifiedSubmitCacheLookup::Miss
    ));

    let pending =
        test_verified_submit_request(&sender, [0xA6; 16], [0xA7; 16], stale_timestamp, 0xA8);
    assert_eq!(
        relay
            .reserve_verified_submit(&pending)
            .expect("seed stale pending reservation"),
        VerifiedSubmitAdmission::Reserved
    );
    let pending_response = Server::handle_verified_chat_submit(
        pending.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(pending_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(matches!(
        relay
            .verified_submit_cache_lookup(&pending)
            .expect("recheck stale pending reservation"),
        VerifiedSubmitCacheLookup::Pending
    ));

    let completed =
        test_verified_submit_request(&sender, [0xA9; 16], [0xAA; 16], stale_timestamp, 0xAB);
    let completed_response = seed_completed_verified_submit(&relay, &completed);
    let conflict = test_verified_submit_request(
        &sender,
        completed.request_id,
        [0xAC; 16],
        stale_timestamp,
        0xAD,
    );
    let conflict_response = Server::handle_verified_chat_submit(
        conflict.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(conflict_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(matches!(
        relay
            .verified_submit_cache_lookup(&conflict)
            .expect("recheck stale conflict"),
        VerifiedSubmitCacheLookup::Conflict
    ));
    let VerifiedSubmitCacheLookup::Exact(still_completed) = relay
        .verified_submit_cache_lookup(&completed)
        .expect("retain original completed response")
    else {
        panic!("stale conflict must not replace completed response");
    };
    assert_eq!(still_completed, completed_response);

    let mut invalid_signature = completed.clone();
    invalid_signature.signature[0] ^= 0x01;
    let invalid_response = Server::handle_verified_chat_submit(
        invalid_signature.clone(),
        &session,
        &relay_option,
        &store,
        &source_node.public_key_bytes(),
        &source_node,
        None,
    )
    .await;
    assert_eq!(invalid_response.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    invalid_response
        .validate_for_request(&invalid_signature)
        .expect("invalid-signature rejection remains request-bound");

    assert_no_verified_submit_delivery_effects(&relay, &sender);
}
