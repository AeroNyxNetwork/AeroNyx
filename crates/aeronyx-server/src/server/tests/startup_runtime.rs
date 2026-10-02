// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn storage_disabled_dispatch_gate_preserves_chat_and_rejects_storage_only_variants() {
    // [CHAT-DISPATCH-STORAGE-DECOUPLING 2026-09-02 by Codex] Classification
    // is closed over every storage-backed legacy family while only the
    // explicit chat handler variants remain admissible without storage.
    let sender = IdentityKeyPair::generate();
    let now = unix_now_secs();
    let verified = test_verified_submit_request(&sender, [0x41; 16], [0x42; 16], now, 0x43);
    let chat_runtime_messages = vec![
        MemChainMessage::ChatRelay(verified.envelope.clone()),
        MemChainMessage::ChatRelayVerifiedSubmitV1(verified.clone()),
        MemChainMessage::ChatPull {
            wallet: sender.public_key_bytes(),
            after_timestamp: 0,
            cursor: [0; 16],
            limit: 1,
            request_timestamp: now,
            signature: [0; 64],
        },
        MemChainMessage::ChatPullV2 {
            wallet: sender.public_key_bytes(),
            after_timestamp: 0,
            cursor: Vec::new(),
            limit: 1,
            request_timestamp: now,
            signature: [0; 64],
        },
        MemChainMessage::ChatAck {
            message_ids: vec![[0x44; 16]],
            wallet: sender.public_key_bytes(),
            ack_timestamp: now,
            signature: [0; 64],
        },
    ];
    let no_record_storage = None;
    for message in &chat_runtime_messages {
        let requirement = MemChainStorageRequirement::for_message(message);
        assert_eq!(requirement, MemChainStorageRequirement::IndependentRuntime);
        assert!(requirement
            .authorize(None, None, &no_record_storage)
            .is_ok());
    }

    let fact_message = MemChainMessage::SyncRequest {
        last_known_hash: [0x46; 32],
    };
    let record_message = MemChainMessage::SyncRecordRequest {
        owner: [0x47; 32],
        after_timestamp: now,
    };
    let legacy_block_message = MemChainMessage::BlockAnnounce(aeronyx_core::ledger::BlockHeader {
        height: 1,
        timestamp: now,
        prev_block_hash: GENESIS_PREV_HASH,
        merkle_root: [0x48; 32],
        block_type: 1,
    });
    let commitment_block =
        RecordCommitmentBlockV1::new_signed(1, now, GENESIS_PREV_HASH, vec![[0x49; 32]], &sender);
    let record_block_message = MemChainMessage::RecordBlockAnnounceV1 {
        header: commitment_block.header,
        proposer_signature: commitment_block.proposer_signature,
    };
    assert!(matches!(
        MemChainStorageRequirement::for_message(&fact_message).authorize(
            None,
            None,
            &no_record_storage,
        ),
        Err(MemChainDispatchGateError::FactAofUnavailable)
    ));
    assert!(matches!(
        MemChainStorageRequirement::for_message(&record_message).authorize(
            None,
            None,
            &no_record_storage,
        ),
        Err(MemChainDispatchGateError::FactAofUnavailable)
    ));

    let partial_directory = tempfile::tempdir().expect("partial runtime directory");
    let mempool = Arc::new(MemPool::new());
    let aof_writer = Arc::new(TokioMutex::new(
        AofWriter::open(partial_directory.path().join("partial-runtime.aof"))
            .await
            .expect("open partial runtime AOF"),
    ));
    let record_storage = Some(Arc::new(
        MemoryStorage::open(":memory:", None).expect("open partial record store"),
    ));
    let record_requirement = MemChainStorageRequirement::for_message(&record_message);
    for (partial_mempool, partial_aof, partial_storage, expected_error) in [
        (
            None,
            None,
            &record_storage,
            MemChainDispatchGateError::FactAofUnavailable,
        ),
        (
            Some(&mempool),
            None,
            &record_storage,
            MemChainDispatchGateError::FactAofUnavailable,
        ),
        (
            None,
            Some(&aof_writer),
            &record_storage,
            MemChainDispatchGateError::FactAofUnavailable,
        ),
        (
            Some(&mempool),
            Some(&aof_writer),
            &no_record_storage,
            MemChainDispatchGateError::RecordStoreUnavailable,
        ),
    ] {
        assert_eq!(
            record_requirement
                .authorize(partial_mempool, partial_aof, partial_storage)
                .err(),
            Some(expected_error),
            "partial record runtime must remain rejected"
        );
    }
    assert!(record_requirement
        .authorize(Some(&mempool), Some(&aof_writer), &record_storage,)
        .is_ok());
    for message in [&legacy_block_message, &record_block_message] {
        let requirement = MemChainStorageRequirement::for_message(message);
        assert_eq!(requirement, MemChainStorageRequirement::FactAof);
        assert!(matches!(
            requirement.authorize(None, None, &no_record_storage),
            Err(MemChainDispatchGateError::FactAofUnavailable)
        ));
    }
}

#[test]
fn configured_supernode_initialization_is_atomic_and_privacy_safe() {
    // [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] Disabled remains
    // backward compatible, while every explicitly configured provider is
    // required. Startup errors expose only the typed aggregate reason.
    let disabled = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    assert!(disabled.init_llm_router().unwrap().is_none());

    let mut disabled_memchain_config = ServerConfig::default();
    disabled_memchain_config.memchain.mode = crate::config_memchain::MemChainMode::Off;
    disabled_memchain_config.memchain.supernode.enabled = true;
    let disabled_memchain_server =
        Server::new(disabled_memchain_config, IdentityKeyPair::generate(), None);
    assert_eq!(
        disabled_memchain_server
            .init_llm_router()
            .err()
            .expect("SuperNode requires an active MemChain runtime")
            .to_string(),
        "Server failed to start: SuperNode initialization failed (memchain_runtime_required)"
    );

    let missing_environment = "AERONYX_TEST_SUPERNODE_SECRET_MUST_NOT_EXIST_20260814";
    let mut missing_secret_config = ServerConfig::default();
    missing_secret_config.memchain.supernode.enabled = true;
    missing_secret_config.memchain.supernode.providers =
        vec![crate::config_supernode::ProviderConfig {
            name: "private-provider-name".into(),
            provider_type: crate::config_supernode::ProviderType::OpenaiCompatible,
            api_base: "https://private-provider.example.com/v1".into(),
            api_key: Some(format!("${missing_environment}")),
            model: "test-model".into(),
            max_tokens: None,
            temperature: None,
        }];
    let missing_secret_server =
        Server::new(missing_secret_config, IdentityKeyPair::generate(), None);
    let error = missing_secret_server
        .init_llm_router()
        .err()
        .expect("missing configured secret must reject startup");
    let rendered = error.to_string();
    assert!(rendered.contains("SuperNode initialization failed (provider_secret_unavailable)"));
    assert!(!rendered.contains(missing_environment));
    assert!(!rendered.contains("private-provider-name"));
    assert!(!rendered.contains("private-provider.example.com"));

    let mut malformed_base_config = ServerConfig::default();
    malformed_base_config.memchain.supernode.enabled = true;
    malformed_base_config.memchain.supernode.providers =
        vec![crate::config_supernode::ProviderConfig {
            name: "local".into(),
            provider_type: crate::config_supernode::ProviderType::OpenaiCompatible,
            api_base: "http://localhost:11434/v1?private=1".into(),
            api_key: None,
            model: "test-model".into(),
            max_tokens: None,
            temperature: None,
        }];
    let malformed_base_server =
        Server::new(malformed_base_config, IdentityKeyPair::generate(), None);
    let malformed_error = malformed_base_server
        .init_llm_router()
        .err()
        .expect("ambiguous provider URL must reject startup");
    assert_eq!(
        malformed_error.to_string(),
        "Server failed to start: SuperNode initialization failed (invalid_api_base)"
    );

    let mut keyless_local_config = ServerConfig::default();
    keyless_local_config.memchain.supernode.enabled = true;
    keyless_local_config.memchain.supernode.providers =
        vec![crate::config_supernode::ProviderConfig {
            name: "local".into(),
            provider_type: crate::config_supernode::ProviderType::OpenaiCompatible,
            api_base: "http://localhost:11434/v1".into(),
            api_key: None,
            model: "test-model".into(),
            max_tokens: None,
            temperature: None,
        }];
    let keyless_local_server = Server::new(keyless_local_config, IdentityKeyPair::generate(), None);
    assert_eq!(
        keyless_local_server
            .init_llm_router()
            .unwrap()
            .unwrap()
            .provider_count(),
        1
    );
}

#[test]
fn follower_certificate_deferral_retry_is_bounded_and_domain_isolated() {
    // [FOLLOWER-CERTIFICATE-RETRY 2026-07-30 by Codex] A deferred
    // certificate retries promptly, but repeated local churn converges to
    // the normal interval. Block backlog retains independent priority.
    let stable = CommitmentFollowerRoundOutcome {
        inserted: 0,
        remote_tip_height: 8,
        block_backlog_remaining: false,
        certificate_retry_pending: false,
        certified_recovered: false,
    };
    assert_eq!(commitment_follower_success_retry_delay(30, &stable, 99), 30);

    let deferred = CommitmentFollowerRoundOutcome {
        certificate_retry_pending: true,
        ..stable
    };
    assert_eq!(commitment_follower_success_retry_delay(30, &deferred, 0), 1);
    assert_eq!(commitment_follower_success_retry_delay(30, &deferred, 1), 1);
    assert_eq!(commitment_follower_success_retry_delay(30, &deferred, 2), 2);
    assert_eq!(commitment_follower_success_retry_delay(30, &deferred, 3), 4);
    assert_eq!(
        commitment_follower_success_retry_delay(30, &deferred, 32),
        30
    );

    let backlog = CommitmentFollowerRoundOutcome {
        block_backlog_remaining: true,
        ..deferred
    };
    assert_eq!(commitment_follower_success_retry_delay(30, &backlog, 32), 1);
    assert_eq!(
        commitment_follower_success_retry_delay(5, &deferred, u32::MAX),
        5
    );
}

#[test]
fn coordinator_lease_degraded_retry_converges_without_hot_looping() {
    assert_eq!(
        commitment_coordinator_lease_degraded_retry_delay(40, true, Some(65)),
        32
    );
    assert_eq!(
        commitment_coordinator_lease_degraded_retry_delay(40, true, Some(9)),
        4
    );
    assert_eq!(
        commitment_coordinator_lease_degraded_retry_delay(40, true, Some(1)),
        1
    );
    assert_eq!(
        commitment_coordinator_lease_degraded_retry_delay(40, false, Some(0)),
        10
    );
    assert_eq!(
        commitment_coordinator_lease_degraded_retry_delay(5, false, None),
        5
    );
}

#[tokio::test]
async fn completed_half_open_progress_survives_abrupt_process_restart() {
    // [DIRECT-RELAY-HALF-OPEN-PROGRESS 2026-08-15 by Codex] This drill uses
    // three processes so neither the first successful probe nor the final
    // closed state can be inherited from process-local memory.
    let directory = tempfile::tempdir().expect("half-open progress drill directory");
    let db_path = directory
        .path()
        .join("direct-relay-half-open-progress.sqlite3");

    let crashed =
        run_direct_relay_restart_drill_child("seed_half_open_progress_crash", &db_path, None, None)
            .await;
    assert_restart_drill_child_crashed(&crashed);

    let resumed =
        run_direct_relay_restart_drill_child("resume_half_open_progress", &db_path, None, None)
            .await;
    assert_restart_drill_child_succeeded("resumed half-open progress", &resumed);

    let verified =
        run_direct_relay_restart_drill_child("verify_closed_restart", &db_path, None, None).await;
    assert_restart_drill_child_succeeded("closed circuit restart", &verified);
}

#[tokio::test]
async fn missing_checkpoint_table_rejects_fresh_process_activation() {
    // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] The first child
    // installs the checkpoint and sentinel, destroys only the checkpoint
    // table, then exits abruptly. A new process must observe corruption
    // before constructing any active ChatRelayService.
    let directory = tempfile::tempdir().expect("checkpoint loss drill directory");
    let db_path = directory
        .path()
        .join("direct-relay-checkpoint-loss.sqlite3");

    let crashed = run_direct_relay_restart_drill_child(
        "seed_checkpoint_table_loss_crash",
        &db_path,
        None,
        None,
    )
    .await;
    assert_restart_drill_child_crashed(&crashed);

    let rejected = run_direct_relay_restart_drill_child(
        "verify_missing_checkpoint_table_restart",
        &db_path,
        None,
        None,
    )
    .await;
    assert_restart_drill_child_succeeded("missing checkpoint table restart", &rejected);
}

#[tokio::test]
async fn directory_chain_startup_reconciles_and_reopens_audited_tip() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-directory-chain-{unique}.db"));
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.directory_chain_path = Some(path.to_string_lossy().to_string());
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());
    let identity = IdentityKeyPair::generate();
    let producer = identity.public_key_bytes();
    let server = Server::new(config, identity, None);
    let peer_http_clients = PeerHttpClients::build(&server.config).unwrap();
    let peer_store = server
        .init_peer_store(false, peer_http_clients.control.as_ref())
        .await
        .unwrap();

    let store = server
        .init_directory_chain(&peer_store)
        .await
        .unwrap()
        .expect("configured Directory Chain store");
    let first_audit = store.audit(unix_now_secs()).unwrap();
    assert_eq!(first_audit.blocks, 1);
    assert!(first_audit.commitments >= 1);
    drop(store);

    let (reopened, second_audit) =
        DirectoryChainStore::open(&path, producer, unix_now_secs()).unwrap();
    assert_eq!(second_audit, first_audit);
    assert!(reopened.block(second_audit.tip_height).unwrap().is_some());
    drop(reopened);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(path.with_extension("db-wal"));
    let _ = std::fs::remove_file(path.with_extension("db-shm"));
}

#[tokio::test]
async fn endpoint_evidence_foreign_schema_fails_startup_closed() {
    std::fs::create_dir_all("target/test-temp").expect("external-disk test temp root");
    let directory = tempfile::TempDir::new_in("target/test-temp").expect("private directory");
    let path = directory.path().join("foreign.sqlite3");
    rusqlite::Connection::open(&path)
        .expect("foreign database")
        .execute_batch("CREATE TABLE unrelated(value INTEGER);")
        .expect("foreign schema");
    let mut discovery = DiscoveryConfig::default();
    discovery.permissionless_endpoint_proof_enabled = true;
    discovery.permissionless_endpoint_evidence_enabled = true;
    discovery.permissionless_endpoint_evidence_db_path = path.to_string_lossy().into_owned();

    assert!(
        open_endpoint_evidence_store(&discovery, &IdentityKeyPair::generate())
            .await
            .is_err()
    );
    let connection = rusqlite::Connection::open(path).expect("foreign database remains");
    let mailbox_objects: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'discovery_endpoint_%'",
            [],
            |row| row.get(0),
        )
        .expect("schema count");
    assert_eq!(mailbox_objects, 0);
}

#[tokio::test]
async fn endpoint_attestation_inbox_foreign_schema_fails_startup_closed() {
    std::fs::create_dir_all("target/test-temp").expect("external-disk test temp root");
    let directory = tempfile::TempDir::new_in("target/test-temp").expect("private directory");
    let path = directory.path().join("foreign-attestations.sqlite3");
    rusqlite::Connection::open(&path)
        .expect("foreign database")
        .execute_batch("CREATE TABLE unrelated(value INTEGER);")
        .expect("foreign schema");
    let mut discovery = DiscoveryConfig::default();
    discovery.permissionless_endpoint_attestation_inbox_enabled = true;
    discovery.permissionless_endpoint_attestation_inbox_db_path =
        path.to_string_lossy().into_owned();

    assert!(open_endpoint_attestation_inbox(&discovery).await.is_err());
}
