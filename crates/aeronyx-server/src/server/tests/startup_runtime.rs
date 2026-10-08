// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

// [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] Shared by loader and
// descriptor regressions. Deliberately retain both legacy origins and peers
// in the mounted config so empty overrides cannot pass by testing defaults.
pub(super) fn phala_role_override_fixture() -> ServerConfig {
    let mut config: ServerConfig = toml::from_str(include_str!(
        "../../../../../deploy/node/server.phala.peer.example.toml",
    )).unwrap();
    config.discovery.public_endpoint = Some("https://old.aeronyx.network".into());
    config.network.public_endpoint = Some("https://fallback.aeronyx.network".into());
    config.discovery.seed_endpoints = vec!["https://seed.aeronyx.network".into()];
    config
}

// [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] Apply the same private
// role literals as Compose, without changing the test runner's environment.
pub(super) fn phala_private_role_environment(recovery_only: bool) -> Vec<(&'static str, String)> {
    let relay_id = hex::encode(
        IdentityKeyPair::from_bytes(&[48; 32]).unwrap().public_key_bytes(),
    );
    vec![
        ("AERONYX_DISCOVERY_PUBLIC_ENDPOINT", String::new()),
        ("AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR", String::new()),
        ("AERONYX_DISCOVERY_PUBLIC_DISCOVERY", "false".into()),
        ("AERONYX_DISCOVERY_SEED_ENDPOINTS", String::new()),
        ("AERONYX_DISCOVERY_PHALA_ATTESTATION_SOCKET_PATH", String::new()),
        ("AERONYX_REVERSE_ONION_RECIPIENT_ENABLED", "true".into()),
        ("AERONYX_REVERSE_ONION_RELAY_NODE_ID", relay_id),
        ("AERONYX_REVERSE_ONION_RELAY_ENDPOINT", "https://relay.aeronyx.network".into()),
        ("AERONYX_REVERSE_ONION_RECOVERY_ONLY", recovery_only.to_string()),
        ("AERONYX_PHALA_ONION_RELAY_ENABLED", "false".into()),
        ("AERONYX_PHALA_REVERSE_ONION_QUEUE_ENABLED", "false".into()),
        ("AERONYX_PHALA_REVERSE_ONION_QUEUE_RECOVERY_ONLY", "false".into()),
    ]
}

// [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] Execute only when the
// deferred test suite is authorized. The exact child test reads a fixture via
// ServerConfig::load; the success marker prevents a zero-matched-test false pass.
pub(super) fn run_phala_config_environment_case(
    full_test_name: &str,
    config: &ServerConfig,
    case: &str,
    overrides: &[(&str, String)],
) {
    let directory = tempfile::Builder::new()
        .prefix("phala-config-environment-")
        .tempdir_in("/Volumes/disk/aeronyx-codex-tmp")
        .unwrap();
    let path = directory.path().join("server.toml");
    std::fs::write(&path, toml::to_string(config).unwrap()).unwrap();
    let (_, test_name) = full_test_name.split_once("::").expect("crate-prefixed module path");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact").arg(test_name).arg("--nocapture").arg("--test-threads=1")
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_else(|| "/usr/bin:/bin".into()))
        .env("AERONYX_PHALA_CONFIG_TEST_CASE", case)
        .env("AERONYX_PHALA_CONFIG_TEST_PATH", &path)
        .envs(overrides.iter().map(|(name, value)| (*name, value)))
        .output()
        .expect("isolated config test child should start");
    assert!(output.status.success(), "{case}: {}\n{}",
        String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("PHALA_CONFIG_CASE_OK:{case}")),
        "child did not complete the selected assertions: {test_name} {case}");
}

// [PHALA-ROLE-ENDPOINT-OVERRIDE 2026-10-07 by Codex] Authored, unexecuted:
// cover the actual async loader with separate process environments, not just
// helper calls that could miss ordering or optional-env handling regressions.
#[tokio::test]
async fn phala_load_resolves_role_overrides_before_final_validation() {
    if let Ok(case) = std::env::var("AERONYX_PHALA_CONFIG_TEST_CASE") {
        let path = std::env::var("AERONYX_PHALA_CONFIG_TEST_PATH").unwrap();
        let loaded = tokio::time::timeout(Duration::from_secs(30), ServerConfig::load(&path))
            .await.expect("local configuration load should complete");
        match case.as_str() {
            "missing" => {
                let config = loaded.unwrap();
                assert_eq!(config.discovery.public_endpoint.as_deref(), Some("https://old.aeronyx.network"));
                assert_eq!(config.network.public_endpoint.as_deref(), Some("https://fallback.aeronyx.network"));
                assert_eq!(config.discovery.seed_endpoints, vec!["https://seed.aeronyx.network"]);
                assert!(config.discovery.phala_attestation_socket_path.is_some());
                assert!(!config.reverse_onion.recipient.enabled);
                assert_eq!(Server::discovery_startup_self_check(&config).0, "ready");
            }
            "private-live" | "private-recovery" => {
                let config = loaded.unwrap();
                assert!(config.discovery.public_endpoint.is_none());
                assert!(config.network.public_endpoint.is_none());
                assert!(config.effective_public_endpoint().is_none());
                assert!(config.discovery.public_api_listen_addr.is_none());
                assert!(config.discovery.phala_attestation_socket_path.is_none());
                assert!(config.discovery.seed_endpoints.is_empty());
                assert!(!config.discovery.public_discovery);
                assert!(config.reverse_onion.recipient.enabled);
                assert_eq!(config.reverse_onion.recipient.recovery_only, case == "private-recovery");
                assert_eq!(config.reverse_onion.recipient.relay_endpoint, "https://relay.aeronyx.network");
                assert!(config.memchain.chat_relay.enabled);
                assert!(config.blind_vault.enabled);
                let (status, detail) = Server::discovery_startup_self_check(&config);
                assert_eq!(status, "ready");
                assert!(detail.contains("pinned_relay_only"));
            }
            "public-listener" => {
                let config = loaded.unwrap();
                assert_eq!(config.discovery.public_api_listen_addr, Some("0.0.0.0:8422".parse().unwrap()));
                assert_eq!(config.effective_public_endpoint(), Some("https://new.aeronyx.network"));
                assert!(config.discovery.phala_attestation_socket_path.is_some());
                assert!(!config.reverse_onion.recipient.enabled);
            }
            "private-quote-conflict" => assert!(matches!(loaded,
                Err(ServerError::ConfigInvalid { field, .. }) if field == "discovery.phala_attestation_socket_path")),
            "missing-listener" | "disabled-discovery" | "whitespace-origin" => assert!(matches!(loaded,
                Err(ServerError::ConfigInvalid { field, .. }) if field == "AERONYX_DISCOVERY_PUBLIC_ENDPOINT")),
            _ => panic!("unknown Phala config case"),
        }
        println!("PHALA_CONFIG_CASE_OK:{case}");
        return;
    }

    let test_name = concat!(module_path!(), "::phala_load_resolves_role_overrides_before_final_validation");
    let base = phala_role_override_fixture();
    run_phala_config_environment_case(test_name, &base, "missing", &[]);
    for recovery_only in [false, true] {
        run_phala_config_environment_case(test_name, &base,
            if recovery_only { "private-recovery" } else { "private-live" },
            &phala_private_role_environment(recovery_only));
    }
    let mut no_listener = base.clone();
    no_listener.discovery.public_api_listen_addr = None;
    let origin = ("AERONYX_DISCOVERY_PUBLIC_ENDPOINT", "https://new.aeronyx.network".into());
    run_phala_config_environment_case(test_name, &no_listener, "public-listener", &[
        origin.clone(), ("AERONYX_DISCOVERY_PUBLIC_API_LISTEN_ADDR", "0.0.0.0:8422".into()),
    ]);
    run_phala_config_environment_case(test_name, &no_listener, "missing-listener", &[origin.clone()]);
    let mut disabled = base.clone();
    disabled.discovery.enabled = false;
    run_phala_config_environment_case(test_name, &disabled, "disabled-discovery", &[origin]);
    for endpoint in [" ", " https://new.aeronyx.network", "https://new.aeronyx.network "] {
        run_phala_config_environment_case(test_name, &base, "whitespace-origin", &[
            ("AERONYX_DISCOVERY_PUBLIC_ENDPOINT", endpoint.into()),
        ]);
    }
    let mut quote_conflict = phala_private_role_environment(false);
    quote_conflict.retain(|(name, _)| *name != "AERONYX_DISCOVERY_PHALA_ATTESTATION_SOCKET_PATH");
    run_phala_config_environment_case(test_name, &base, "private-quote-conflict", &quote_conflict);
}

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
    // [MEMCHAIN-SEALED-P2P 2026-10-05 by Codex] Appended sealed-V2 P2P
    // variants require the record-store gate, never the legacy AOF gate.
    let sealed_replica = aeronyx_core::protocol::memchain::SealedMemoryV2ReplicaV1 {
        record_id: [0x4a; 32],
        owner: sender.public_key_bytes(),
        created_at: now,
        envelope: vec![0; aeronyx_core::ledger::record::MEMORY_SEALED_V2_MIN_ENVELOPE_BYTES],
        signature: [0; 64],
    };
    let sealed_messages = [
        MemChainMessage::BroadcastSealedMemoryV2ReplicaV1(sealed_replica),
        MemChainMessage::SyncSealedMemoryV2RequestV1 {
            owner: sender.public_key_bytes(),
            after_record_id: None,
            limit: 8,
        },
        MemChainMessage::SyncSealedMemoryV2ResponseV1 {
            owner: sender.public_key_bytes(),
            after_record_id: None,
            records: vec![],
            next_cursor: None,
        },
    ];
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
    for message in &sealed_messages {
        let requirement = MemChainStorageRequirement::for_message(message);
        assert_eq!(requirement, MemChainStorageRequirement::RecordStore);
        assert!(requirement
            .authorize(Some(&mempool), Some(&aof_writer), &record_storage)
            .is_ok());
    }
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
fn configured_supernode_requires_client_to_tee_e2ee_transport() {
    // [MEMCHAIN-PHALA-E2EE-BOUNDARY 2026-10-06 by Codex] Disabled remains
    // backward compatible; enabled server-side inference is refused until
    // request and response fields are encrypted to the source client.
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
            .expect("SuperNode model IO requires source-bound E2EE")
            .to_string(),
        "Server failed to start: SuperNode initialization failed (client_to_tee_e2ee_transport_unavailable)"
    );

    let missing_environment = "AERONYX_TEST_SUPERNODE_SECRET_MUST_NOT_EXIST_20260814";
    let mut missing_secret_config = ServerConfig::default();
    missing_secret_config.memchain.supernode.enabled = true;
    missing_secret_config.memchain.supernode.accepted_compose_hashes =
        vec![format!("sha256:{}", "a".repeat(64))];
    missing_secret_config.memchain.supernode.providers =
        vec![crate::config_supernode::ProviderConfig {
            name: "private-provider-name".into(),
            provider_type: crate::config_supernode::ProviderType::PhalaAci,
            api_base: String::new(),
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
    assert!(rendered.contains("SuperNode initialization failed (client_to_tee_e2ee_transport_unavailable)"));
    assert!(!rendered.contains(missing_environment));
    assert!(!rendered.contains("private-provider-name"));
    assert!(!rendered.contains("private-provider.example.com"));

    let mut malformed_base_config = ServerConfig::default();
    malformed_base_config.memchain.supernode.enabled = true;
    malformed_base_config.memchain.supernode.accepted_compose_hashes =
        vec![format!("sha256:{}", "a".repeat(64))];
    malformed_base_config.memchain.supernode.providers =
        vec![crate::config_supernode::ProviderConfig {
            name: "local".into(),
            provider_type: crate::config_supernode::ProviderType::PhalaAci,
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
        "Server failed to start: SuperNode initialization failed (client_to_tee_e2ee_transport_unavailable)"
    );

    // [MEMCHAIN-PHALA-E2EE-BOUNDARY 2026-10-06 by Codex] Even a correctly
    // configured ACI verifier does not provide client-bound payload secrecy.
    let mut phala_config = ServerConfig::default();
    phala_config.memchain.supernode.enabled = true;
    phala_config.memchain.supernode.accepted_compose_hashes =
        vec![format!("sha256:{}", "a".repeat(64))];
    phala_config.memchain.supernode.providers =
        vec![crate::config_supernode::ProviderConfig {
            name: "phala".into(),
            provider_type: crate::config_supernode::ProviderType::PhalaAci,
            api_base: String::new(),
            api_key: Some("test-only-secret".into()),
            model: "test-model".into(),
            max_tokens: None,
            temperature: None,
        }];
    let phala_server = Server::new(phala_config, IdentityKeyPair::generate(), None);
    assert_eq!(
        phala_server
            .init_llm_router()
            .err()
            .expect("server-side Phala inference without source E2EE must reject startup")
            .to_string(),
        "Server failed to start: SuperNode initialization failed (client_to_tee_e2ee_transport_unavailable)"
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
