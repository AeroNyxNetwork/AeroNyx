// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn commitment_sync_liveness_guard_marks_unexpected_task_exit_stopped() {
    // [FOLLOWER-TASK-LIVENESS 2026-07-30 by Codex] Drop is the common
    // cleanup path for normal return, panic unwinding, and Tokio abort.
    // The guard must revoke readiness without changing the legacy role.
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    storage.configure_record_commitment_sync(false, true);
    storage.record_commitment_sync_checkpoint_success(100, 4);
    assert_eq!(storage.record_commitment_sync_status().state, "current");

    {
        let _liveness_guard = CommitmentSyncTaskLivenessGuard::new(Arc::clone(&storage));
    }

    let stopped = storage.record_commitment_sync_status();
    assert_eq!(stopped.role, "follower");
    assert_eq!(stopped.state, "stopped");
    assert_eq!(stopped.follower_readiness_state, "stopped");
    assert!(!stopped.follower_fully_ready);
}

#[test]
fn configured_commitment_sync_task_construction_fails_closed() {
    // [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] Programmatic
    // callers and future reload paths must receive an error rather than a
    // healthy process with no required follower task. Each failure also
    // leaves one fixed, identity-blind operational code.
    let assert_startup_failure = |config: ServerConfig,
                                  identity: IdentityKeyPair,
                                  expected_error: &str,
                                  expected_status_code: &str| {
        let server = Server::new(config, identity, None);
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.configure_record_commitment_sync(false, true);
        let (_tip_tx, tip_rx) = tokio::sync::mpsc::channel(1);
        let error = server
            .spawn_memchain_commitment_sync_task(
                Arc::clone(&storage),
                Arc::new(PeerStore::new()),
                tip_rx,
                test_peer_http_client(),
            )
            .err()
            .expect("configured invalid follower must fail startup");
        assert!(error.to_string().contains(expected_error));
        let status = storage.record_commitment_sync_status();
        assert_eq!(status.state, "backoff");
        assert_eq!(
            status.last_error_code.as_deref(),
            Some(expected_status_code)
        );
    };

    let mut missing_coordinator = ServerConfig::default();
    missing_coordinator.memchain.commitment_sync_enabled = true;
    assert_startup_failure(
        missing_coordinator,
        IdentityKeyPair::generate(),
        "invalid pinned coordinator",
        "invalid_pinned_coordinator",
    );

    let self_identity = IdentityKeyPair::generate();
    let mut self_coordinator = ServerConfig::default();
    self_coordinator.memchain.commitment_sync_enabled = true;
    self_coordinator.memchain.commitment_coordinator_node_id =
        hex::encode(self_identity.public_key_bytes());
    assert_startup_failure(
        self_coordinator,
        self_identity,
        "coordinator cannot follow itself",
        "coordinator_self_reference",
    );

    let invalid_carrier_identity = IdentityKeyPair::generate();
    let mut distinct_coordinator = invalid_carrier_identity.public_key_bytes();
    distinct_coordinator[0] ^= 0x01;
    let mut invalid_carrier = ServerConfig::default();
    invalid_carrier.memchain.commitment_sync_enabled = true;
    invalid_carrier.memchain.commitment_coordinator_node_id = hex::encode(distinct_coordinator);
    invalid_carrier
        .memchain
        .commitment_authority_carrier_node_ids = vec!["not-a-node-id".into()];
    assert_startup_failure(
        invalid_carrier,
        invalid_carrier_identity,
        "memchain.commitment_authority_carrier_node_ids",
        "invalid_authority_carrier_policy",
    );
}

#[tokio::test]
async fn newer_commitment_tip_supersedes_inflight_announcement() {
    let (tip_tx, mut tip_rx) = tokio::sync::mpsc::channel(1);
    tip_tx.send(11).await.unwrap();

    let outcome =
        await_commitment_tip_announcement_or_newer(10, &mut tip_rx, std::future::pending::<u8>())
            .await;

    assert_eq!(
        outcome,
        CommitmentTipAnnouncementWaitOutcome::Superseded(11)
    );
}

#[tokio::test]
async fn stale_commitment_tip_does_not_supersede_inflight_announcement() {
    let (tip_tx, mut tip_rx) = tokio::sync::mpsc::channel(1);
    tip_tx.send(10).await.unwrap();

    let outcome = await_commitment_tip_announcement_or_newer(10, &mut tip_rx, async {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        42u8
    })
    .await;

    assert_eq!(outcome, CommitmentTipAnnouncementWaitOutcome::Completed(42));
}

#[tokio::test]
async fn newer_commitment_tip_supersedes_slow_http_announcement_and_delivers_latest() {
    let now = unix_now_secs();
    let coordinator = IdentityKeyPair::generate();
    let follower = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    storage.configure_record_commitment_sync(true, false);
    let first_block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0xA1; 32]],
        &coordinator,
    );
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let second_block = RecordCommitmentBlockV1::new_signed(
        2,
        now.saturating_add(1),
        first_block.hash(),
        vec![[0xA2; 32]],
        &coordinator,
    );

    let first_request_seen = Arc::new(tokio::sync::Notify::new());
    let handler_first_request_seen = Arc::clone(&first_request_seen);
    let received_heights = Arc::new(tokio::sync::Mutex::new(Vec::<u64>::new()));
    let handler_received_heights = Arc::clone(&received_heights);
    let coordinator_id = coordinator.public_key_bytes();
    let router = Router::new().route(
        "/api/memchain/peer/block-announce",
        post(move |body: axum::body::Bytes| {
            let first_request_seen = Arc::clone(&handler_first_request_seen);
            let received_heights = Arc::clone(&handler_received_heights);
            async move {
                assert_eq!(
                    body.first().copied(),
                    Some(aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC)
                );
                let message = aeronyx_core::protocol::memchain::decode_memchain(&body[1..])
                    .expect("announcement frame must decode");
                let (header, proposer_signature) = match message {
                    aeronyx_core::protocol::memchain::MemChainMessage::RecordBlockAnnounceV1 {
                        header,
                        proposer_signature,
                    } => (header, proposer_signature),
                    _ => panic!("expected commitment tip announcement"),
                };
                assert_eq!(header.proposer, coordinator_id);
                IdentityPublicKey::from_bytes(&header.proposer)
                    .unwrap()
                    .verify(&header.hash(), &proposer_signature)
                    .unwrap();
                let height = header.height;
                received_heights.lock().await.push(height);
                if height == 1 {
                    first_request_seen.notify_one();
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    axum::http::StatusCode::SERVICE_UNAVAILABLE
                } else {
                    axum::http::StatusCode::ACCEPTED
                }
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let http_server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let mut descriptor = NodeDescriptor::new(
        follower.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now.saturating_add(600),
        "tip-supersession-http-test",
    );
    descriptor.public_endpoint = Some(format!("http://{address}"));
    descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
    let descriptor = SignedNodeDescriptor::sign(descriptor, &follower).unwrap();
    let peer_store = Arc::new(PeerStore::new());
    let import = peer_store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
        now,
    );
    assert_eq!(import.inserted, 1);

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();
    let (tip_tx, mut tip_rx) = tokio::sync::mpsc::channel(1);
    let advance_storage = Arc::clone(&storage);
    let advance_tip = tokio::spawn(async move {
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            first_request_seen.notified(),
        )
        .await
        .expect("first HTTP announcement must arrive");
        advance_storage
            .append_record_commitment_block(&second_block, None)
            .await
            .unwrap();
        tip_tx.send(2).await.unwrap();
    });

    let superseded = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        await_commitment_tip_announcement_or_newer(
            1,
            &mut tip_rx,
            announce_current_record_commitment_tip_for_test(
                &storage,
                &peer_store,
                &coordinator,
                &client,
                &[follower.public_key_bytes()],
                3,
                std::time::Duration::from_millis(10),
            ),
        ),
    )
    .await
    .expect("new tip must cancel the slow HTTP round");
    assert_eq!(
        superseded,
        CommitmentTipAnnouncementWaitOutcome::Superseded(2)
    );
    advance_tip.await.unwrap();
    storage.record_commitment_outbound_announcement_superseded(now.saturating_add(2));
    let after_supersession = storage.record_commitment_sync_status();
    assert_eq!(after_supersession.outbound_announcement_rounds_total, 1);
    assert_eq!(
        after_supersession.outbound_announcement_rounds_superseded_total,
        1
    );
    assert_eq!(after_supersession.outbound_announcements_attempted_total, 0);

    let latest = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        await_commitment_tip_announcement_or_newer(
            2,
            &mut tip_rx,
            announce_current_record_commitment_tip_for_test(
                &storage,
                &peer_store,
                &coordinator,
                &client,
                &[follower.public_key_bytes()],
                1,
                std::time::Duration::ZERO,
            ),
        ),
    )
    .await
    .expect("latest HTTP announcement must complete");
    let CommitmentTipAnnouncementWaitOutcome::Completed(Ok(delivery)) = latest else {
        panic!("latest tip must produce a delivery outcome");
    };
    assert_eq!(delivery.announced_height, 2);
    assert_eq!(delivery.attempted, 1);
    assert_eq!(delivery.accepted, 1);
    assert_eq!(delivery.failed, 0);
    storage.record_commitment_outbound_announcement(
        now.saturating_add(3),
        delivery.announced_height,
        delivery.attempted,
        delivery.accepted,
        delivery.stale,
        delivery.failed,
        delivery.retries_attempted,
        delivery.retries_succeeded,
        delivery.retries_exhausted,
    );
    let status = storage.record_commitment_sync_status();
    assert_eq!(status.outbound_announcement_rounds_total, 2);
    assert_eq!(status.outbound_announcement_rounds_superseded_total, 1);
    assert_eq!(status.outbound_announcements_attempted_total, 1);
    assert_eq!(status.outbound_announcements_accepted_total, 1);
    assert_eq!(status.last_outbound_announced_height, Some(2));
    assert_eq!(
        status.last_outbound_announcement_result.as_deref(),
        Some("all_woken")
    );
    assert_eq!(*received_heights.lock().await, vec![1, 2]);

    http_server.abort();
    let _ = http_server.await;
}

#[test]
fn commitment_witness_startup_gate_enforces_threshold_and_conflicts() {
    assert_eq!(
        CommitmentWitnessStartupBlockReason::Equivocation.as_str(),
        "signed_checkpoint_equivocation"
    );
    let unavailable = CommitmentReconciliationOutcome::default();
    assert_eq!(
        commitment_witness_startup_decision(&unavailable, false, 1),
        Ok(CommitmentWitnessStartupDecision::DegradedUnverified)
    );
    assert_eq!(
        commitment_witness_startup_decision(&unavailable, true, 1),
        Err(CommitmentWitnessStartupBlockReason::Unavailable)
    );

    let converged = CommitmentReconciliationOutcome {
        verified: 1,
        converged: 1,
        ..CommitmentReconciliationOutcome::default()
    };
    assert_eq!(
        commitment_witness_startup_decision(&converged, true, 1),
        Ok(CommitmentWitnessStartupDecision::Verified)
    );
    assert_eq!(
        commitment_witness_startup_decision(&converged, false, 2),
        Ok(CommitmentWitnessStartupDecision::DegradedBelowThreshold)
    );
    assert_eq!(
        commitment_witness_startup_decision(&converged, true, 2),
        Err(CommitmentWitnessStartupBlockReason::ThresholdUnmet)
    );

    let two_converged = CommitmentReconciliationOutcome {
        verified: 2,
        converged: 2,
        ..CommitmentReconciliationOutcome::default()
    };
    assert_eq!(
        commitment_witness_startup_decision(&two_converged, true, 2),
        Ok(CommitmentWitnessStartupDecision::Verified)
    );

    let remote_ahead = CommitmentReconciliationOutcome {
        verified: 1,
        remote_ahead: 1,
        ..CommitmentReconciliationOutcome::default()
    };
    assert_eq!(
        commitment_witness_startup_decision(&remote_ahead, false, 2),
        Err(CommitmentWitnessStartupBlockReason::RemoteAhead)
    );

    let diverged = CommitmentReconciliationOutcome {
        verified: 1,
        diverged: 1,
        ..CommitmentReconciliationOutcome::default()
    };
    assert_eq!(
        commitment_witness_startup_decision(&diverged, false, 2),
        Err(CommitmentWitnessStartupBlockReason::Divergence)
    );
}

#[test]
fn custody_witness_guards_require_clean_current_anchor_quorum() {
    // [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Startup and
    // runtime share these fixed privacy-safe outcomes. Authentic adverse
    // evidence always wins over count, while absence and partial coverage
    // remain distinct operator diagnostics.
    assert_eq!(
        CustodyWitnessReadinessBlockReason::ReceiptVaultInvalid.as_str(),
        "receipt_vault_invalid"
    );
    assert_eq!(
        custody_witness_readiness_decision(CustodyAuditWitnessPolicyReadiness::EvidenceUnavailable),
        Err(CustodyWitnessReadinessBlockReason::EvidenceUnavailable)
    );
    assert_eq!(
        custody_witness_readiness_decision(CustodyAuditWitnessPolicyReadiness::ThresholdUnmet),
        Err(CustodyWitnessReadinessBlockReason::ThresholdUnmet)
    );
    assert_eq!(
        custody_witness_readiness_decision(CustodyAuditWitnessPolicyReadiness::AdverseEvidence),
        Err(CustodyWitnessReadinessBlockReason::AdverseEvidence)
    );
    assert_eq!(
        custody_witness_readiness_decision(CustodyAuditWitnessPolicyReadiness::Ready),
        Ok(())
    );
}

#[test]
fn custody_witness_runtime_cadence_and_renewal_lifecycle_are_bounded() {
    // [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Small
    // freshness windows cannot create a hot loop and large windows cannot
    // defer detection beyond five minutes.
    assert_eq!(custody_witness_runtime_audit_interval_secs(60), 30);
    assert_eq!(custody_witness_runtime_audit_interval_secs(120), 30);
    assert_eq!(custody_witness_runtime_audit_interval_secs(600), 150);
    assert_eq!(custody_witness_runtime_audit_interval_secs(1_200), 300);
    assert_eq!(custody_witness_runtime_audit_interval_secs(7_200), 300);
    assert_eq!(custody_witness_renewal_warning_window_secs(60), 60);
    assert_eq!(custody_witness_renewal_warning_window_secs(400), 100);
    assert_eq!(custody_witness_renewal_warning_window_secs(7_200), 900);

    let snapshot = CustodyAuditWitnessReceiptReadinessSnapshot {
        vault: CustodyAuditWitnessReceiptVaultAudit::default(),
        policy: CustodyAuditWitnessReceiptPolicyEvidence {
            configured: 1,
            fresh_verified: 1,
            accepted: 1,
            minimum_verified: 1,
            quorum_satisfied: true,
            quorum_valid_through: Some(1_000),
            ..CustodyAuditWitnessReceiptPolicyEvidence::default()
        },
        readiness: CustodyAuditWitnessPolicyReadiness::Ready,
    };
    let warning = custody_witness_renewal_status(
        &CustodyWitnessAuditEvidence {
            checkpoint_generation: 1,
            evaluated_at: 900,
            snapshot,
        },
        400,
    )
    .expect("ready policy must expose a renewal horizon");
    assert_eq!(warning.valid_for_secs, 100);
    assert_eq!(warning.warning_window_secs, 100);
    assert!(warning.renewal_recommended);

    // [CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex] One expiry horizon
    // opens one warning. Explicitly refreshed evidence either opens a new
    // horizon warning or closes the incident when it is healthy again.
    let healthy = CustodyWitnessRenewalStatus {
        valid_through: 1_200,
        valid_for_secs: 300,
        warning_window_secs: 100,
        renewal_recommended: false,
    };
    // [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Network renewal
    // requires both explicit operator enablement and an expiring quorum.
    assert!(!custody_witness_auto_renewal_due(false, warning));
    assert!(!custody_witness_auto_renewal_due(true, healthy));
    assert!(custody_witness_auto_renewal_due(true, warning));
    let refreshed_warning = CustodyWitnessRenewalStatus {
        valid_through: 1_100,
        valid_for_secs: 80,
        warning_window_secs: 100,
        renewal_recommended: true,
    };
    let recovered = CustodyWitnessRenewalStatus {
        valid_through: 2_000,
        valid_for_secs: 900,
        warning_window_secs: 100,
        renewal_recommended: false,
    };
    let mut log_state = CustodyWitnessRenewalLogState::default();
    assert_eq!(
        log_state.observe(healthy),
        CustodyWitnessRenewalLogAction::Healthy
    );
    assert_eq!(
        log_state.observe(warning),
        CustodyWitnessRenewalLogAction::WarningEntered
    );
    assert_eq!(
        log_state.observe(warning),
        CustodyWitnessRenewalLogAction::WarningSuppressed
    );
    assert_eq!(
        log_state.observe(refreshed_warning),
        CustodyWitnessRenewalLogAction::WarningEntered
    );
    assert_eq!(
        log_state.observe(recovered),
        CustodyWitnessRenewalLogAction::Recovered
    );
    assert_eq!(
        log_state.observe(recovered),
        CustodyWitnessRenewalLogAction::Healthy
    );

    let failure =
        custody_witness_runtime_failure(CustodyWitnessReadinessBlockReason::ReceiptPolicyInvalid);
    assert_eq!(failure.task, "custody-witness-runtime");
    assert_eq!(failure.reason, "receipt_policy_invalid");
}

#[test]
fn custody_witness_renewal_retry_backoff_is_bounded_by_expiry() {
    let renewal = CustodyWitnessRenewalStatus {
        valid_through: 10_000,
        valid_for_secs: 900,
        warning_window_secs: 900,
        renewal_recommended: true,
    };
    let now = Instant::now();
    let node_id = [0x42; 32];
    let mut state = CustodyWitnessRenewalRetryState::default();

    // [CUSTODY-RENEWAL-BACKOFF 2026-08-21 by Codex] The first attempt is
    // immediate. A failed round then cools only network collection while
    // the independent local audit continues on every timer tick.
    assert_eq!(
        state.action(renewal, now),
        CustodyWitnessRenewalRetryAction::Attempt
    );
    let schedule = state.record_failure(renewal, 300, &node_id, now);
    assert_eq!(schedule.consecutive_failures, 1);
    assert!(schedule.retry_before_expiry);
    assert!((300..=600).contains(&schedule.delay_secs));
    assert_eq!(schedule.delay_secs % 300, 0);
    assert_eq!(
        state.action(renewal, now + Duration::from_secs(schedule.delay_secs - 1)),
        CustodyWitnessRenewalRetryAction::BackingOff {
            retry_in_secs: 1,
            consecutive_failures: 1,
        }
    );
    assert_eq!(
        state.action(renewal, now + Duration::from_secs(schedule.delay_secs)),
        CustodyWitnessRenewalRetryAction::Attempt
    );

    // A refreshed horizon never inherits an older incident's cooldown.
    let refreshed = CustodyWitnessRenewalStatus {
        valid_through: 20_000,
        ..renewal
    };
    state.record_failure(renewal, 300, &node_id, now);
    assert_eq!(
        state.action(refreshed, now),
        CustodyWitnessRenewalRetryAction::Attempt
    );

    // When there is no future audit tick before expiry, the state reports
    // that honestly. The next strict audit fails closed instead of making
    // a network request after the evidence lifetime.
    let final_window = CustodyWitnessRenewalStatus {
        valid_for_secs: 300,
        ..renewal
    };
    let schedule = state.record_failure(final_window, 300, &node_id, now);
    assert_eq!(schedule.delay_secs, 300);
    assert!(!schedule.retry_before_expiry);
    assert_eq!(
        state.action(final_window, now + Duration::from_secs(300)),
        CustodyWitnessRenewalRetryAction::Exhausted {
            consecutive_failures: 1,
        }
    );

    state.record_success(refreshed);
    assert_eq!(state.consecutive_failures, 0);
    assert_eq!(state.retry_not_before, None);
    assert!(!state.exhausted_for_horizon);
}

#[test]
fn custody_witness_runtime_telemetry_is_aggregate_and_process_local() {
    // [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex] The management
    // contract exposes one node-wide process snapshot, never a witness,
    // route, user, receipt, signature, anchor or encrypted object.
    let telemetry = CustodyWitnessRuntimeTelemetry::new(true, true, 7_200);
    let initial = telemetry.snapshot();
    assert_eq!(initial.status, "monitoring");
    assert_eq!(initial.audit_interval_seconds, 300);
    assert_eq!(initial.freshness_window_seconds, 7_200);
    assert_eq!(initial.audits_total, 0);

    let renewal_due = CustodyWitnessRenewalStatus {
        valid_through: 10_000,
        valid_for_secs: 900,
        warning_window_secs: 900,
        renewal_recommended: true,
    };
    telemetry.record_audit(7, 9_100, renewal_due);
    telemetry.record_attempt(9_101);
    telemetry.record_failure(
        9_102,
        "collection_failed",
        CustodyWitnessRenewalRetrySchedule {
            delay_secs: 300,
            consecutive_failures: 1,
            retry_before_expiry: true,
        },
    );
    telemetry.record_backoff_skip(120, 1);
    let backing_off = telemetry.snapshot();
    assert_eq!(backing_off.status, "backing_off");
    assert_eq!(backing_off.audits_total, 1);
    assert_eq!(backing_off.renewal_attempts_total, 1);
    assert_eq!(backing_off.renewal_failures_total, 1);
    assert_eq!(backing_off.backoff_skips_total, 1);
    assert_eq!(backing_off.retry_after_seconds, Some(120));
    assert_eq!(backing_off.last_checkpoint_generation, Some(7));
    assert_eq!(backing_off.last_failure_reason, Some("collection_failed"));

    telemetry.record_failure(
        9_103,
        "quorum_not_refreshed",
        CustodyWitnessRenewalRetrySchedule {
            delay_secs: 300,
            consecutive_failures: 2,
            retry_before_expiry: false,
        },
    );
    telemetry.record_exhausted_skip(2);
    let exhausted = telemetry.snapshot();
    assert_eq!(exhausted.status, "exhausted");
    assert_eq!(exhausted.exhausted_skips_total, 1);
    assert_eq!(exhausted.retry_before_expiry, Some(false));
    assert_eq!(exhausted.retry_after_seconds, None);

    let healthy = CustodyWitnessRenewalStatus {
        valid_through: 20_000,
        valid_for_secs: 10_000,
        warning_window_secs: 900,
        renewal_recommended: false,
    };
    telemetry.record_audit(8, 10_000, healthy);
    telemetry.record_success(10_001, healthy, true);
    let recovered = telemetry.snapshot();
    assert_eq!(recovered.status, "healthy");
    assert_eq!(recovered.renewal_successes_total, 1);
    assert_eq!(recovered.partial_collection_successes_total, 1);
    assert_eq!(recovered.consecutive_failures, 0);
    assert_eq!(recovered.retry_after_seconds, None);
    assert_eq!(recovered.last_checkpoint_generation, Some(8));
    assert_eq!(recovered.quorum_valid_through, Some(20_000));

    telemetry.record_fail_closed(
        10_002,
        CustodyWitnessReadinessBlockReason::ReceiptPolicyInvalid,
    );
    let failed_closed = telemetry.snapshot();
    assert_eq!(failed_closed.status, "failed_closed");
    assert_eq!(failed_closed.fail_closed_total, 1);
    assert_eq!(
        failed_closed.last_fail_closed_reason,
        Some("receipt_policy_invalid")
    );

    let encoded = serde_json::to_string(&failed_closed).unwrap();
    for forbidden in [
        "node_id",
        "endpoint",
        "signature",
        "anchor",
        "message_id",
        "wallet",
        "payload",
        "ciphertext",
    ] {
        assert!(
            !encoded.contains(forbidden),
            "telemetry leaked forbidden field: {forbidden}"
        );
    }

    let disabled = CustodyWitnessRuntimeTelemetry::new(false, false, 7_200).snapshot();
    assert_eq!(disabled.status, "disabled");
    assert!(!disabled.runtime_required);
    assert!(!disabled.auto_renewal_enabled);
}

#[test]
fn coordinator_lease_requires_every_witness_and_a_safe_window() {
    let complete = CommitmentCoordinatorLeaseRound {
        attempted: 3,
        granted: 3,
        minimum_valid_for_secs: 120,
        ..Default::default()
    };
    assert_eq!(
        commitment_coordinator_lease_production_valid_for(&complete, 3),
        Some(105)
    );

    let partial = CommitmentCoordinatorLeaseRound {
        attempted: 3,
        granted: 2,
        failed: 1,
        minimum_valid_for_secs: 120,
        ..Default::default()
    };
    assert_eq!(
        commitment_coordinator_lease_production_valid_for(&partial, 3),
        None
    );

    let too_close_to_expiry = CommitmentCoordinatorLeaseRound {
        attempted: 3,
        granted: 3,
        minimum_valid_for_secs: COORDINATOR_LEASE_PRODUCTION_SAFETY_SECS,
        ..Default::default()
    };
    assert_eq!(
        commitment_coordinator_lease_production_valid_for(&too_close_to_expiry, 3),
        None
    );
    assert_eq!(
        commitment_coordinator_lease_production_valid_for(&complete, 0),
        None
    );
}

#[tokio::test]
async fn signed_divergent_witness_blocks_coordinator_startup_policy() {
    let now = unix_now_secs();
    let coordinator = IdentityKeyPair::generate();
    let witness = Arc::new(IdentityKeyPair::generate());
    let coordinator_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

    let coordinator_block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(2),
        GENESIS_PREV_HASH,
        vec![[0xA1; 32]],
        &coordinator,
    );
    let divergent_block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(1),
        GENESIS_PREV_HASH,
        vec![[0xB1; 32]],
        &coordinator,
    );
    assert_ne!(coordinator_block.hash(), divergent_block.hash());
    coordinator_storage
        .append_record_commitment_block(&coordinator_block, None)
        .await
        .unwrap();
    witness_storage
        .append_record_commitment_block(&divergent_block, None)
        .await
        .unwrap();
    coordinator_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();
    coordinator_storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    witness_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();

    let admit_encrypted_storage_peer =
        |peer_store: &PeerStore, identity: &IdentityKeyPair, endpoint: Option<String>| {
            let mut descriptor = NodeDescriptor::new(
                identity.public_key_bytes(),
                1,
                now.saturating_sub(1),
                now.saturating_add(600),
                "signed-witness-startup-test",
            );
            descriptor.public_endpoint = endpoint;
            descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
            let descriptor = SignedNodeDescriptor::sign(descriptor, identity).unwrap();
            let import = peer_store.apply_discovery_message(
                &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
                now,
            );
            assert_eq!(import.inserted, 1);
        };

    let witness_peers = Arc::new(PeerStore::new());
    admit_encrypted_storage_peer(&witness_peers, &coordinator, None);
    let witness_router = build_memchain_peer_router(
        Arc::clone(&witness_storage),
        witness_peers,
        Arc::clone(&witness),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let witness_address = listener.local_addr().unwrap();
    let witness_server = tokio::spawn(async move {
        axum::serve(listener, witness_router).await.unwrap();
    });

    let coordinator_peers = PeerStore::new();
    admit_encrypted_storage_peer(
        &coordinator_peers,
        &witness,
        Some(format!("http://{witness_address}")),
    );
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let round = crate::api::memchain_peer::reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        &coordinator_storage,
        &coordinator_peers,
        &coordinator,
        &client,
        &[witness.public_key_bytes()],
        2,
        |_| true,
    )
    .await;

    assert_eq!(round.eligible_witnesses, 1);
    assert_eq!(round.attempted, 1);
    assert_eq!(round.verified, 1);
    assert_eq!(round.diverged, 1);
    assert_eq!(round.failed, 0);
    assert_eq!(
        commitment_witness_startup_decision(&round, true, 1),
        Err(CommitmentWitnessStartupBlockReason::Divergence)
    );
    assert_eq!(
        coordinator_storage.record_commitment_chain_tip().await,
        (1, coordinator_block.hash())
    );
    let checkpoint_status = coordinator_storage.record_commitment_checkpoint_status();
    assert_eq!(checkpoint_status.state, "diverged");
    assert_eq!(checkpoint_status.divergences_total, 1);
    assert_eq!(checkpoint_status.evidence_records, 1);

    witness_server.abort();
    let _ = witness_server.await;
}

#[tokio::test]
async fn target_bound_v3_recovers_when_ack_body_is_lost_after_durable_custody() {
    // [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] Exercise the complete
    // source -> TCP -> real target router -> SQLite pending-store path.
    // The target publishes its replay-cache entry before test middleware
    // truncates ACK #1, so ACK #2 must be the exact cached custody proof.
    let directory = tempfile::tempdir().expect("ACK-loss relay directory");
    let mut relay_config = ChatRelayConfig::default();
    relay_config.enabled = true;
    relay_config.db_path = directory
        .path()
        .join("ack-loss-relay.sqlite3")
        .to_string_lossy()
        .into_owned();
    let target_relay = Arc::new(
        ChatRelayService::new(relay_config, [0x71; 32])
            .expect("initialize real target relay storage"),
    );
    let source_relay = test_chat_relay_service(
        &directory.path().join("ack-loss-source.sqlite3"),
        [0x72; 32],
    );
    let target_identity = Arc::new(IdentityKeyPair::generate());
    let target_node_id = target_identity.public_key_bytes();
    let target_sessions = Arc::new(SessionManager::new(16, Duration::from_secs(60)));
    let target_udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let injection = Arc::new(DirectRelayAckLossInjection::default());
    let target_app = build_chat_peer_router(
        Some(Arc::clone(&target_relay)),
        target_sessions,
        target_udp,
        Arc::new(PeerStore::new()),
        Arc::clone(&target_identity),
        test_peer_http_client(),
        None,
    )
    .layer(middleware::from_fn_with_state(
        Arc::clone(&injection),
        truncate_first_direct_relay_ack,
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let target_node = tokio::spawn(async move {
        axum::serve(listener, target_app).await.unwrap();
    });

    let now = unix_now_secs();
    let source_identity = IdentityKeyPair::generate();
    let envelope = signed_test_chat_envelope(now);
    let receiver = envelope.receiver;
    let source_peer_store = PeerStore::new();
    let descriptor = signed_chat_relay_peer_descriptor_for_identity(
        endpoint,
        now.saturating_sub(1),
        now + 300,
        &[
            NodeProtocolFeature::DirectPeerRelayAuthV2,
            NodeProtocolFeature::DirectPeerRelayReceiptV2,
            NodeProtocolFeature::DirectPeerRelayTargetBindingV3,
        ],
        target_identity.as_ref(),
    );
    source_peer_store
        .upsert_verified(descriptor, now)
        .expect("real target descriptor should verify");
    source_peer_store.record_route_forward_success(&target_node_id, now.saturating_sub(1));

    let source_client = test_peer_http_client();
    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(source_client.as_ref()),
        Some(source_relay.as_ref()),
        &source_peer_store,
        &source_identity,
        &envelope,
    )
    .await;

    assert_eq!(accepted, 1);
    assert_eq!(
        target_relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("durable target message should remain readable")
            .0
            .len(),
        1
    );
    let target_status = target_relay.peer_status();
    assert_eq!(target_status.inbound_accepted_total, 2);
    assert_eq!(target_status.inbound_duplicate_total, 1);
    let retry = source_relay.peer_status().direct_peer_retry;
    assert_eq!(retry.retry_triggered_total, 1);
    assert_eq!(retry.retry_recovered_total, 1);
    assert_eq!(retry.retry_exhausted_total, 0);
    assert_eq!(retry.deterministic_failure_total, 0);
    assert_eq!(retry.last_outcome.as_deref(), Some("recovered"));

    let ack_bodies = injection
        .successful_ack_bodies
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(ack_bodies.len(), 2);
    assert_eq!(ack_bodies[0], ack_bodies[1]);
    let ack: PeerChatRelayResponseV2 =
        serde_json::from_slice(&ack_bodies[1]).expect("cached ACK should retain its schema");
    let exact_request = PeerChatRelayRequestV3::sign(envelope, target_node_id, &source_identity)
        .expect("source request should be reproducible");
    ack.receipt
        .as_ref()
        .expect("recovered ACK should contain custody evidence")
        .verify_expected_commitment(
            &exact_request
                .request_commitment()
                .expect("reproduced request commitment"),
            &target_node_id,
            unix_now_secs(),
        )
        .expect("recovered receipt must bind the exact target request");
    drop(ack_bodies);

    let route_status = source_peer_store.route_candidate_status(unix_now_secs());
    let target_prefix = hex::encode(&target_node_id[..4]);
    let target_route = route_status
        .chat_relay
        .iter()
        .find(|candidate| candidate.node_id_prefix == target_prefix)
        .expect("target route status should remain observable");
    assert_eq!(target_route.route_health, "healthy");
    assert_eq!(target_route.route_consecutive_failures, 0);
    target_node.abort();
}

#[tokio::test]
async fn target_bound_v3_circuit_and_custody_survive_abrupt_process_restart() {
    // [DIRECT-RELAY-CRASH-DRILL 2026-08-15 by Codex] This is deliberately
    // stronger than dropping and recreating ChatRelayService in one test
    // process. The seed worker exits without running Rust destructors, then
    // an independent worker must recover both ciphertext custody and the
    // source-blind outage gate from the same SQLite database.
    let directory = tempfile::tempdir().expect("restart drill directory");
    let db_path = directory.path().join("direct-relay-restart.sqlite3");

    let v3_calls = Arc::new(AtomicUsize::new(0));
    let v3_calls_for_handler = Arc::clone(&v3_calls);
    let v3_app = Router::new().route(
        "/api/chat/peer/relay-v3",
        post(move || {
            let v3_calls_for_handler = Arc::clone(&v3_calls_for_handler);
            async move {
                v3_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }),
    );
    let v3_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind restart drill v3 target");
    let v3_endpoint = format!(
        "http://{}",
        v3_listener
            .local_addr()
            .expect("read restart drill v3 address")
    );
    let v3_node = tokio::spawn(async move {
        axum::serve(v3_listener, v3_app)
            .await
            .expect("serve restart drill v3 target");
    });

    let v2_calls = Arc::new(AtomicUsize::new(0));
    let v2_calls_for_handler = Arc::clone(&v2_calls);
    let v2_app = Router::new().route(
        "/api/chat/peer/relay-v2",
        post(move || {
            let v2_calls_for_handler = Arc::clone(&v2_calls_for_handler);
            async move {
                v2_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                StatusCode::OK
            }
        }),
    );
    let v2_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind restart drill v2 target");
    let v2_endpoint = format!(
        "http://{}",
        v2_listener
            .local_addr()
            .expect("read restart drill v2 address")
    );
    let v2_node = tokio::spawn(async move {
        axum::serve(v2_listener, v2_app)
            .await
            .expect("serve restart drill v2 target");
    });

    let crashed = run_direct_relay_restart_drill_child("seed_crash", &db_path, None, None).await;
    assert_restart_drill_child_crashed(&crashed);

    let restarted = run_direct_relay_restart_drill_child(
        "verify_restart",
        &db_path,
        Some(&v3_endpoint),
        Some(&v2_endpoint),
    )
    .await;
    assert_restart_drill_child_succeeded("fresh restart", &restarted);
    assert_eq!(
        v3_calls.load(AtomicOrdering::SeqCst),
        0,
        "open restart checkpoint must deny v3 network I/O"
    );
    assert_eq!(
        v2_calls.load(AtomicOrdering::SeqCst),
        0,
        "open restart checkpoint must not downgrade to v2"
    );

    v3_node.abort();
    v2_node.abort();
    let _ = v3_node.await;
    let _ = v2_node.await;
}

#[tokio::test]
async fn peer_store_cache_reports_unprotected_witness_as_deferred() {
    // [PEER-CACHE-RETRY-STATE 2026-08-12 by Codex] A successful local
    // write is still pending when its configured external witness cannot
    // protect the new generation. It must not be reported as stable.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let witness = IdentityKeyPair::generate();
    let mut discovery = server.config.discovery.clone();
    discovery.verified_delivery_witness_node_ids = vec![hex::encode(witness.public_key_bytes())];
    let peer_store = PeerStore::new();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-deferred-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();

    // [PEER-CACHE-TEST-HTTP-CLIENT 2026-08-23 by Codex] The test expects
    // an unavailable witness result, not host proxy initialization failure.
    let http_client = test_peer_http_client();
    let outcome = Server::persist_peer_store_cache_with_delivery_witnesses(
        &server.identity,
        &peer_store,
        &discovery,
        http_client.as_ref(),
        &path_str,
        1_800_040_100,
        false,
    )
    .await
    .unwrap();

    assert_eq!(outcome, PeerStoreCachePersistOutcome::Deferred);
    assert!(peer_store.take_client_delivery_cache_dirty());
    tokio::fs::remove_file(&path).await.unwrap();
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str))
        .await
        .unwrap();
}

#[tokio::test]
async fn peer_store_startup_witness_gate_revokes_complete_readiness_bundle() {
    // [EXTERNAL-WITNESS-STARTUP-REGRESSION 2026-08-21 by Codex] Exercise
    // the real startup reconciliation entry point rather than only the
    // PeerStore reset primitive. A required witness with no local anchor
    // must revoke all recovered readiness before listeners can start.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let witness = IdentityKeyPair::generate();
    let mut discovery = server.config.discovery.clone();
    discovery.verified_delivery_witness_node_ids = vec![hex::encode(witness.public_key_bytes())];
    discovery.verified_delivery_witness_required_for_restore = true;

    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let descriptor = signed_chat_relay_peer_descriptor(
        "https://startup-gate.example".to_string(),
        7,
        now + 4_000,
    );
    let node_id = descriptor.node_id();
    assert!(peer_store.upsert_verified(descriptor, now).unwrap());
    peer_store.record_route_forward_success(&node_id, now + 1);
    for offset in 2..=4 {
        peer_store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            2,
            2,
            2,
            1,
        );
        peer_store.record_blind_relay_three_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            3,
            3,
            3,
            2,
        );
    }
    peer_store.record_verified_client_onion_delivery(now + 4);
    peer_store.record_blind_relay_terminal(now + 4, 0, 32);
    assert!(peer_store.is_routeable_now(&node_id, now + 5));

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let missing_path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-startup-witness-gate-{unique}.json"
    ));
    // [PEER-CACHE-TEST-HTTP-CLIENT 2026-08-23 by Codex] Reuse the
    // proxy-free test client so startup-gate assertions stay deterministic.
    let http_client = test_peer_http_client();
    let decision = Server::reconcile_peer_cache_delivery_witnesses(
        &server.identity,
        &peer_store,
        &discovery,
        http_client.as_ref(),
        &missing_path.to_string_lossy(),
        true,
    )
    .await;

    assert_eq!(
        decision,
        PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Missing
    );
    let status = peer_store.status(now + 6);
    assert_eq!(status.snapshot.valid_peers, 1);
    assert!(peer_store.get_valid(&node_id, now + 6).is_some());
    assert!(!peer_store.is_routeable_now(&node_id, now + 6));
    assert_eq!(status.two_hop_path_proof_history.attempted, 0);
    assert_eq!(status.three_hop_path_proof_history.attempted, 0);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert_eq!(status.runtime.blind_relay.terminal, 1);
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "peer_cache_external_witness_gate"
            && event.outcome == "rejected"
            && event.detail.contains("reason=external_witness_unavailable")
            && !event.detail.contains("startup-gate.example")
    }));
}

#[tokio::test]
async fn peer_store_startup_witness_gate_rejects_cache_ahead_of_anchor() {
    // [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] Model a
    // crash between the signed cache rename and matching anchor rename.
    // The older valid anchor must not authorize newer restored readiness,
    // and no network request should be attempted for the wrong generation.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let descriptor = signed_chat_relay_peer_descriptor(
        "https://cache-ahead.example".to_string(),
        8,
        now + 4_000,
    );
    let node_id = descriptor.node_id();
    assert!(peer_store.upsert_verified(descriptor, now).unwrap());
    peer_store.record_route_forward_success(&node_id, now + 1);
    for offset in 2..=4 {
        peer_store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            2,
            2,
            2,
            1,
        );
        peer_store.record_blind_relay_three_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            3,
            3,
            3,
            2,
        );
    }
    peer_store.record_verified_client_onion_delivery(now + 4);
    peer_store.record_blind_relay_terminal(now + 4, 0, 32);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-generation-binding-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::persist_peer_store_cache_once(&server.identity, &peer_store, &path_str, now + 5)
        .await
        .unwrap();
    assert_eq!(peer_store.peer_cache_recovery_generation(), 1);

    // The new cache generation became durable, but the old generation-one
    // anchor is still on disk because the process crashed before replace.
    peer_store.record_client_delivery_cache_persisted(now + 6, 1, 2);
    assert_eq!(peer_store.peer_cache_recovery_generation(), 2);

    let witness = IdentityKeyPair::generate();
    let mut discovery = server.config.discovery.clone();
    discovery.verified_delivery_witness_node_ids = vec![hex::encode(witness.public_key_bytes())];
    discovery.verified_delivery_witness_required_for_restore = true;
    // [PEER-CACHE-TEST-HTTP-CLIENT 2026-08-23 by Codex] Avoid leaking
    // platform proxy configuration into cache-ahead recovery tests.
    let http_client = test_peer_http_client();
    let decision = Server::reconcile_peer_cache_delivery_witnesses(
        &server.identity,
        &peer_store,
        &discovery,
        http_client.as_ref(),
        &path_str,
        true,
    )
    .await;

    assert_eq!(
        decision,
        PeerStoreVerifiedClientDeliveryExternalWitnessDecision::Unprotected("unavailable")
    );
    let status = peer_store.status(now + 7);
    assert_eq!(status.snapshot.valid_peers, 1);
    assert!(peer_store.get_valid(&node_id, now + 7).is_some());
    assert!(!peer_store.is_routeable_now(&node_id, now + 7));
    assert_eq!(status.two_hop_path_proof_history.attempted, 0);
    assert_eq!(status.three_hop_path_proof_history.attempted, 0);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert_eq!(status.runtime.blind_relay.terminal, 1);
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_witness_status
            .as_deref(),
        Some("unavailable")
    );
    assert_eq!(status.bootstrap.last_client_delivery_witness_generation, 2);
    assert_eq!(status.bootstrap.last_client_delivery_witness_attempted, 0);
    assert_eq!(status.bootstrap.last_client_delivery_witness_failed, 1);

    let _ = tokio::fs::remove_file(&path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str)).await;
}
