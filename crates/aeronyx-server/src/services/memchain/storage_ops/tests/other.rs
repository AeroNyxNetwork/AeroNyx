// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn test_has_active_content() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xBB; 32];
    let r = make_rec_owner(100, owner, MemoryLayer::Episode);
    s.insert(&r, "m").await;
    assert!(s.has_active_content(&owner, &r.encrypted_content).await);
    assert!(!s.has_active_content(&owner, b"nonexistent").await);
    assert!(
        !s.has_active_content(&[0xCC; 32], &r.encrypted_content)
            .await
    );
}

#[tokio::test]
async fn test_insert_raw_log_plaintext() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let log_id = s
        .insert_raw_log("s1", 0, "user", "hello", "test", None, 1, None, None)
        .await
        .unwrap();
    assert!(log_id > 0);
    let content = s.read_rawlog_content(log_id, None).await.unwrap();
    assert_eq!(content, "hello");
}

#[tokio::test]
async fn test_insert_raw_log_encrypted() {
    use super::super::super::storage_crypto::derive_rawlog_key;
    let rlk = derive_rawlog_key(&[0x42; 32]);
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let log_id = s
        .insert_raw_log("s1", 0, "user", "secret", "test", None, 1, None, Some(&rlk))
        .await
        .unwrap();
    let content = s.read_rawlog_content(log_id, Some(&rlk)).await.unwrap();
    assert_eq!(content, "secret");
    assert!(s.read_rawlog_content(log_id, None).await.is_none());
}

#[tokio::test]
async fn test_feedback_operations() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    let owner = [0xAA; 32];
    let r = make_rec_owner(100, owner, MemoryLayer::Episode);
    let rid = r.record_id;
    s.insert(&r, "m").await;
    s.increment_positive_feedback(&rid).await;
    s.increment_negative_feedback(&rid).await;
    s.increment_negative_feedback(&rid).await;
    let got = s.get(&rid).await.unwrap();
    assert_eq!(got.positive_feedback, 1);
    assert_eq!(got.negative_feedback, 2);
}

#[tokio::test]
async fn test_coordinator_production_fence_rejects_duplicate_and_recovers_after_release() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("coordinator-fence.db");
    let first = MemoryStorage::open(&db_path, None).unwrap();
    let second = MemoryStorage::open(&db_path, None).unwrap();

    first
        .configure_record_commitment_durability(true)
        .await
        .unwrap();
    let first_status = first.record_commitment_chain_integrity_status();
    assert_eq!(first_status.coordinator_fence_state, "held");
    assert!(first_status.coordinator_fence_acquired_at.is_some());
    assert_eq!(first_status.coordinator_fence_acquisition_failures_total, 0);

    let error = second
        .configure_record_commitment_durability(true)
        .await
        .unwrap_err();
    assert_eq!(
        error,
        "commitment coordinator production fence is held by another local process"
    );
    let contended = second.record_commitment_chain_integrity_status();
    assert_eq!(contended.coordinator_fence_state, "contended");
    assert_eq!(contended.coordinator_fence_acquisition_failures_total, 1);
    assert!(contended.coordinator_fence_acquired_at.is_none());

    let fence_path = commitment_coordinator_fence_path(&db_path).unwrap();
    let mode = std::fs::metadata(&fence_path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);

    drop(first);
    second
        .configure_record_commitment_durability(true)
        .await
        .unwrap();
    let recovered = second.record_commitment_chain_integrity_status();
    assert_eq!(recovered.coordinator_fence_state, "held");
    assert!(recovered.coordinator_fence_acquired_at.is_some());
    assert_eq!(recovered.coordinator_fence_acquisition_failures_total, 1);
}

#[tokio::test]
async fn test_non_coordinator_does_not_claim_production_fence() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("follower-compatible.db");
    let follower = MemoryStorage::open(&db_path, None).unwrap();
    follower
        .configure_record_commitment_durability(false)
        .await
        .unwrap();
    assert_eq!(
        follower
            .record_commitment_chain_integrity_status()
            .coordinator_fence_state,
        "not_required"
    );

    let coordinator = MemoryStorage::open(&db_path, None).unwrap();
    coordinator
        .configure_record_commitment_durability(true)
        .await
        .unwrap();
    assert_eq!(
        coordinator
            .record_commitment_chain_integrity_status()
            .coordinator_fence_state,
        "held"
    );
}

#[tokio::test]
async fn test_coordinator_production_fence_rejects_symlink() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("coordinator-symlink.db");
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    let fence_path = commitment_coordinator_fence_path(&db_path).unwrap();
    let target = directory.path().join("unexpected-target");
    File::create(&target).unwrap();
    std::os::unix::fs::symlink(&target, &fence_path).unwrap();

    let error = storage
        .configure_record_commitment_durability(true)
        .await
        .unwrap_err();
    assert_eq!(
        error,
        "commitment coordinator production fence file is unsafe"
    );
    let status = storage.record_commitment_chain_integrity_status();
    assert_eq!(status.coordinator_fence_state, "failed");
    assert_eq!(status.coordinator_fence_acquisition_failures_total, 1);
}

#[tokio::test]
async fn test_coordinator_handover_history_is_contiguous_and_idempotent() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let root = IdentityKeyPair::generate();
    let second = IdentityKeyPair::generate();
    let third = IdentityKeyPair::generate();

    let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x41, &root);
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let first_handover = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x51; 16],
        2_000,
        &root,
        &second,
    );
    assert_eq!(
        storage
            .persist_record_coordinator_handover(&root.public_key_bytes(), &first_handover, 2_001,)
            .await
            .unwrap(),
        RecordCoordinatorHandoverPersistOutcome::Inserted
    );
    assert_eq!(
        storage
            .persist_record_coordinator_handover(&root.public_key_bytes(), &first_handover, 2_002,)
            .await
            .unwrap(),
        RecordCoordinatorHandoverPersistOutcome::AlreadyPresent
    );

    let second_block = signed_commitment_block(2, first_block.hash(), 0x42, &second);
    storage
        .append_record_commitment_block(&second_block, None)
        .await
        .unwrap();
    let second_handover = RecordCoordinatorHandoverV1::new_dual_signed(
        2,
        3,
        second_block.hash(),
        [0x52; 16],
        2_003,
        &second,
        &third,
    );
    assert_eq!(
        storage
            .persist_record_coordinator_handover(&root.public_key_bytes(), &second_handover, 2_004,)
            .await
            .unwrap(),
        RecordCoordinatorHandoverPersistOutcome::Inserted
    );

    let history = storage
        .record_coordinator_handover_history(&root.public_key_bytes())
        .await
        .unwrap();
    assert_eq!(history, vec![first_handover, second_handover]);
}

#[tokio::test]
async fn test_configured_authority_rejects_old_coordinator_after_handover() {
    // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Once epoch one
    // activates at height two, the old coordinator remains a valid
    // cryptographic signer but is no longer an authorised proposer.
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let root = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    storage
        .configure_record_commitment_authority_root(Some(root.public_key_bytes()))
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();

    let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x61, &root);
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let handover = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x62; 16],
        7_000,
        &root,
        &next,
    );
    storage
        .persist_record_coordinator_handover(&root.public_key_bytes(), &handover, 7_001)
        .await
        .unwrap();

    let stale_authority = signed_commitment_block(2, first_block.hash(), 0x63, &root);
    assert_eq!(
        storage
            .append_record_commitment_block(&stale_authority, None)
            .await
            .unwrap_err(),
        "commitment block has an unauthorized proposer at height 2"
    );
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .verified_tip_height,
        1
    );

    let authorised = signed_commitment_block(2, first_block.hash(), 0x64, &next);
    assert_eq!(
        storage
            .append_record_commitment_block(&authorised, None)
            .await
            .unwrap(),
        RecordCommitmentAppendOutcome::Inserted
    );
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .verified_tip_height,
        2
    );
}

#[tokio::test]
async fn test_authority_snapshot_pages_exact_next_handover() {
    // [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] Runtime consumers
    // receive only the exact next epoch while height resolution follows
    // the same immutable schedule enforced by atomic block append.
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let root = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    storage
        .configure_record_commitment_authority_root(Some(root.public_key_bytes()))
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();

    assert_eq!(
        storage.record_commitment_authority_state().await.unwrap(),
        Some(RecordCommitmentAuthorityState {
            authority_epoch: 0,
            coordinator: root.public_key_bytes(),
            next_block_height: 1,
        })
    );
    assert_eq!(
        storage
            .next_record_coordinator_handover_page(0)
            .await
            .unwrap(),
        RecordCoordinatorHandoverPage {
            handover: None,
            latest_authority_epoch: 0,
        }
    );

    let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x69, &root);
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let handover = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x6A; 16],
        9_000,
        &root,
        &next,
    );
    storage
        .persist_record_coordinator_handover(&root.public_key_bytes(), &handover, 9_001)
        .await
        .unwrap();

    assert_eq!(
        storage.record_commitment_authority_state().await.unwrap(),
        Some(RecordCommitmentAuthorityState {
            authority_epoch: 1,
            coordinator: next.public_key_bytes(),
            next_block_height: 2,
        })
    );
    assert_eq!(
        storage
            .record_commitment_authority_for_height(1)
            .await
            .unwrap(),
        Some(root.public_key_bytes())
    );
    assert_eq!(
        storage
            .record_commitment_authority_for_height(2)
            .await
            .unwrap(),
        Some(next.public_key_bytes())
    );
    assert_eq!(
        storage
            .next_record_coordinator_handover_page(0)
            .await
            .unwrap(),
        RecordCoordinatorHandoverPage {
            handover: Some(handover),
            latest_authority_epoch: 1,
        }
    );
    assert_eq!(
        storage
            .next_record_coordinator_handover_page(1)
            .await
            .unwrap(),
        RecordCoordinatorHandoverPage {
            handover: None,
            latest_authority_epoch: 1,
        }
    );
    assert_eq!(
        storage
            .next_record_coordinator_handover_page(2)
            .await
            .unwrap_err(),
        "requested authority epoch is ahead of local history"
    );
}

#[tokio::test]
async fn test_authority_root_is_immutable_and_rejects_sentinel() {
    // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Reusing the same
    // root is idempotent, while replacement or disabling must not provide
    // an alternate path around the dual-signed handover schedule.
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let root = IdentityKeyPair::generate().public_key_bytes();
    let replacement = IdentityKeyPair::generate().public_key_bytes();

    assert_eq!(
        storage
            .configure_record_commitment_authority_root(Some([0; 32]))
            .unwrap_err(),
        "commitment authority root must be non-zero"
    );
    storage
        .configure_record_commitment_authority_root(Some(root))
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    assert_eq!(
        storage.record_commitment_chain_integrity_status().state,
        "verified"
    );

    storage
        .configure_record_commitment_authority_root(Some(root))
        .unwrap();
    assert_eq!(
        storage.record_commitment_chain_integrity_status().state,
        "verified"
    );

    assert_eq!(
        storage
            .configure_record_commitment_authority_root(Some(replacement))
            .unwrap_err(),
        "commitment authority root is immutable after installation"
    );
    assert_eq!(
        storage
            .configure_record_commitment_authority_root(None)
            .unwrap_err(),
        "commitment authority root cannot be disabled after installation"
    );
    assert_eq!(
        storage.record_commitment_chain_integrity_status().state,
        "verified"
    );
    assert_eq!(
        storage
            .record_coordinator_handover_history(&replacement)
            .await
            .unwrap_err(),
        "coordinator handover root does not match configured authority"
    );
}

#[tokio::test]
async fn test_startup_authority_audit_rejects_legacy_unauthorised_proposer() {
    // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Model an older
    // binary that verified signatures but did not enforce the handover key
    // schedule. The upgraded node must fail closed during startup audit.
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("authority-upgrade.db");
    let root = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    {
        let legacy = MemoryStorage::open(&db_path, None).unwrap();
        legacy.audit_record_commitment_chain().await.unwrap();
        let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x65, &root);
        legacy
            .append_record_commitment_block(&first_block, None)
            .await
            .unwrap();
        let handover = RecordCoordinatorHandoverV1::new_dual_signed(
            1,
            2,
            first_block.hash(),
            [0x66; 16],
            8_000,
            &root,
            &next,
        );
        legacy
            .persist_record_coordinator_handover(&root.public_key_bytes(), &handover, 8_001)
            .await
            .unwrap();
        let stale_authority = signed_commitment_block(2, first_block.hash(), 0x67, &root);
        legacy
            .append_record_commitment_block(&stale_authority, Some(&[0x68; 32]))
            .await
            .unwrap();
    }

    let upgraded = MemoryStorage::open(&db_path, None).unwrap();
    upgraded
        .configure_record_commitment_authority_root(Some(root.public_key_bytes()))
        .unwrap();
    assert_eq!(
        upgraded.audit_record_commitment_chain().await.unwrap_err(),
        "commitment block has an unauthorized proposer at height 2"
    );
    assert_eq!(
        upgraded.record_commitment_chain_integrity_status().state,
        "not_verified"
    );
}

#[tokio::test]
async fn test_coordinator_handover_rejects_epoch_gap_and_active_lease() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let root = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x43, &root);
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();

    let skipped_epoch = RecordCoordinatorHandoverV1::new_dual_signed(
        2,
        2,
        first_block.hash(),
        [0x53; 16],
        3_000,
        &root,
        &next,
    );
    assert!(storage
        .persist_record_coordinator_handover(&root.public_key_bytes(), &skipped_epoch, 3_001,)
        .await
        .unwrap_err()
        .contains("authority epoch is not contiguous"));

    let instance = [0x54; 32];
    assert!(matches!(
        storage
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &root.public_key_bytes(),
                &instance,
                1,
                &first_block.hash(),
                3_002,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted { .. }
    ));
    let valid = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x55; 16],
        3_003,
        &root,
        &next,
    );
    assert_eq!(
        storage
            .persist_record_coordinator_handover(&root.public_key_bytes(), &valid, 3_004,)
            .await
            .unwrap_err(),
        "coordinator handover rejected while a witness lease remains active"
    );
    assert!(matches!(
        storage
            .release_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &root.public_key_bytes(),
                &instance,
                3_005,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::Released { .. }
    ));
    assert_eq!(
        storage
            .persist_record_coordinator_handover(&root.public_key_bytes(), &valid, 3_006,)
            .await
            .unwrap(),
        RecordCoordinatorHandoverPersistOutcome::Inserted
    );
}

#[tokio::test]
async fn test_coordinator_handover_history_rejects_denormalized_tampering() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let root = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x44, &root);
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let proof = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x56; 16],
        4_000,
        &root,
        &next,
    );
    storage
        .persist_record_coordinator_handover(&root.public_key_bytes(), &proof, 4_001)
        .await
        .unwrap();
    {
        let conn = storage.conn.lock().await;
        conn.execute(
            "UPDATE record_coordinator_handovers SET next_signature=?1
                 WHERE authority_epoch=1",
            params![[0xFF_u8; 64].as_slice()],
        )
        .unwrap();
    }
    assert!(storage
        .record_coordinator_handover_history(&root.public_key_bytes())
        .await
        .unwrap_err()
        .contains("stored coordinator handover row mismatch"));
}

#[tokio::test]
async fn test_coordinator_handover_history_rejects_acceptance_time_rollback() {
    // [HANDOVER-ACCEPTANCE-AUDIT 2026-08-14 by Codex] Model an operator or
    // disk attacker rewriting non-consensus audit metadata after a valid
    // insert. Restart/history replay must fail closed instead of trusting
    // the insert-time check forever.
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let root = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x46, &root);
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let proof = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x58; 16],
        6_000,
        &root,
        &next,
    );
    storage
        .persist_record_coordinator_handover(&root.public_key_bytes(), &proof, 6_001)
        .await
        .unwrap();
    {
        let conn = storage.conn.lock().await;
        conn.execute(
            "UPDATE record_coordinator_handovers SET accepted_at=?1
                 WHERE authority_epoch=1",
            params![5_999_i64],
        )
        .unwrap();
    }

    assert_eq!(
        storage
            .record_coordinator_handover_history(&root.public_key_bytes())
            .await
            .unwrap_err(),
        "stored coordinator handover acceptance time is invalid at epoch 1"
    );
}

#[tokio::test]
async fn test_coordinator_handover_rejects_unauthorized_historical_proposer() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let root = IdentityKeyPair::generate();
    let unauthorized = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let first_block = signed_commitment_block(1, GENESIS_PREV_HASH, 0x45, &unauthorized);
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let proof = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x57; 16],
        5_000,
        &root,
        &next,
    );

    assert!(storage
        .persist_record_coordinator_handover(&root.public_key_bytes(), &proof, 5_001)
        .await
        .unwrap_err()
        .contains("unauthorized proposer at height 1"));
}

#[tokio::test]
async fn test_coordinator_lease_runtime_gates_production_and_expires_monotonically() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.configure_record_commitment_coordinator_lease(true, 2);
    assert!(!storage.record_commitment_production_permitted());
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .coordinator_lease_state,
        "acquiring"
    );

    storage
        .apply_record_commitment_coordinator_lease(2, 1, 2_000)
        .unwrap();
    assert!(storage.record_commitment_production_permitted());
    let held = storage.record_commitment_chain_integrity_status();
    assert_eq!(held.coordinator_lease_state, "held");
    assert!(held.coordinator_lease_production_permitted);
    assert_eq!(held.coordinator_lease_seconds_remaining, Some(1));
    assert_eq!(held.coordinator_lease_last_attempted_at, Some(2_000));
    assert_eq!(held.coordinator_lease_consecutive_failures, 0);
    assert_eq!(held.coordinator_lease_recoveries_total, 0);

    storage.record_commitment_coordinator_lease_failure(1);
    assert!(storage.record_commitment_production_permitted());
    let degraded = storage.record_commitment_chain_integrity_status();
    assert_eq!(degraded.coordinator_lease_state, "renewal_degraded");
    assert!(degraded.coordinator_lease_production_permitted);
    assert_eq!(degraded.coordinator_lease_granted_witnesses, 1);
    assert!(degraded.coordinator_lease_last_failure_at.is_some());
    assert_eq!(degraded.coordinator_lease_renewal_failures_total, 1);
    assert_eq!(degraded.coordinator_lease_consecutive_failures, 1);

    tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
    assert!(!storage.record_commitment_production_permitted());
    let expired = storage.record_commitment_chain_integrity_status();
    assert_eq!(expired.coordinator_lease_state, "expired");
    assert!(!expired.coordinator_lease_production_permitted);
    assert_eq!(expired.coordinator_lease_seconds_remaining, Some(0));
    assert_eq!(expired.coordinator_lease_renewal_failures_total, 1);
    assert_eq!(expired.coordinator_lease_consecutive_failures, 1);

    storage.record_commitment_coordinator_lease_failure(1);
    let expired_retry = storage.record_commitment_chain_integrity_status();
    assert_eq!(expired_retry.coordinator_lease_state, "expired");
    assert!(!expired_retry.coordinator_lease_production_permitted);
    assert_eq!(expired_retry.coordinator_lease_seconds_remaining, Some(0));
    assert_eq!(expired_retry.coordinator_lease_renewal_failures_total, 2);
    assert_eq!(expired_retry.coordinator_lease_consecutive_failures, 2);

    storage
        .apply_record_commitment_coordinator_lease(2, 1, 2_002)
        .unwrap();
    let recovered = storage.record_commitment_chain_integrity_status();
    assert_eq!(recovered.coordinator_lease_state, "held");
    assert!(recovered.coordinator_lease_production_permitted);
    assert_eq!(recovered.coordinator_lease_last_attempted_at, Some(2_002));
    assert_eq!(recovered.coordinator_lease_last_renewed_at, Some(2_002));
    assert!(recovered.coordinator_lease_last_failure_at.is_some());
    assert_eq!(recovered.coordinator_lease_renewal_failures_total, 2);
    assert_eq!(recovered.coordinator_lease_consecutive_failures, 0);
    assert_eq!(recovered.coordinator_lease_recoveries_total, 1);

    storage.configure_record_commitment_coordinator_lease(false, 0);
    assert!(storage.record_commitment_production_permitted());
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .coordinator_lease_state,
        "disabled"
    );
}

#[test]
fn test_block_page_pull_runtime_is_aggregate_monotonic_and_follower_only() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();

    storage.record_commitment_block_page_pull_outcome(
        90,
        RecordCommitmentBlockPagePullDisposition::Coordinator,
        0,
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .block_page_pulls_total,
        0
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_block_page_security_stop_at,
        None
    );
    // [FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex] Exercise all
    // mutually exclusive terminal dispositions, timestamp clamping, and
    // actual carrier-attempt accounting without retaining source details.
    storage.configure_record_commitment_sync(false, true);
    storage.record_commitment_block_page_pull_outcome(
        100,
        RecordCommitmentBlockPagePullDisposition::Coordinator,
        0,
    );
    storage.record_commitment_block_page_pull_outcome(
        110,
        RecordCommitmentBlockPagePullDisposition::CarrierRecovered,
        2,
    );
    storage.record_commitment_block_page_pull_outcome(
        105,
        RecordCommitmentBlockPagePullDisposition::AvailabilityExhausted,
        1,
    );
    storage.record_commitment_block_page_pull_outcome(
        120,
        RecordCommitmentBlockPagePullDisposition::SecurityStopped,
        1,
    );
    storage.record_commitment_block_page_pull_outcome(
        130,
        RecordCommitmentBlockPagePullDisposition::Coordinator,
        0,
    );

    let status = storage.record_commitment_sync_status();
    assert_eq!(status.last_block_page_pull_at, Some(130));
    assert_eq!(
        status.last_block_page_pull_result.as_deref(),
        Some("coordinator")
    );
    assert_eq!(status.last_block_carrier_recovered_at, Some(110));
    // [STICKY-SECURITY-EVIDENCE 2026-07-29 by Codex] A later successful
    // retrieval updates the latest result without erasing incident time.
    assert_eq!(status.last_block_page_security_stop_at, Some(120));
    assert_eq!(status.block_page_pulls_total, 5);
    assert_eq!(status.block_page_coordinator_success_total, 2);
    assert_eq!(status.block_carrier_attempts_total, 4);
    assert_eq!(status.block_carrier_recoveries_total, 1);
    assert_eq!(status.block_page_availability_exhausted_total, 1);
    assert_eq!(status.block_page_security_stops_total, 1);
    assert_eq!(
        status.block_page_pulls_total,
        status
            .block_page_coordinator_success_total
            .saturating_add(status.block_carrier_recoveries_total)
            .saturating_add(status.block_page_availability_exhausted_total)
            .saturating_add(status.block_page_security_stops_total)
    );

    storage.configure_record_commitment_sync(true, false);
    storage.record_commitment_block_page_pull_outcome(
        130,
        RecordCommitmentBlockPagePullDisposition::Coordinator,
        0,
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .block_page_pulls_total,
        0
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_block_page_security_stop_at,
        None
    );
}

#[test]
fn test_authority_carrier_runtime_is_aggregate_isolated_and_follower_only() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();

    storage.record_commitment_authority_sync_outcome(
        90,
        RecordCommitmentAuthoritySyncDisposition::Coordinator,
        0,
    );
    storage.record_commitment_authority_carrier_circuit_observation(2, 3, 1);
    let disabled = storage.record_commitment_sync_status();
    assert_eq!(disabled.authority_sync_rounds_total, 0);
    assert_eq!(disabled.authority_carrier_cooling_slots, 0);

    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Exercise every
    // source-blind terminal result, monotonic timestamps, sticky security
    // evidence, and an independent typed circuit without source details.
    storage.configure_record_commitment_sync(false, true);
    storage.record_commitment_block_carrier_circuit_observation(1, 2, 1);
    storage.record_commitment_authority_carrier_circuit_observation(2, 3, 1);
    storage.record_commitment_authority_carrier_circuit_observation(1, 4, 2);
    storage.record_commitment_authority_sync_outcome(
        100,
        RecordCommitmentAuthoritySyncDisposition::Coordinator,
        0,
    );
    storage.record_commitment_authority_sync_outcome(
        110,
        RecordCommitmentAuthoritySyncDisposition::CarrierRecovered,
        2,
    );
    storage.record_commitment_authority_sync_outcome(
        105,
        RecordCommitmentAuthoritySyncDisposition::AvailabilityExhausted,
        1,
    );
    storage.record_commitment_authority_sync_outcome(
        120,
        RecordCommitmentAuthoritySyncDisposition::SecurityStopped,
        1,
    );
    storage.record_commitment_authority_sync_outcome(
        130,
        RecordCommitmentAuthoritySyncDisposition::Coordinator,
        0,
    );

    let follower = storage.record_commitment_sync_status();
    assert_eq!(follower.last_authority_sync_at, Some(130));
    assert_eq!(
        follower.last_authority_sync_result.as_deref(),
        Some("coordinator")
    );
    assert_eq!(follower.last_authority_carrier_recovered_at, Some(110));
    assert_eq!(follower.last_authority_security_stop_at, Some(120));
    assert_eq!(follower.authority_sync_rounds_total, 5);
    assert_eq!(follower.authority_coordinator_success_total, 2);
    assert_eq!(follower.authority_carrier_attempts_total, 4);
    assert_eq!(follower.authority_carrier_recoveries_total, 1);
    assert_eq!(follower.authority_availability_exhausted_total, 1);
    assert_eq!(follower.authority_security_stops_total, 1);
    assert_eq!(follower.authority_carrier_cooling_slots, 1);
    assert_eq!(follower.authority_carrier_cooldown_skips_total, 7);
    assert_eq!(follower.authority_carrier_half_open_attempts_total, 3);
    assert_eq!(
        follower.authority_sync_rounds_total,
        follower
            .authority_coordinator_success_total
            .saturating_add(follower.authority_carrier_recoveries_total)
            .saturating_add(follower.authority_availability_exhausted_total)
            .saturating_add(follower.authority_security_stops_total)
    );
    assert_eq!(follower.block_carrier_cooling_slots, 1);
    assert_eq!(follower.block_carrier_cooldown_skips_total, 2);
    assert_eq!(follower.block_carrier_half_open_attempts_total, 1);

    storage.configure_record_commitment_sync(true, false);
    storage.record_commitment_authority_sync_outcome(
        140,
        RecordCommitmentAuthoritySyncDisposition::CarrierRecovered,
        1,
    );
    storage.record_commitment_authority_carrier_circuit_observation(3, 8, 5);
    let coordinator = storage.record_commitment_sync_status();
    assert_eq!(coordinator.authority_sync_rounds_total, 0);
    assert_eq!(coordinator.authority_carrier_attempts_total, 0);
    assert_eq!(coordinator.authority_carrier_cooling_slots, 0);
    assert_eq!(coordinator.authority_carrier_cooldown_skips_total, 0);
    assert_eq!(coordinator.authority_carrier_half_open_attempts_total, 0);
}

#[test]
fn test_block_carrier_circuit_runtime_is_aggregate_and_follower_only() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();

    storage.record_commitment_block_carrier_circuit_observation(2, 3, 1);
    let disabled = storage.record_commitment_sync_status();
    assert_eq!(disabled.block_carrier_cooling_slots, 0);
    assert_eq!(disabled.block_carrier_cooldown_skips_total, 0);
    assert_eq!(disabled.block_carrier_half_open_attempts_total, 0);

    // [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] The gauge is
    // replaced by the latest anonymous observation while event totals are
    // additive. No per-slot history exists in the storage contract.
    storage.configure_record_commitment_sync(false, true);
    storage.record_commitment_block_carrier_circuit_observation(2, 3, 1);
    storage.record_commitment_block_carrier_circuit_observation(1, 4, 2);
    let follower = storage.record_commitment_sync_status();
    assert_eq!(follower.block_carrier_cooling_slots, 1);
    assert_eq!(follower.block_carrier_cooldown_skips_total, 7);
    assert_eq!(follower.block_carrier_half_open_attempts_total, 3);

    storage.configure_record_commitment_sync(true, false);
    storage.record_commitment_block_carrier_circuit_observation(3, 8, 5);
    let coordinator = storage.record_commitment_sync_status();
    assert_eq!(coordinator.block_carrier_cooling_slots, 0);
    assert_eq!(coordinator.block_carrier_cooldown_skips_total, 0);
    assert_eq!(coordinator.block_carrier_half_open_attempts_total, 0);
}

#[test]
fn test_follower_readiness_keeps_security_failures_above_transport_backoff() {
    // [FOLLOWER-EFFECTIVE-READINESS 2026-07-30 by Codex] A later transport
    // outage must not make an existing proof-security stop look like an
    // ordinary retryable availability fault.
    assert_eq!(
        record_commitment_follower_readiness(
            "follower",
            true,
            "backoff",
            "security_stopped",
            false,
            true,
        ),
        RecordCommitmentFollowerReadiness::SecurityStopped
    );
    assert_eq!(
        record_commitment_follower_readiness(
            "follower",
            true,
            "catching_up",
            "configuration_error",
            false,
            true,
        ),
        RecordCommitmentFollowerReadiness::ConfigurationError
    );
    assert_eq!(
        record_commitment_follower_readiness(
            "coordinator",
            false,
            "producing",
            "security_stopped",
            false,
            true,
        ),
        RecordCommitmentFollowerReadiness::NotApplicable
    );
    assert_eq!(
        record_commitment_follower_readiness(
            "follower",
            true,
            "stopped",
            "security_stopped",
            false,
            true,
        ),
        RecordCommitmentFollowerReadiness::Stopped
    );
    assert_eq!(
        record_commitment_follower_readiness("follower", true, "current", "ready", true, true,),
        RecordCommitmentFollowerReadiness::Stale
    );
}

#[test]
fn test_follower_readiness_expires_after_three_missed_poll_windows() {
    // [FOLLOWER-READINESS-FRESHNESS 2026-07-30 by Codex] A successful
    // signed checkpoint has a bounded operational lifetime. The legacy
    // block state remains `current`, while the additive readiness contract
    // fails closed exactly at the configured deadline.
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.configure_record_commitment_sync(false, true);
    storage.configure_record_commitment_sync_readiness_freshness(30);
    storage.record_commitment_sync_checkpoint_success(100, 7);

    let fresh = storage.record_commitment_sync_status_at(189);
    assert_eq!(fresh.state, "current");
    assert_eq!(fresh.follower_readiness_state, "ready");
    assert!(fresh.follower_fully_ready);
    assert_eq!(fresh.follower_convergence_confirmed_at, Some(100));
    assert_eq!(fresh.follower_readiness_stale_after, Some(190));

    let stale = storage.record_commitment_sync_status_at(190);
    assert_eq!(stale.state, "current");
    assert_eq!(stale.follower_readiness_state, "stale");
    assert!(!stale.follower_fully_ready);

    storage.record_commitment_sync_attempt(191);
    let syncing = storage.record_commitment_sync_status_at(191);
    assert_eq!(syncing.follower_readiness_state, "synchronizing");

    storage.record_commitment_sync_checkpoint_success(192, 7);
    let refreshed = storage.record_commitment_sync_status_at(192);
    assert_eq!(refreshed.follower_readiness_state, "ready");
    assert_eq!(refreshed.follower_convergence_confirmed_at, Some(192));
    assert_eq!(refreshed.follower_readiness_stale_after, Some(282));
}

#[tokio::test]
async fn test_trusted_divergence_halts_local_production_and_survives_convergence() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-trusted-divergence.db");
    let storage = MemoryStorage::open(&path, None).unwrap();
    let proposer = IdentityKeyPair::generate();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0xD1, &proposer);
    storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();

    let witness = IdentityKeyPair::generate();
    let diverged_at = 1_700_525_000;
    let fork_hash = [0xD2; 32];
    let divergent = signed_checkpoint_response_frame(
        &witness,
        [0x71; 16],
        diverged_at,
        1,
        fork_hash,
        1,
        fork_hash,
    );
    let divergent_digest: [u8; 32] = Sha256::digest(&divergent).into();
    assert_eq!(
        storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                diverged_at,
                "diverged",
                1,
                1,
                1,
                &divergent_digest,
                &divergent,
                true,
            )
            .await
            .unwrap(),
        RecordCommitmentCheckpointEvidencePersistOutcome::TrustedDivergenceDetected
    );
    assert!(storage.record_commitment_production_halted());

    let blocked = signed_commitment_block(2, first.hash(), 0xD3, &proposer);
    assert!(storage
        .append_record_commitment_block(&blocked, None)
        .await
        .unwrap_err()
        .contains("production halted"));

    // A verified follower append remains available for recovery and lets
    // us prove that later convergence at another height cannot wash out
    // the original trusted incident.
    let recovery_source = [0xD4; 32];
    storage
        .append_record_commitment_block(&blocked, Some(&recovery_source))
        .await
        .unwrap();
    let converged_at = diverged_at + 1;
    let converged = signed_checkpoint_response_frame(
        &witness,
        [0x72; 16],
        converged_at,
        2,
        blocked.hash(),
        2,
        blocked.hash(),
    );
    let converged_digest: [u8; 32] = Sha256::digest(&converged).into();
    assert_eq!(
        storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                converged_at,
                "converged",
                2,
                2,
                2,
                &converged_digest,
                &converged,
                true,
            )
            .await
            .unwrap(),
        RecordCommitmentCheckpointEvidencePersistOutcome::Stored
    );
    let audit = storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(audit.equivocation_incidents, 0);
    assert_eq!(audit.trusted_divergence_incidents, 1);
    assert_eq!(
        storage
            .count_record_commitment_checkpoint_trusted_divergences_for_witnesses(&[
                witness.public_key_bytes(),
            ])
            .await
            .unwrap(),
        1
    );
    assert!(storage.record_commitment_production_halted());
    drop(storage);

    let reopened = MemoryStorage::open(&path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    let reopened_audit = reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(reopened_audit.trusted_divergence_incidents, 1);
    assert_eq!(
        reopened
            .count_record_commitment_checkpoint_trusted_divergences_for_witnesses(&[
                witness.public_key_bytes(),
            ])
            .await
            .unwrap(),
        1
    );
}

#[tokio::test]
async fn test_permissionless_conflict_cannot_create_authoritative_incident() {
    let (storage, block) = commitment_audit_fixture().await;
    let witness = IdentityKeyPair::generate();
    let first_at = 1_700_530_000;
    let first = signed_checkpoint_response_frame(
        &witness,
        [0x61; 16],
        first_at,
        1,
        block.hash(),
        1,
        block.hash(),
    );
    let first_digest: [u8; 32] = Sha256::digest(&first).into();
    storage
        .persist_record_commitment_checkpoint_evidence(
            first_at,
            "converged",
            1,
            1,
            1,
            &first_digest,
            &first,
        )
        .await
        .unwrap();

    let fork_hash = [0xE2; 32];
    let second = signed_checkpoint_response_frame(
        &witness,
        [0x62; 16],
        first_at + 1,
        1,
        fork_hash,
        1,
        fork_hash,
    );
    let second_digest: [u8; 32] = Sha256::digest(&second).into();
    storage
        .persist_record_commitment_checkpoint_evidence(
            first_at + 1,
            "diverged",
            1,
            1,
            1,
            &second_digest,
            &second,
        )
        .await
        .unwrap();
    let audit = storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(audit.evidence_records, 2);
    assert_eq!(audit.equivocation_incidents, 0);
    assert_eq!(audit.trusted_divergence_incidents, 0);
    assert!(!storage.record_commitment_production_halted());
}

#[tokio::test]
async fn test_block_packer_source_is_active_blind_and_uncommitted_only() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let owner = IdentityKeyPair::generate();
    let mut blind = make_rec_owner(
        1_700_400_001,
        owner.public_key_bytes(),
        MemoryLayer::Episode,
    );
    blind.blind = true;
    blind.signature = owner.sign(&blind.record_id);
    assert!(storage.insert_blind_replica(&blind, "sealed").await);

    let local = make_rec_owner(1_700_400_002, [0x55; 32], MemoryLayer::Episode);
    assert!(storage.insert(&local, "local").await);
    let pending = storage.get_uncommitted_blind_records(32).await;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].record_id, blind.record_id);

    let proposer = IdentityKeyPair::generate();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        1_700_400_003,
        GENESIS_PREV_HASH,
        vec![blind.record_id],
        &proposer,
    );
    storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    assert!(storage.get_uncommitted_blind_records(32).await.is_empty());
}
