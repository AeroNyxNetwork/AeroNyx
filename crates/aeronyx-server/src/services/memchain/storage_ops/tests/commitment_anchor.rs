// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn blocking_anchor_writer_redacts_panic_payload() {
    // [ANCHOR-WORKER-PRIVACY 2026-07-30 by Codex] A panic payload must
    // not enter startup/readiness error strings returned by anchor writes.
    let error = run_blocking_local_anchor_write("test anchor", || {
        panic!("test-only-sensitive-anchor-payload");
    })
    .await
    .unwrap_err();

    assert_eq!(error, "test anchor write worker task panicked");
    assert!(!error.contains("sensitive"));
}

#[tokio::test]
async fn test_count_distinct_owners_multiple() {
    let s = MemoryStorage::open(":memory:", None).unwrap();
    s.insert(&make_rec_owner(100, [0xAA; 32], MemoryLayer::Episode), "m")
        .await;
    s.insert(&make_rec_owner(200, [0xBB; 32], MemoryLayer::Episode), "m")
        .await;
    s.insert(&make_rec_owner(300, [0xCC; 32], MemoryLayer::Episode), "m")
        .await;
    s.insert(&make_rec_owner(400, [0xAA; 32], MemoryLayer::Identity), "m")
        .await;
    assert_eq!(s.count_distinct_owners().await, 3);
}

#[tokio::test]
async fn test_verified_delivery_anchor_witness_is_contiguous_and_restart_durable() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("verified-delivery-witness.db");
    let requester = [0x41; 32];
    let digest_10 = [0x51; 32];
    let digest_11 = [0x52; 32];
    let storage = MemoryStorage::open(&db_path, None).unwrap();

    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 10, &digest_10, 1_000)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Advanced {
            generation: 10,
            anchor_digest: digest_10,
        }
    );
    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 10, &digest_10, 1_001)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Idempotent {
            generation: 10,
            anchor_digest: digest_10,
        }
    );
    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 10, &[0x61; 32], 1_002)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Conflict {
            generation: 10,
            anchor_digest: digest_10,
        }
    );
    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 9, &[0x62; 32], 1_003)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Stale {
            generation: 10,
            anchor_digest: digest_10,
        }
    );
    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 12, &[0x63; 32], 1_004)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Gap {
            generation: 10,
            anchor_digest: digest_10,
        }
    );
    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 11, &digest_11, 1_005)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Advanced {
            generation: 11,
            anchor_digest: digest_11,
        }
    );
    drop(storage);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    assert_eq!(
        reopened
            .witness_verified_delivery_anchor(&requester, 11, &digest_11, 1_006)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Idempotent {
            generation: 11,
            anchor_digest: digest_11,
        }
    );
    let row_count: i64 = {
        let conn = reopened.conn_lock().await;
        conn.query_row(
            "SELECT COUNT(*) FROM verified_delivery_anchor_witnesses",
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    assert_eq!(
        row_count, 1,
        "witness storage must stay bounded per requester"
    );
}

#[tokio::test]
async fn test_verified_delivery_anchor_witness_rejects_sentinel_values() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let requester = [0x71; 32];
    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 0, &[0x72; 32], 1)
            .await
            .unwrap_err(),
        "verified-delivery witness generation must be positive"
    );
    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&requester, 1, &[0u8; 32], 1)
            .await
            .unwrap_err(),
        "verified-delivery witness digest must be non-zero"
    );
}

#[tokio::test]
async fn custody_audit_witness_is_contiguous_restart_durable_and_domain_isolated() {
    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] The same producer may
    // legitimately have unrelated delivery-cache and custody generation
    // values. Each state machine must therefore advance independently.
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("custody-audit-witness.db");
    let producer = [0x73; 32];
    let frame_3 = [0x74; 32];
    let frame_4 = [0x75; 32];
    let storage = MemoryStorage::open(&db_path, None).unwrap();

    assert_eq!(
        storage
            .witness_verified_delivery_anchor(&producer, 40, &[0x76; 32], 900)
            .await
            .unwrap(),
        VerifiedDeliveryAnchorWitnessOutcome::Advanced {
            generation: 40,
            anchor_digest: [0x76; 32],
        }
    );
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&producer, 3, &frame_3, 1_000)
            .await
            .unwrap(),
        CustodyAuditAnchorWitnessOutcome::Advanced {
            generation: 3,
            anchor_digest: frame_3,
        }
    );
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&producer, 3, &frame_3, 1_001)
            .await
            .unwrap(),
        CustodyAuditAnchorWitnessOutcome::Idempotent {
            generation: 3,
            anchor_digest: frame_3,
        }
    );
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&producer, 3, &[0x77; 32], 1_002)
            .await
            .unwrap(),
        CustodyAuditAnchorWitnessOutcome::Conflict {
            generation: 3,
            anchor_digest: frame_3,
        }
    );
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&producer, 2, &[0x78; 32], 1_003)
            .await
            .unwrap(),
        CustodyAuditAnchorWitnessOutcome::Stale {
            generation: 3,
            anchor_digest: frame_3,
        }
    );
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&producer, 5, &[0x79; 32], 1_004)
            .await
            .unwrap(),
        CustodyAuditAnchorWitnessOutcome::Gap {
            generation: 3,
            anchor_digest: frame_3,
        }
    );
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&producer, 4, &frame_4, 1_005)
            .await
            .unwrap(),
        CustodyAuditAnchorWitnessOutcome::Advanced {
            generation: 4,
            anchor_digest: frame_4,
        }
    );
    drop(storage);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    assert_eq!(
        reopened
            .witness_custody_audit_anchor(&producer, 4, &frame_4, 1_006)
            .await
            .unwrap(),
        CustodyAuditAnchorWitnessOutcome::Idempotent {
            generation: 4,
            anchor_digest: frame_4,
        }
    );
    let (custody_rows, delivery_generation): (i64, i64) = {
        let conn = reopened.conn_lock().await;
        let custody_rows = conn
            .query_row(
                "SELECT COUNT(*) FROM custody_audit_anchor_witnesses",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let delivery_generation = conn
            .query_row(
                "SELECT generation FROM verified_delivery_anchor_witnesses WHERE requester=?1",
                params![producer.as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        (custody_rows, delivery_generation)
    };
    assert_eq!(
        custody_rows, 1,
        "custody witness stays bounded per producer"
    );
    assert_eq!(delivery_generation, 40, "witness domains must not collide");
}

#[tokio::test]
async fn custody_audit_witness_rejects_sentinel_values() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&[0u8; 32], 1, &[0x7a; 32], 1)
            .await
            .unwrap_err(),
        "custody-audit witness subject must be non-zero"
    );
    assert_eq!(
        storage
            .witness_custody_audit_anchor(&[0x7b; 32], 1, &[0x7c; 32], 0)
            .await
            .unwrap_err(),
        "custody-audit witness time must be positive"
    );
}

#[tokio::test]
async fn test_commitment_coordinator_upgrades_sqlite_durability() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .durability_mode,
        "normal"
    );
    assert_eq!(
        storage
            .configure_record_commitment_durability(false)
            .await
            .unwrap(),
        "normal"
    );
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .coordinator_fence_state,
        "not_required"
    );
    assert_eq!(
        storage
            .configure_record_commitment_durability(true)
            .await
            .unwrap(),
        "full"
    );
    let effective_level: i64 = {
        let conn = storage.conn_lock().await;
        conn.query_row("PRAGMA synchronous", [], |row| row.get(0))
            .unwrap()
    };
    assert!(effective_level >= 2);
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .durability_mode,
        "full"
    );
    assert_eq!(
        storage
            .record_commitment_chain_integrity_status()
            .coordinator_fence_state,
        "isolated_in_memory"
    );
}

#[tokio::test]
async fn test_witness_coordinator_lease_is_exclusive_renewable_and_restart_durable() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("witness-lease.db");
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x31; 32];
    let first_instance = [0x32; 32];
    let second_instance = [0x33; 32];
    let storage = MemoryStorage::open(&db_path, None).unwrap();

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                1_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 1,
            lease_expires_at: 1_060,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                1_020,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 1,
            lease_expires_at: 1_080,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                1_081,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    drop(storage);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    assert_eq!(
        reopened
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                1_096,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    assert_eq!(
        reopened
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                1_097,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch: 1,
            released_at: 1_097,
        }
    );
    assert_eq!(
        reopened
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                1_098,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 2,
            lease_expires_at: 1_158,
        }
    );

    assert_eq!(
        reopened
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                1,
                &[0xFF; 32],
                1_100,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::TipMismatch
    );

    assert_eq!(
        reopened
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                1_101,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::NotHolder
    );
    assert_eq!(
        reopened
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                1_101,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch: 2,
            released_at: 1_101,
        }
    );
    assert_eq!(
        reopened
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                1_102,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    assert_eq!(
        reopened
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                1_102,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 3,
            lease_expires_at: 1_162,
        }
    );
    assert_eq!(
        reopened
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                1_103,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::NotHolder
    );
}

#[tokio::test]
async fn test_witness_lease_os_authority_excludes_child_until_durable_release() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("witness-process-release.db");
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x45; 32];
    let first_instance = [0x46; 32];
    let parent = MemoryStorage::open(&db_path, None).unwrap();

    assert_eq!(
        parent
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                1_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 1,
            lease_expires_at: 1_060,
        }
    );
    for stage in ["grant_while_parent_locked", "release_while_parent_locked"] {
        let output = run_witness_lease_child(stage, &db_path).await;
        assert_witness_lease_child_succeeded(stage, &output);
    }

    assert_eq!(
        parent
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                1_001,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch: 1,
            released_at: 1_001,
        }
    );
    drop(parent);

    let output = run_witness_lease_child("grant_after_parent_release", &db_path).await;
    assert_witness_lease_child_succeeded("grant_after_parent_release", &output);
}

#[tokio::test]
async fn test_witness_lease_unreleased_child_crash_keeps_restart_hold() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("witness-process-crash.db");
    let crashed = run_witness_lease_child("seed_unreleased_crash", &db_path).await;
    assert_eq!(
        crashed.status.code(),
        Some(WITNESS_LEASE_PROCESS_CRASH_EXIT_CODE),
        "crash worker did not reach the committed lease boundary\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&crashed.stdout),
        String::from_utf8_lossy(&crashed.stderr)
    );
    assert!(
        String::from_utf8_lossy(&crashed.stdout).contains(WITNESS_LEASE_PROCESS_COMMITTED_SIGNAL),
        "crash worker did not flush its post-commit barrier"
    );

    let storage = MemoryStorage::open(&db_path, None).unwrap();
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x45; 32];
    let first_instance = [0x46; 32];
    let second_instance = [0x47; 32];
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                20_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    assert_eq!(
        storage
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                20_001,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch: 1,
            released_at: 20_001,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                20_002,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 2,
            lease_expires_at: 20_062,
        }
    );
}

#[tokio::test]
async fn test_witness_lease_hardlink_alias_path_cannot_create_authority() {
    // [MEMCHAIN-WITNESS-DB-IDENTITY 2026-09-05 by Codex] A path-derived
    // sidecar gives each hardlink spelling a different lock even though
    // both SQLite handles mutate the same main database inode.
    let directory = TempDir::new().unwrap();
    let primary_path = directory.path().join("witness-hardlink-primary.db");
    let alias_path = directory.path().join("witness-hardlink-alias.db");
    drop(MemoryStorage::open(&primary_path, None).unwrap());
    std::fs::hard_link(&primary_path, &alias_path).unwrap();
    let primary = MemoryStorage::open(&primary_path, None).unwrap();
    let alias = MemoryStorage::open(&alias_path, None).unwrap();

    for (storage, instance) in [(&primary, [0x49; 32]), (&alias, [0x4a; 32])] {
        assert_eq!(
            storage
                .grant_record_commitment_coordinator_lease(
                    &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                    &[0x48; 32],
                    &instance,
                    0,
                    &GENESIS_PREV_HASH,
                    1_000,
                    MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                )
                .await
                .unwrap_err(),
            "witness lease database identity is unsafe"
        );
    }
    drop(primary);
    drop(alias);
    let connection = rusqlite::Connection::open(&primary_path).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM record_coordinator_leases",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn test_witness_lease_symlink_alias_path_cannot_split_authority() {
    let directory = TempDir::new().unwrap();
    let primary_path = directory.path().join("witness-symlink-primary.db");
    let alias_path = directory.path().join("witness-symlink-alias.db");
    drop(MemoryStorage::open(&primary_path, None).unwrap());
    std::os::unix::fs::symlink(&primary_path, &alias_path).unwrap();
    let primary = MemoryStorage::open(&primary_path, None).unwrap();
    let alias = MemoryStorage::open(&alias_path, None).unwrap();
    let monotonic_start = Instant::now();

    assert!(matches!(
        primary
            .grant_record_commitment_coordinator_lease_at(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &[0x4b; 32],
                &[0x4c; 32],
                0,
                &GENESIS_PREV_HASH,
                2_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start,
                WitnessLeaseCommitObservation::Observed,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted { lease_epoch: 1, .. }
    ));
    let first_alias_attempt = alias
        .grant_record_commitment_coordinator_lease_at(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &[0x4b; 32],
            &[0x4d; 32],
            0,
            &GENESIS_PREV_HASH,
            20_000,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            monotonic_start,
            WitnessLeaseCommitObservation::Observed,
        )
        .await;
    let expired_restart_hold_attempt = alias
        .grant_record_commitment_coordinator_lease_at(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &[0x4b; 32],
            &[0x4d; 32],
            0,
            &GENESIS_PREV_HASH,
            20_001,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            monotonic_start + Duration::from_secs(WITNESS_LEASE_RESTART_HOLD_SECS + 1),
            WitnessLeaseCommitObservation::Observed,
        )
        .await;
    assert_eq!(
        expired_restart_hold_attempt.unwrap_err(),
        "witness lease database identity is unsafe"
    );
    assert_eq!(
        first_alias_attempt.unwrap_err(),
        "witness lease database identity is unsafe"
    );
}

#[tokio::test]
async fn test_witness_lease_database_replacement_after_authority_is_fail_closed() {
    let directory = TempDir::new().unwrap();
    let active_path = directory.path().join("witness-replaced-active.db");
    let displaced_path = directory.path().join("witness-replaced-displaced.db");
    let replacement_path = directory.path().join("witness-replacement.db");
    let storage = MemoryStorage::open(&active_path, None).unwrap();
    let coordinator = [0x4e; 32];
    let instance = [0x4f; 32];

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &coordinator,
                &instance,
                0,
                &GENESIS_PREV_HASH,
                3_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 1,
            lease_expires_at: 3_060,
        }
    );

    // Control: before replacement a second handle to this exact inode
    // remains fenced by the original identity-derived advisory lock.
    let same_inode_handle = MemoryStorage::open(&active_path, None).unwrap();
    assert_eq!(
        same_inode_handle
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &coordinator,
                &[0x50; 32],
                0,
                &GENESIS_PREV_HASH,
                3_001,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap_err(),
        "witness lease clock is held by another database handle"
    );
    drop(same_inode_handle);

    drop(MemoryStorage::open(&replacement_path, None).unwrap());
    std::fs::rename(&active_path, &displaced_path).unwrap();

    // A missing configured path is as unsafe as a replacement.  The
    // retained connection must not renew its lease through the unlinked
    // SQLite handle while the configured pathname is absent.
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &coordinator,
                &instance,
                0,
                &GENESIS_PREV_HASH,
                3_002,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap_err(),
        "witness lease database identity is unsafe"
    );
    std::fs::rename(&replacement_path, &active_path).unwrap();

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &coordinator,
                &instance,
                0,
                &GENESIS_PREV_HASH,
                3_003,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap_err(),
        "witness lease database identity is unsafe"
    );
    assert_eq!(
        storage
            .release_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &coordinator,
                &instance,
                3_004,
            )
            .await
            .unwrap_err(),
        "witness lease database identity is unsafe"
    );
    assert_eq!(
        storage
            .conn
            .lock()
            .await
            .query_row(
                "SELECT COUNT(*) FROM record_coordinator_leases",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1,
        "path revalidation must reject before the displaced SQLite handle mutates"
    );

    // The process-local witness lock remains conservative after the old
    // handle is dropped, preventing a replacement inode from becoming a
    // second authority in this process.
    drop(storage);

    // A replacement database identity cannot bypass that process-local
    // fence merely because it has a distinct inode-derived lock name.
    let replacement = MemoryStorage::open(&active_path, None).unwrap();
    assert_eq!(
        replacement
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &[0x51; 32],
                &[0x52; 32],
                0,
                &GENESIS_PREV_HASH,
                3_005,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
}

#[tokio::test]
#[ignore = "invoked only as an isolated child of witness lease process tests"]
async fn test_witness_lease_cross_process_worker() {
    let Ok(stage) = std::env::var(WITNESS_LEASE_PROCESS_STAGE_ENV) else {
        return;
    };
    let db_path = std::env::var_os(WITNESS_LEASE_PROCESS_DB_ENV)
        .map(PathBuf::from)
        .expect("witness child database path");
    let storage = MemoryStorage::open(db_path, None).unwrap();
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x45; 32];
    let first_instance = [0x46; 32];
    let second_instance = [0x47; 32];

    match stage.as_str() {
        "grant_while_parent_locked" => assert_eq!(
            storage
                .grant_record_commitment_coordinator_lease(
                    &chain_id,
                    &coordinator,
                    &second_instance,
                    0,
                    &GENESIS_PREV_HASH,
                    10_000,
                    MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                )
                .await
                .unwrap_err(),
            "witness lease clock is held by another database handle"
        ),
        "release_while_parent_locked" => assert_eq!(
            storage
                .release_record_commitment_coordinator_lease(
                    &chain_id,
                    &coordinator,
                    &first_instance,
                    1_001,
                )
                .await
                .unwrap_err(),
            "witness lease clock is held by another database handle"
        ),
        "grant_after_parent_release" => assert_eq!(
            storage
                .grant_record_commitment_coordinator_lease(
                    &chain_id,
                    &coordinator,
                    &second_instance,
                    0,
                    &GENESIS_PREV_HASH,
                    1_002,
                    MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                )
                .await
                .unwrap(),
            RecordCoordinatorLeaseGrantOutcome::Granted {
                lease_epoch: 2,
                lease_expires_at: 1_062,
            }
        ),
        "seed_unreleased_crash" => {
            assert_eq!(
                storage
                    .grant_record_commitment_coordinator_lease(
                        &chain_id,
                        &coordinator,
                        &first_instance,
                        0,
                        &GENESIS_PREV_HASH,
                        2_000,
                        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                    )
                    .await
                    .unwrap(),
                RecordCoordinatorLeaseGrantOutcome::Granted {
                    lease_epoch: 1,
                    lease_expires_at: 2_060,
                }
            );
            // Deliberately skip MemoryStorage/Flock destructors after the
            // SQLite commit; the parent owns and reaps this exact child.
            let mut stdout = std::io::stdout();
            writeln!(stdout, "{WITNESS_LEASE_PROCESS_COMMITTED_SIGNAL}")
                .expect("write witness child post-commit barrier");
            stdout
                .flush()
                .expect("flush witness child post-commit barrier");
            std::process::exit(WITNESS_LEASE_PROCESS_CRASH_EXIT_CODE);
        }
        other => panic!("unsupported witness lease child stage: {other}"),
    }
}

#[tokio::test]
async fn test_witness_lease_correlated_forward_wall_step_does_not_split_same_key() {
    // [MEMCHAIN-WITNESS-CLOCK 2026-09-05 by Codex] Independent witness
    // databases must not all release the same still-live holder merely
    // because their correlated wall clocks jump forward together.
    let directory = TempDir::new().unwrap();
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x35; 32];
    let first_instance = [0x36; 32];
    let second_instance = [0x37; 32];
    let mut witnesses = Vec::new();
    for index in 0..2 {
        let storage = MemoryStorage::open(
            directory
                .path()
                .join(format!("witness-forward-step-{index}.db")),
            None,
        )
        .unwrap();
        assert!(matches!(
            storage
                .grant_record_commitment_coordinator_lease(
                    &chain_id,
                    &coordinator,
                    &first_instance,
                    0,
                    &GENESIS_PREV_HASH,
                    1_000,
                    MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                )
                .await
                .unwrap(),
            RecordCoordinatorLeaseGrantOutcome::Granted { lease_epoch: 1, .. }
        ));
        witnesses.push(storage);
    }

    for storage in &witnesses {
        assert_eq!(
            storage
                .grant_record_commitment_coordinator_lease(
                    &chain_id,
                    &coordinator,
                    &second_instance,
                    0,
                    &GENESIS_PREV_HASH,
                    10_000,
                    MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                )
                .await
                .unwrap(),
            RecordCoordinatorLeaseGrantOutcome::Contended
        );
    }
}

#[tokio::test]
async fn test_witness_lease_monotonic_hold_survives_forward_and_backward_wall_steps() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x38; 32];
    let first_instance = [0x39; 32];
    let second_instance = [0x3a; 32];
    let monotonic_start = Instant::now();

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                1_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start,
                WitnessLeaseCommitObservation::Observed,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 1,
            lease_expires_at: 1_060,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                900,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start + Duration::from_secs(1),
                WitnessLeaseCommitObservation::Observed,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 1,
            lease_expires_at: 1_060,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                10_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start + Duration::from_secs(2),
                WitnessLeaseCommitObservation::Observed,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                10_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start + Duration::from_secs(76),
                WitnessLeaseCommitObservation::Observed,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 2,
            lease_expires_at: 10_060,
        }
    );
}

#[tokio::test]
async fn test_witness_lease_rejects_multiple_handles_for_one_database() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("witness-single-clock.db");
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x3b; 32];
    let first_instance = [0x3c; 32];
    let second_instance = [0x3d; 32];
    let first = MemoryStorage::open(&db_path, None).unwrap();
    let second = MemoryStorage::open(&db_path, None).unwrap();

    assert!(matches!(
        first
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                1_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted { .. }
    ));
    assert_eq!(
        second
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                10_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap_err(),
        "witness lease clock is held by another database handle"
    );
    drop(first);
    assert_eq!(
        second
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                10_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
}

#[tokio::test]
async fn test_witness_lease_clock_rejects_hardlink_without_chmod_side_effect() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("witness-hardlink.db");
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    let external = directory.path().join("external-state");
    std::fs::write(&external, b"external").unwrap();
    std::fs::set_permissions(&external, Permissions::from_mode(0o644)).unwrap();
    let lock_path = commitment_witness_lease_clock_path(storage.database_identity.unwrap());
    if lock_path.exists() {
        std::fs::remove_file(&lock_path).unwrap();
    }
    std::fs::hard_link(&external, &lock_path).unwrap();

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &[0x41; 32],
                &[0x42; 32],
                0,
                &GENESIS_PREV_HASH,
                1_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap_err(),
        "witness lease clock file is unsafe"
    );
    assert_eq!(
        std::fs::metadata(&external).unwrap().permissions().mode() & 0o777,
        0o644
    );
    std::fs::remove_file(lock_path).unwrap();
}

#[tokio::test]
async fn test_witness_lease_clock_converges_across_symlink_parent() {
    let directory = TempDir::new().unwrap();
    let real_parent = directory.path().join("real");
    let linked_parent = directory.path().join("linked");
    std::fs::create_dir(&real_parent).unwrap();
    std::os::unix::fs::symlink(&real_parent, &linked_parent).unwrap();
    let linked_path = linked_parent.join("witness-symlink-parent.db");
    let real_path = real_parent.join("witness-symlink-parent.db");
    let linked = MemoryStorage::open(&linked_path, None).unwrap();
    let real = MemoryStorage::open(&real_path, None).unwrap();

    assert!(matches!(
        linked
            .grant_record_commitment_coordinator_lease(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                &[0x43; 32],
                &[0x44; 32],
                0,
                &GENESIS_PREV_HASH,
                1_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted { lease_epoch: 1, .. }
    ));
    assert_eq!(
        real.grant_record_commitment_coordinator_lease(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &[0x43; 32],
            &[0x45; 32],
            0,
            &GENESIS_PREV_HASH,
            10_000,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        )
        .await
        .unwrap_err(),
        "witness lease clock is held by another database handle"
    );
}

#[tokio::test]
async fn test_witness_lease_unknown_commit_outcomes_keep_conservative_hold() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let coordinator = [0x3e; 32];
    let first_instance = [0x3f; 32];
    let second_instance = [0x40; 32];
    let monotonic_start = Instant::now();

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                1_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start,
                WitnessLeaseCommitObservation::UnknownAfterCommit,
            )
            .await
            .unwrap_err(),
        "coordinator lease commit outcome is unknown"
    );
    assert_eq!(
        storage
            .release_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &first_instance,
                1_001,
                monotonic_start + Duration::from_secs(1),
                WitnessLeaseCommitObservation::UnknownAfterCommit,
            )
            .await
            .unwrap_err(),
        "coordinator lease release commit outcome is unknown"
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                10_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start + Duration::from_secs(2),
                WitnessLeaseCommitObservation::Observed,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    assert!(matches!(
        storage
            .grant_record_commitment_coordinator_lease_at(
                &chain_id,
                &coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                10_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
                monotonic_start + Duration::from_secs(WITNESS_LEASE_RESTART_HOLD_SECS + 2),
                WitnessLeaseCommitObservation::Observed,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted { lease_epoch: 2, .. }
    ));
}

#[tokio::test]
async fn test_witness_coordinator_lease_is_exclusive_across_coordinator_identities() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("witness-chain-lease.db");
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let first_coordinator = [0x41; 32];
    let first_instance = [0x42; 32];
    let second_coordinator = [0x51; 32];
    let second_instance = [0x52; 32];
    let storage = MemoryStorage::open(&db_path, None).unwrap();

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &first_coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                2_000,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 1,
            lease_expires_at: 2_060,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &second_coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                2_020,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    assert_eq!(
        storage
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &first_coordinator,
                &first_instance,
                2_021,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch: 1,
            released_at: 2_021,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &second_coordinator,
                &second_instance,
                0,
                &GENESIS_PREV_HASH,
                2_022,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 2,
            lease_expires_at: 2_082,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &first_coordinator,
                &first_instance,
                0,
                &GENESIS_PREV_HASH,
                2_023,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );

    let conn = storage.conn.lock().await;
    let row_count = conn
        .query_row(
            "SELECT COUNT(*) FROM record_coordinator_leases WHERE chain_id=?1",
            params![chain_id.as_slice()],
            |row| row.get::<_, i64>(0),
        )
        .unwrap();
    assert_eq!(row_count, 1, "a chain must retain exactly one lease row");
    let stored_coordinator = conn
        .query_row(
            "SELECT coordinator FROM record_coordinator_leases WHERE chain_id=?1",
            params![chain_id.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .unwrap();
    assert_eq!(stored_coordinator.as_slice(), second_coordinator.as_slice());
}

#[tokio::test]
async fn test_witness_coordinator_lease_normalizes_legacy_rows_with_chain_wide_epoch() {
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let first_coordinator = [0x61_u8; 32];
    let second_coordinator = [0x62_u8; 32];
    let second_instance = [0x72_u8; 32];
    let replacement_coordinator = [0x63; 32];
    let replacement_instance = [0x64; 32];
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    {
        let conn = storage.conn.lock().await;
        conn.execute(
            "INSERT INTO record_coordinator_leases
                    (coordinator, chain_id, instance_id, lease_epoch,
                     lease_expires_at, updated_at)
                 VALUES (?1, ?2, ?3, 4, 100, 100)",
            params![
                first_coordinator.as_slice(),
                chain_id.as_slice(),
                [0x71_u8; 32].as_slice(),
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO record_coordinator_leases
                    (coordinator, chain_id, instance_id, lease_epoch,
                     lease_expires_at, updated_at)
                 VALUES (?1, ?2, ?3, 7, 120, 110)",
            params![
                second_coordinator.as_slice(),
                chain_id.as_slice(),
                second_instance.as_slice(),
            ],
        )
        .unwrap();
    }

    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &replacement_coordinator,
                &replacement_instance,
                0,
                &GENESIS_PREV_HASH,
                200,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Contended
    );
    assert_eq!(
        storage
            .release_record_commitment_coordinator_lease(
                &chain_id,
                &second_coordinator,
                &second_instance,
                200,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch: 7,
            released_at: 200,
        }
    );
    assert_eq!(
        storage
            .grant_record_commitment_coordinator_lease(
                &chain_id,
                &replacement_coordinator,
                &replacement_instance,
                0,
                &GENESIS_PREV_HASH,
                200,
                MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            )
            .await
            .unwrap(),
        RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch: 8,
            lease_expires_at: 260,
        }
    );

    let conn = storage.conn.lock().await;
    let (row_count, stored_coordinator, stored_epoch) = conn
        .query_row(
            "SELECT COUNT(*), coordinator, lease_epoch
                 FROM record_coordinator_leases WHERE chain_id=?1",
            params![chain_id.as_slice()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(row_count, 1);
    assert_eq!(
        stored_coordinator.as_slice(),
        replacement_coordinator.as_slice()
    );
    assert_eq!(stored_epoch, 8);
}

#[tokio::test]
async fn test_commitment_tip_anchor_initializes_advances_and_verifies_after_restart() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("anchor-restart.db");
    let anchor_path = directory.path().join("commitment-tip.json");
    let identity = IdentityKeyPair::generate();

    let storage = MemoryStorage::open(&db_path, None).unwrap();
    storage
        .configure_record_commitment_durability(true)
        .await
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    assert_eq!(
        storage
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap(),
        "initialized"
    );
    let initialized = storage.record_commitment_chain_integrity_status();
    assert_eq!(initialized.rollback_guard_state, "initialized");
    assert_eq!(initialized.rollback_guard_height, 0);
    assert!(anchor_path.is_file());

    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x61, &identity);
    assert_eq!(
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap(),
        RecordCommitmentAppendOutcome::Inserted
    );
    let advanced = storage.record_commitment_chain_integrity_status();
    assert_eq!(advanced.state, "verified");
    assert_eq!(advanced.rollback_guard_state, "verified");
    assert_eq!(advanced.rollback_guard_height, 1);
    assert!(advanced.rollback_guard_last_verified_at.is_some());
    assert!(advanced.rollback_guard_last_persisted_at.is_some());
    drop(storage);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    reopened
        .configure_record_commitment_durability(true)
        .await
        .unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    assert_eq!(
        reopened
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap(),
        "verified"
    );
    let verified = reopened.record_commitment_chain_integrity_status();
    assert_eq!(verified.rollback_guard_state, "verified");
    assert_eq!(verified.rollback_guard_height, 1);
    assert_eq!(verified.verified_tip_height, 1);
}

#[tokio::test]
async fn test_commitment_tip_anchor_rejects_valid_older_database_snapshot() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("anchor-rollback.db");
    let anchor_path = directory.path().join("commitment-tip.json");
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    storage
        .configure_record_commitment_tip_anchor(&anchor_path, &identity)
        .await
        .unwrap();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x62, &identity);
    let second = signed_commitment_block(2, first.hash(), 0x63, &identity);
    storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();
    storage
        .append_record_commitment_block(&second, None)
        .await
        .unwrap();
    drop(storage);

    // Simulate restoring a self-consistent SQLite snapshot from height 1
    // while leaving the separately persisted signed high-water mark at 2.
    let mut rollback = rusqlite::Connection::open(&db_path).unwrap();
    rollback.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
    let transaction = rollback.transaction().unwrap();
    transaction
        .execute(
            "DELETE FROM record_block_commitments WHERE block_height=2",
            [],
        )
        .unwrap();
    transaction
        .execute("DELETE FROM record_commitment_blocks WHERE height=2", [])
        .unwrap();
    for (key, value) in [
        ("record_block_tip_hash", first.hash().to_vec()),
        ("record_block_tip_height", 1u64.to_le_bytes().to_vec()),
    ] {
        transaction
            .execute(
                "INSERT OR REPLACE INTO chain_state (key,value) VALUES (?1,?2)",
                params![key, value],
            )
            .unwrap();
    }
    transaction.commit().unwrap();
    drop(rollback);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    let audit = reopened.audit_record_commitment_chain().await.unwrap();
    assert_eq!(audit.tip_height, 1);
    let error = reopened
        .configure_record_commitment_tip_anchor(&anchor_path, &identity)
        .await
        .unwrap_err();
    assert!(error.contains("behind signed local anchor height 2"));
    let rejected = reopened.record_commitment_chain_integrity_status();
    assert_eq!(rejected.state, "not_verified");
    assert_eq!(rejected.rollback_guard_state, "rollback_detected");
    assert_eq!(rejected.rollback_guard_height, 2);
}

#[tokio::test]
async fn test_commitment_tip_anchor_rejects_same_height_hash_conflict() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("anchor-conflict.db");
    let anchor_path = directory.path().join("commitment-tip.json");
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    storage
        .configure_record_commitment_tip_anchor(&anchor_path, &identity)
        .await
        .unwrap();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x64, &identity);
    storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();
    drop(storage);

    // The conflict is signed by the correct node identity. Rejection must
    // therefore come from ancestry comparison, not signature validation.
    let conflicting =
        RecordCommitmentTipAnchorV1::new_signed(1, [0xA7; 32], &identity, unix_now_secs());
    let bytes = serde_json::to_vec(&conflicting).unwrap();
    write_record_commitment_tip_anchor_atomic(&anchor_path, &bytes).unwrap();

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    let error = reopened
        .configure_record_commitment_tip_anchor(&anchor_path, &identity)
        .await
        .unwrap_err();
    assert!(error.contains("ancestry mismatch at height 1"));
    let rejected = reopened.record_commitment_chain_integrity_status();
    assert_eq!(rejected.state, "not_verified");
    assert_eq!(rejected.rollback_guard_state, "rollback_detected");
}

#[tokio::test]
async fn test_commitment_tip_anchor_repairs_audited_database_ahead_state() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("anchor-db-ahead.db");
    let anchor_path = directory.path().join("commitment-tip.json");
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    storage
        .configure_record_commitment_tip_anchor(&anchor_path, &identity)
        .await
        .unwrap();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x65, &identity);
    storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();
    drop(storage);

    // Model the only expected mismatch window: SQLite committed a later
    // verified block, then the process stopped before sidecar replacement.
    let db_ahead = MemoryStorage::open(&db_path, None).unwrap();
    db_ahead.audit_record_commitment_chain().await.unwrap();
    let second = signed_commitment_block(2, first.hash(), 0x66, &identity);
    db_ahead
        .append_record_commitment_block(&second, None)
        .await
        .unwrap();
    drop(db_ahead);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    assert_eq!(
        reopened
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap(),
        "repaired"
    );
    let repaired = reopened.record_commitment_chain_integrity_status();
    assert_eq!(repaired.state, "verified");
    assert_eq!(repaired.rollback_guard_state, "repaired");
    assert_eq!(repaired.rollback_guard_height, 2);
}

#[tokio::test]
async fn test_commitment_tip_anchor_write_failure_commits_once_then_fails_closed() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("anchor-write-failure.db");
    let anchor_parent = directory.path().join("anchor-parent");
    let anchor_path = anchor_parent.join("commitment-tip.json");
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    storage
        .configure_record_commitment_tip_anchor(&anchor_path, &identity)
        .await
        .unwrap();

    std::fs::remove_file(&anchor_path).unwrap();
    std::fs::remove_dir(&anchor_parent).unwrap();
    std::fs::write(&anchor_parent, b"blocks child creation").unwrap();

    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x67, &identity);
    let error = storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap_err();
    assert!(error.contains("block was committed"));
    assert!(error.contains("anchor persistence failed"));
    assert_eq!(storage.record_commitment_chain_tip().await.0, 1);
    let failed = storage.record_commitment_chain_integrity_status();
    assert_eq!(failed.state, "not_verified");
    assert_eq!(failed.rollback_guard_state, "write_failed");
    assert_eq!(failed.rollback_guard_height, 0);
    assert_eq!(failed.rollback_guard_write_failures_total, 1);

    let second = signed_commitment_block(2, first.hash(), 0x68, &identity);
    let blocked = storage
        .append_record_commitment_block(&second, None)
        .await
        .unwrap_err();
    assert!(blocked.contains("anchor is not ready"));
    assert_eq!(storage.record_commitment_chain_tip().await.0, 1);
}

#[tokio::test]
async fn test_checkpoint_certificate_anchor_initializes_and_verifies_after_restart() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("certificate-anchor-restart.db");
    let tip_anchor_path = directory.path().join("commitment-tip.json");
    let certificate_anchor_path = checkpoint_certificate_anchor_path(&tip_anchor_path).unwrap();
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    let (witnesses, digests) = seed_checkpoint_certificate(&storage, 1_700_610_000).await;
    storage
        .persist_record_commitment_checkpoint_certificate(1_700_610_010, 2, &witnesses, &digests)
        .await
        .unwrap();
    storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(
        storage
            .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity,)
            .await
            .unwrap(),
        "initialized"
    );
    assert!(certificate_anchor_path.is_file());
    let initialized = storage.record_commitment_checkpoint_status();
    assert_eq!(initialized.certificate_rollback_guard_state, "initialized");
    assert_eq!(initialized.certificate_rollback_guard_height, 1);
    drop(storage);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(
        reopened
            .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity,)
            .await
            .unwrap(),
        "verified"
    );
    let verified = reopened.record_commitment_checkpoint_status();
    assert_eq!(verified.certificate_rollback_guard_state, "verified");
    assert_eq!(verified.certificate_rollback_guard_height, 1);
    assert_eq!(verified.checkpoint_certificates, 1);
}

#[tokio::test]
async fn test_checkpoint_certificate_anchor_rejects_older_vault_snapshot() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("certificate-anchor-rollback.db");
    let tip_anchor_path = directory.path().join("commitment-tip.json");
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    let (witnesses, digests) = seed_checkpoint_certificate(&storage, 1_700_610_100).await;
    storage
        .persist_record_commitment_checkpoint_certificate(1_700_610_110, 2, &witnesses, &digests)
        .await
        .unwrap();
    storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    storage
        .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity)
        .await
        .unwrap();
    drop(storage);

    // Restore only the certificate vault to its valid pre-certificate
    // state while preserving the independently signed high-water sidecar.
    let rollback = rusqlite::Connection::open(&db_path).unwrap();
    rollback
        .execute("DELETE FROM record_checkpoint_certificate_members", [])
        .unwrap();
    rollback
        .execute("DELETE FROM record_checkpoint_certificates", [])
        .unwrap();
    drop(rollback);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    let error = reopened
        .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity)
        .await
        .unwrap_err();
    assert!(error.contains("behind signed local anchor height 1"));
    let rejected = reopened.record_commitment_checkpoint_status();
    assert_eq!(
        rejected.certificate_rollback_guard_state,
        "rollback_detected"
    );
    assert_eq!(rejected.certificate_rollback_guard_height, 1);
    assert_eq!(
        reopened.record_commitment_chain_integrity_status().state,
        "not_verified"
    );
}

#[tokio::test]
async fn test_checkpoint_certificate_anchor_rejects_signed_same_height_conflict() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("certificate-anchor-conflict.db");
    let tip_anchor_path = directory.path().join("commitment-tip.json");
    let certificate_anchor_path = checkpoint_certificate_anchor_path(&tip_anchor_path).unwrap();
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    let (witnesses, digests) = seed_checkpoint_certificate(&storage, 1_700_610_200).await;
    storage
        .persist_record_commitment_checkpoint_certificate(1_700_610_210, 2, &witnesses, &digests)
        .await
        .unwrap();
    storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    storage
        .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity)
        .await
        .unwrap();
    let mut conflicting = {
        let conn = storage.conn_lock().await;
        read_latest_checkpoint_certificate_anchor_state(&conn)
            .unwrap()
            .unwrap()
    };
    conflicting.certificate_digest = [0xA9; 32];
    let signed =
        RecordCheckpointCertificateAnchorV1::new_signed(conflicting, &identity, unix_now_secs());
    write_checkpoint_certificate_anchor_atomic(
        &certificate_anchor_path,
        &serde_json::to_vec(&signed).unwrap(),
    )
    .unwrap();
    drop(storage);

    let reopened = MemoryStorage::open(&db_path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    let error = reopened
        .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity)
        .await
        .unwrap_err();
    assert!(error.contains("conflicts at height 1"));
    assert_eq!(
        reopened
            .record_commitment_checkpoint_status()
            .certificate_rollback_guard_state,
        "rollback_detected"
    );
}

#[tokio::test]
async fn test_checkpoint_certificate_anchor_repairs_db_ahead_and_fails_closed_on_write_error() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("certificate-anchor-repair.db");
    let tip_anchor_path = directory.path().join("anchor-parent/commitment-tip.json");
    let certificate_anchor_path = checkpoint_certificate_anchor_path(&tip_anchor_path).unwrap();
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    let (witnesses, digests) = seed_checkpoint_certificate(&storage, 1_700_610_300).await;
    storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    storage
        .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity)
        .await
        .unwrap();
    drop(storage);

    // Model SQLite committing first and the process stopping before the
    // certificate sidecar replacement. Startup must safely advance it.
    let db_ahead = MemoryStorage::open(&db_path, None).unwrap();
    db_ahead.audit_record_commitment_chain().await.unwrap();
    db_ahead
        .persist_record_commitment_checkpoint_certificate(1_700_610_310, 2, &witnesses, &digests)
        .await
        .unwrap();
    drop(db_ahead);
    let repaired = MemoryStorage::open(&db_path, None).unwrap();
    repaired.audit_record_commitment_chain().await.unwrap();
    repaired
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(
        repaired
            .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity,)
            .await
            .unwrap(),
        "repaired"
    );
    assert_eq!(
        repaired
            .record_commitment_checkpoint_status()
            .certificate_rollback_guard_height,
        1
    );
    drop(repaired);

    // Reset to a pre-certificate DB plus an empty anchor, then inject a
    // filesystem failure after the next certificate transaction commits.
    let rollback = rusqlite::Connection::open(&db_path).unwrap();
    rollback
        .execute("DELETE FROM record_checkpoint_certificate_members", [])
        .unwrap();
    rollback
        .execute("DELETE FROM record_checkpoint_certificates", [])
        .unwrap();
    drop(rollback);
    let empty = RecordCheckpointCertificateAnchorV1::new_signed(
        CheckpointCertificateAnchorState::EMPTY,
        &identity,
        unix_now_secs(),
    );
    write_checkpoint_certificate_anchor_atomic(
        &certificate_anchor_path,
        &serde_json::to_vec(&empty).unwrap(),
    )
    .unwrap();
    let failing = MemoryStorage::open(&db_path, None).unwrap();
    failing.audit_record_commitment_chain().await.unwrap();
    failing
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    failing
        .configure_record_commitment_checkpoint_certificate_anchor(&tip_anchor_path, &identity)
        .await
        .unwrap();
    std::fs::remove_file(&certificate_anchor_path).unwrap();
    let anchor_parent = certificate_anchor_path.parent().unwrap();
    std::fs::remove_dir(anchor_parent).unwrap();
    std::fs::write(anchor_parent, b"blocks child creation").unwrap();
    let error = failing
        .persist_record_commitment_checkpoint_certificate(1_700_610_320, 2, &witnesses, &digests)
        .await
        .unwrap_err();
    assert!(error.contains("certificate was committed"));
    assert!(error.contains("anchor persistence failed"));
    let failed = failing.record_commitment_checkpoint_status();
    assert_eq!(failed.checkpoint_certificates, 1);
    assert_eq!(failed.certificate_rollback_guard_state, "write_failed");
    assert_eq!(failed.certificate_rollback_guard_write_failures_total, 1);
    assert_eq!(
        failing.record_commitment_chain_integrity_status().state,
        "not_verified"
    );
    let blocked = failing
        .persist_record_commitment_checkpoint_certificate(1_700_610_321, 2, &witnesses, &digests)
        .await
        .unwrap_err();
    assert!(blocked.contains("anchor is not ready"));
}

#[tokio::test]
async fn test_commitment_append_failure_rolls_back_entire_block_after_reopen() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("commitment-atomicity.db");
    let storage = MemoryStorage::open(&path, None).unwrap();
    storage
        .configure_record_commitment_durability(true)
        .await
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();

    // Inject a deterministic storage failure after the first membership
    // row. The block row, both memberships, and chain_state tip must remain
    // one atomic unit even when the transaction aborts mid-append.
    {
        let conn = storage.conn_lock().await;
        conn.execute_batch(
            "CREATE TEMP TRIGGER fail_second_commitment_membership
                 BEFORE INSERT ON record_block_commitments
                 WHEN (SELECT COUNT(*) FROM record_block_commitments) = 1
                 BEGIN
                     SELECT RAISE(ABORT, 'injected membership failure');
                 END;",
        )
        .unwrap();
    }
    let identity = IdentityKeyPair::generate();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        1_700_190_001,
        GENESIS_PREV_HASH,
        vec![[0x91; 32], [0x92; 32]],
        &identity,
    );
    assert!(storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap_err()
        .contains("injected membership failure"));
    {
        let conn = storage.conn_lock().await;
        conn.execute_batch("DROP TRIGGER fail_second_commitment_membership;")
            .unwrap();
        let (blocks, memberships, tip_keys): (i64, i64, i64) = conn
            .query_row(
                "SELECT
                         (SELECT COUNT(*) FROM record_commitment_blocks),
                         (SELECT COUNT(*) FROM record_block_commitments),
                         (SELECT COUNT(*) FROM chain_state WHERE key IN
                            ('record_block_tip_hash','record_block_tip_height',
                             'record_block_chain_id'))",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!((blocks, memberships, tip_keys), (0, 0, 0));
    }
    drop(storage);

    // A fresh process must see the same empty, valid chain. This catches
    // accidental reliance on connection-local rollback visibility.
    let reopened = MemoryStorage::open(&path, None).unwrap();
    reopened
        .configure_record_commitment_durability(true)
        .await
        .unwrap();
    assert_eq!(
        reopened.audit_record_commitment_chain().await.unwrap(),
        RecordCommitmentChainAudit {
            block_count: 0,
            commitment_count: 0,
            tip_height: 0,
        }
    );
}

#[tokio::test]
async fn test_commitment_page_failure_rolls_back_every_block() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("commitment-page-atomicity.db");
    let storage = MemoryStorage::open(&path, None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();

    // Fail while inserting the second block. A per-block transaction would
    // leave height 1 durable; the page transaction must leave no trace.
    {
        let conn = storage.conn_lock().await;
        conn.execute_batch(
            "CREATE TEMP TRIGGER fail_second_page_block
                 BEFORE INSERT ON record_commitment_blocks
                 WHEN NEW.height = 2
                 BEGIN
                     SELECT RAISE(ABORT, 'injected second block failure');
                 END;",
        )
        .unwrap();
    }
    let identity = IdentityKeyPair::generate();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0xA1, &identity);
    let second = signed_commitment_block(2, first.hash(), 0xA2, &identity);
    let error = storage
        .append_record_commitment_blocks_atomic(&[first.clone(), second.clone()], None)
        .await
        .unwrap_err();
    assert!(error.contains("injected second block failure"));
    assert_eq!(
        storage.record_commitment_chain_tip().await,
        (0, GENESIS_PREV_HASH)
    );
    {
        let conn = storage.conn_lock().await;
        let (blocks, memberships, tip_keys): (i64, i64, i64) = conn
            .query_row(
                "SELECT
                         (SELECT COUNT(*) FROM record_commitment_blocks),
                         (SELECT COUNT(*) FROM record_block_commitments),
                         (SELECT COUNT(*) FROM chain_state WHERE key IN
                            ('record_block_tip_hash','record_block_tip_height',
                             'record_block_chain_id'))",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!((blocks, memberships, tip_keys), (0, 0, 0));
        conn.execute_batch("DROP TRIGGER fail_second_page_block;")
            .unwrap();
    }

    // Cross-block commitment reuse is a consensus-invalid page, not a
    // reason to retain the valid prefix that happened to be processed first.
    let duplicate_second = signed_commitment_block(2, first.hash(), 0xA1, &identity);
    let duplicate_error = storage
        .append_record_commitment_blocks_atomic(&[first.clone(), duplicate_second], None)
        .await
        .unwrap_err();
    assert!(duplicate_error.contains("persist commitment membership"));
    assert_eq!(
        storage.record_commitment_chain_tip().await,
        (0, GENESIS_PREV_HASH)
    );

    let outcome = storage
        .append_record_commitment_blocks_atomic(&[first, second.clone()], None)
        .await
        .unwrap();
    assert_eq!(
        outcome,
        RecordCommitmentBatchAppendOutcome {
            inserted: 2,
            already_present: 0,
        }
    );
    assert_eq!(
        storage.record_commitment_chain_tip().await,
        (2, second.hash())
    );
    assert_eq!(
        storage.record_commitment_chain_integrity_status().state,
        "verified"
    );
}

#[tokio::test]
async fn test_commitment_page_retry_is_idempotent_and_rejects_forks() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let identity = IdentityKeyPair::generate();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0xB1, &identity);
    let second = signed_commitment_block(2, first.hash(), 0xB2, &identity);
    let third = signed_commitment_block(3, second.hash(), 0xB3, &identity);

    storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();
    let mixed = storage
        .append_record_commitment_blocks_atomic(
            &[first.clone(), second.clone(), third.clone()],
            None,
        )
        .await
        .unwrap();
    assert_eq!(mixed.inserted, 2);
    assert_eq!(mixed.already_present, 1);

    let replay = storage
        .append_record_commitment_blocks_atomic(
            &[first.clone(), second.clone(), third.clone()],
            None,
        )
        .await
        .unwrap();
    assert_eq!(replay.inserted, 0);
    assert_eq!(replay.already_present, 3);

    let fork = signed_commitment_block(2, first.hash(), 0xBF, &identity);
    let error = storage
        .append_record_commitment_blocks_atomic(&[first.clone(), fork], None)
        .await
        .unwrap_err();
    assert!(error.contains("fork at height 2"));
    assert_eq!(
        storage.record_commitment_chain_tip().await,
        (3, third.hash())
    );

    let oversized = vec![first.clone(); MAX_ATOMIC_COMMITMENT_BLOCK_BATCH + 1];
    let oversized_error = storage
        .append_record_commitment_blocks_atomic(&oversized, None)
        .await
        .unwrap_err();
    assert!(oversized_error.contains("exceeds maximum"));

    let non_contiguous = storage
        .append_record_commitment_blocks_atomic(&[first, third], None)
        .await
        .unwrap_err();
    assert!(non_contiguous.contains("not height-contiguous"));
    assert_eq!(
        storage
            .append_record_commitment_blocks_atomic(&[], None)
            .await
            .unwrap(),
        RecordCommitmentBatchAppendOutcome::default()
    );
}

#[tokio::test]
async fn test_commitment_page_advances_integrity_and_anchor_to_final_tip() {
    let directory = TempDir::new().unwrap();
    let db_path = directory.path().join("commitment-page-anchor.db");
    let anchor_path = directory.path().join("commitment-page-tip.json");
    let identity = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(&db_path, None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    storage
        .configure_record_commitment_tip_anchor(&anchor_path, &identity)
        .await
        .unwrap();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0xC1, &identity);
    let second = signed_commitment_block(2, first.hash(), 0xC2, &identity);

    let outcome = storage
        .append_record_commitment_blocks_atomic(&[first, second.clone()], None)
        .await
        .unwrap();
    assert_eq!(outcome.inserted, 2);
    let status = storage.record_commitment_chain_integrity_status();
    assert_eq!(status.state, "verified");
    assert_eq!(status.verified_block_count, 2);
    assert_eq!(status.verified_commitment_count, 2);
    assert_eq!(status.verified_tip_height, 2);
    assert_eq!(status.rollback_guard_state, "verified");
    assert_eq!(status.rollback_guard_height, 2);
    assert_eq!(
        storage.record_commitment_chain_tip().await,
        (2, second.hash())
    );
}

#[tokio::test]
async fn test_commitment_chain_append_range_and_status() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    assert!(storage
        .record_commitment_chain_checkpoint(0)
        .await
        .unwrap_err()
        .contains("not fully audited"));
    assert_eq!(
        storage.audit_record_commitment_chain().await.unwrap(),
        RecordCommitmentChainAudit {
            block_count: 0,
            commitment_count: 0,
            tip_height: 0,
        }
    );
    let empty_integrity = storage.record_commitment_chain_integrity_status();
    assert_eq!(empty_integrity.state, "verified");
    assert_eq!(empty_integrity.verified_block_count, 0);
    assert_eq!(empty_integrity.verified_commitment_count, 0);
    assert_eq!(empty_integrity.verified_tip_height, 0);
    assert!(empty_integrity.baseline_verified_at.is_some());
    assert!(empty_integrity.verification_duration_ms.is_some());
    assert_eq!(
        storage.record_commitment_chain_checkpoint(0).await.unwrap(),
        (0, GENESIS_PREV_HASH, 0, GENESIS_PREV_HASH)
    );
    let identity = IdentityKeyPair::generate();
    let first = RecordCommitmentBlockV1::new_signed(
        1,
        1_700_200_001,
        GENESIS_PREV_HASH,
        vec![[0x01; 32], [0x02; 32]],
        &identity,
    );
    assert_eq!(
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap(),
        RecordCommitmentAppendOutcome::Inserted
    );
    assert_eq!(
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap(),
        RecordCommitmentAppendOutcome::AlreadyPresent
    );

    let second = RecordCommitmentBlockV1::new_signed(
        2,
        1_700_200_002,
        first.hash(),
        vec![[0x03; 32]],
        &identity,
    );
    storage
        .append_record_commitment_block(&second, Some(&identity.public_key_bytes()))
        .await
        .unwrap();

    let range = storage
        .get_record_commitment_block_range(1, 10)
        .await
        .unwrap();
    assert_eq!(range, vec![first.clone(), second.clone()]);
    let first_page = storage
        .get_verified_record_commitment_block_page(1, 1)
        .await
        .unwrap();
    assert_eq!(first_page.blocks, vec![first.clone()]);
    assert_eq!(first_page.tip_height, 2);
    assert_eq!(first_page.tip_hash, second.hash());
    let terminal_page = storage
        .get_verified_record_commitment_block_page(2, 10)
        .await
        .unwrap();
    assert_eq!(terminal_page.blocks, vec![second.clone()]);
    assert_eq!(terminal_page.tip_height, 2);
    assert_eq!(terminal_page.tip_hash, second.hash());
    assert_eq!(
        storage.record_commitment_chain_tip().await,
        (2, second.hash())
    );
    assert_eq!(storage.last_block_height().await, 2);
    assert_eq!(storage.last_block_hash().await, second.hash());
    assert_eq!(
        storage.record_commitment_chain_checkpoint(1).await.unwrap(),
        (1, first.hash(), 2, second.hash())
    );
    assert_eq!(
        storage
            .record_commitment_chain_checkpoint(u64::MAX)
            .await
            .unwrap(),
        (2, second.hash(), 2, second.hash())
    );

    let status = storage.record_commitment_chain_status().await;
    assert_eq!(status.block_count, 2);
    assert_eq!(status.commitment_count, 3);
    assert_eq!(status.tip_height, 2);
    assert_eq!(status.tip_hash, Some(hex::encode(second.hash())));
    assert_eq!(status.integrity.state, "verified");
    assert_eq!(status.integrity.verified_block_count, 2);
    assert_eq!(status.integrity.verified_commitment_count, 3);
    assert_eq!(status.integrity.verified_tip_height, 2);
    assert!(status.integrity.last_verified_at >= status.integrity.baseline_verified_at);
    assert_eq!(
        storage.audit_record_commitment_chain().await.unwrap(),
        RecordCommitmentChainAudit {
            block_count: 2,
            commitment_count: 3,
            tip_height: 2,
        }
    );
}

#[tokio::test]
async fn test_verified_commitment_range_fails_closed_without_current_audit() {
    let unaudited = MemoryStorage::open(":memory:", None).unwrap();
    assert!(unaudited
        .get_verified_record_commitment_block_page(1, 16)
        .await
        .unwrap_err()
        .contains("not fully audited"));

    let stale = MemoryStorage::open(":memory:", None).unwrap();
    stale.audit_record_commitment_chain().await.unwrap();
    let identity = IdentityKeyPair::generate();
    let block = signed_commitment_block(1, GENESIS_PREV_HASH, 0xD1, &identity);
    stale
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    {
        let conn = stale.conn.lock().await;
        conn.execute(
            "UPDATE record_commitment_blocks SET block_hash=?1 WHERE height=1",
            params![[0xD2_u8; 32].as_slice()],
        )
        .unwrap();
    }
    assert!(stale
        .get_verified_record_commitment_block_page(1, 16)
        .await
        .unwrap_err()
        .contains("baseline is stale"));

    let corrupt = MemoryStorage::open(":memory:", None).unwrap();
    corrupt.audit_record_commitment_chain().await.unwrap();
    corrupt
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    {
        let conn = corrupt.conn.lock().await;
        conn.execute(
            "UPDATE record_commitment_blocks SET payload=?1 WHERE height=1",
            params![vec![0xFF_u8; 32]],
        )
        .unwrap();
    }
    assert!(corrupt
        .get_verified_record_commitment_block_page(1, 16)
        .await
        .unwrap_err()
        .contains("payload decode failed"));
}

#[tokio::test]
async fn test_commitment_chain_audit_rejects_payload_corruption() {
    let (storage, _) = commitment_audit_fixture().await;
    assert_eq!(
        storage.record_commitment_chain_integrity_status().state,
        "verified"
    );
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "UPDATE record_commitment_blocks SET payload=?1 WHERE height=1",
            params![vec![0xFFu8]],
        )
        .unwrap();
    }
    assert!(storage
        .audit_record_commitment_chain()
        .await
        .unwrap_err()
        .contains("payload decode failed"));
    let integrity = storage.record_commitment_chain_integrity_status();
    assert_eq!(integrity.state, "not_verified");
    assert_eq!(integrity.verified_block_count, 0);
    assert_eq!(integrity.verified_commitment_count, 0);
    assert_eq!(integrity.verified_tip_height, 0);
}

#[tokio::test]
async fn test_commitment_chain_audit_bounds_persisted_payload_before_decode() {
    let (storage, _) = commitment_audit_fixture().await;
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "UPDATE record_commitment_blocks SET payload=?1 WHERE height=1",
            params![vec![0u8; MAX_STORED_COMMITMENT_BLOCK_BYTES + 1]],
        )
        .unwrap();
    }
    assert!(storage
        .audit_record_commitment_chain()
        .await
        .unwrap_err()
        .contains("payload exceeds bound"));
}

#[tokio::test]
async fn test_commitment_chain_audit_rejects_denormalized_row_tampering() {
    let (storage, _) = commitment_audit_fixture().await;
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "UPDATE record_commitment_blocks SET merkle_root=?1 WHERE height=1",
            params![[0xEEu8; 32].as_slice()],
        )
        .unwrap();
    }
    assert!(storage
        .audit_record_commitment_chain()
        .await
        .unwrap_err()
        .contains("stored row mismatch"));
}

#[tokio::test]
async fn test_checkpoint_refuses_same_height_tip_tampering_after_audit() {
    let (storage, _) = commitment_audit_fixture().await;
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "UPDATE record_commitment_blocks SET block_hash=?1 WHERE height=1",
            params![[0xE7u8; 32].as_slice()],
        )
        .unwrap();
    }
    assert!(storage
        .record_commitment_chain_checkpoint(1)
        .await
        .unwrap_err()
        .contains("audit baseline is stale"));
    // Public integrity evidence remains hash-free even though the private
    // runtime baseline retains the value needed for this fail-closed gate.
    let value = serde_json::to_value(storage.record_commitment_chain_integrity_status()).unwrap();
    assert!(value.get("verified_tip_hash").is_none());
    assert!(value.get("tip_hash").is_none());
}

#[tokio::test]
async fn test_commitment_chain_audit_rejects_membership_index_tampering() {
    let (storage, block) = commitment_audit_fixture().await;
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "DELETE FROM record_block_commitments WHERE record_id=?1",
            params![block.record_ids[0].as_slice()],
        )
        .unwrap();
    }
    assert!(storage
        .audit_record_commitment_chain()
        .await
        .unwrap_err()
        .contains("membership index mismatch"));
}

#[tokio::test]
async fn test_commitment_chain_audit_reverifies_persisted_signature() {
    let (storage, mut block) = commitment_audit_fixture().await;
    block.proposer_signature[0] ^= 0x01;
    let payload = bincode::serialize(&block).unwrap();
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "UPDATE record_commitment_blocks
                 SET proposer_signature=?1,payload=?2 WHERE height=1",
            params![block.proposer_signature.as_slice(), payload],
        )
        .unwrap();
    }
    assert!(storage
        .audit_record_commitment_chain()
        .await
        .unwrap_err()
        .contains("proposer signature is invalid"));
}

#[tokio::test]
async fn test_commitment_chain_append_rejects_sqlite_integer_overflow() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let identity = IdentityKeyPair::generate();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        u64::MAX,
        GENESIS_PREV_HASH,
        vec![[0x55; 32]],
        &identity,
    );
    assert!(storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap_err()
        .contains("timestamp exceeds SQLite range"));
}

#[tokio::test]
async fn test_commitment_chain_rejects_fork_and_rolls_back_duplicate() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let identity = IdentityKeyPair::generate();
    let first = RecordCommitmentBlockV1::new_signed(
        1,
        1_700_300_001,
        GENESIS_PREV_HASH,
        vec![[0x11; 32]],
        &identity,
    );
    storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();

    let fork = RecordCommitmentBlockV1::new_signed(
        1,
        1_700_300_002,
        GENESIS_PREV_HASH,
        vec![[0x22; 32]],
        &identity,
    );
    assert!(storage
        .append_record_commitment_block(&fork, None)
        .await
        .unwrap_err()
        .contains("fork"));

    let duplicate = RecordCommitmentBlockV1::new_signed(
        2,
        1_700_300_003,
        first.hash(),
        vec![[0x11; 32]],
        &identity,
    );
    assert!(storage
        .append_record_commitment_block(&duplicate, None)
        .await
        .is_err());
    assert_eq!(
        storage.record_commitment_chain_tip().await,
        (1, first.hash())
    );
    assert_eq!(
        storage
            .get_record_commitment_block_range(1, 10)
            .await
            .unwrap(),
        vec![first]
    );
}

#[test]
fn test_commitment_sync_runtime_tracks_failure_recovery_and_bounds_events() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let initial = storage.record_commitment_sync_status();
    assert_eq!(initial.role, "verifier");
    assert_eq!(initial.state, "disabled");
    assert!(!initial.enabled);
    assert_eq!(initial.last_trigger, "none");
    assert_eq!(initial.announcements_accepted_total, 0);
    assert_eq!(initial.certificate_policy_state, "not_applicable");
    assert!(!initial.certificate_policy_ready);

    storage.configure_record_commitment_sync(false, true);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .certificate_policy_state,
        "disabled"
    );
    storage.record_commitment_sync_attempt(100);
    storage.record_commitment_sync_failure(101, "request timeout https://private.example", 1, 131);
    let failed = storage.record_commitment_sync_status();
    assert_eq!(failed.role, "follower");
    assert_eq!(failed.state, "backoff");
    assert_eq!(failed.last_trigger, "scheduled");
    assert_eq!(failed.last_attempt_at, Some(100));
    assert_eq!(failed.last_failure_at, Some(101));
    assert_eq!(failed.next_poll_at, Some(131));
    assert_eq!(
        failed.last_error_code.as_deref(),
        Some("internal_sync_error")
    );
    assert_eq!(failed.recent_events.len(), 1);
    assert_eq!(failed.recent_events[0].kind, "failure");

    storage.record_commitment_sync_attempt(132);
    storage.record_commitment_sync_page_success(133, 3, 9, false);
    let awaiting_proof = storage.record_commitment_sync_status();
    assert_eq!(awaiting_proof.state, "checkpointing");
    assert_eq!(awaiting_proof.consecutive_failures, 1);
    storage.record_commitment_sync_checkpoint_success(134, 9);
    storage.schedule_next_commitment_sync_poll(163);
    let recovered = storage.record_commitment_sync_status();
    assert_eq!(recovered.state, "current");
    assert_eq!(recovered.last_success_at, Some(134));
    assert_eq!(recovered.last_recovered_at, Some(134));
    assert_eq!(recovered.remote_tip_height, Some(9));
    assert_eq!(recovered.pages_received_total, 1);
    assert_eq!(recovered.blocks_received_total, 3);
    assert_eq!(recovered.consecutive_failures, 0);
    assert_eq!(recovered.next_poll_at, Some(163));
    assert_eq!(recovered.failure_events_total, 1);
    assert_eq!(recovered.recovery_events_total, 1);
    assert_eq!(recovered.recent_events.len(), 2);
    assert_eq!(recovered.recent_events[1].kind, "recovered");

    // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] A threshold-certified
    // prefix recovered without the producer remains distinct from a live
    // equal-tip coordinator checkpoint.
    storage.record_commitment_sync_attempt(135);
    storage.record_commitment_sync_certified_recovery_success(136, 9);
    let certified = storage.record_commitment_sync_status();
    assert_eq!(certified.state, "certified_recovered");
    assert_eq!(certified.last_success_at, Some(136));
    assert_eq!(certified.remote_tip_height, Some(9));
    assert_eq!(certified.consecutive_failures, 0);

    storage.record_commitment_sync_announcement(
        140,
        12,
        RecordCommitmentAnnouncementDisposition::Accepted,
    );
    storage.record_commitment_sync_announcement(
        141,
        11,
        RecordCommitmentAnnouncementDisposition::Coalesced,
    );
    storage.record_commitment_sync_announcement(
        142,
        10,
        RecordCommitmentAnnouncementDisposition::Stale,
    );
    storage.record_commitment_sync_announcement(
        143,
        9,
        RecordCommitmentAnnouncementDisposition::Unavailable,
    );
    storage.record_commitment_sync_announcement_attempt(144);
    let announced = storage.record_commitment_sync_status();
    assert_eq!(announced.last_trigger, "block_announce");
    assert_eq!(announced.last_announcement_at, Some(143));
    assert_eq!(announced.last_announced_height, Some(12));
    assert_eq!(
        announced.last_announcement_result.as_deref(),
        Some("unavailable")
    );
    assert_eq!(announced.announcements_accepted_total, 1);
    assert_eq!(announced.announcements_coalesced_total, 1);
    assert_eq!(announced.announcements_stale_total, 1);
    assert_eq!(announced.announcements_unavailable_total, 1);

    for failure in 1..=20u32 {
        storage.record_commitment_sync_failure(
            200 + u64::from(failure),
            "request_timeout",
            failure,
            500 + u64::from(failure),
        );
    }
    let bounded = storage.record_commitment_sync_status();
    assert_eq!(bounded.recent_events.len(), COMMITMENT_SYNC_EVENT_CAPACITY);
    assert_eq!(bounded.recent_events.first().unwrap().sequence, 7);
    assert_eq!(bounded.recent_events.last().unwrap().sequence, 22);
    assert_eq!(bounded.failure_events_total, 21);

    storage.stop_record_commitment_sync();
    assert_eq!(storage.record_commitment_sync_status().state, "stopped");
}

#[test]
fn test_certificate_carrier_circuit_runtime_is_isolated_and_follower_only() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();

    storage.record_commitment_certificate_carrier_circuit_observation(2, 3, 1);
    let disabled = storage.record_commitment_sync_status();
    assert_eq!(disabled.certificate_carrier_cooling_slots, 0);
    assert_eq!(disabled.certificate_carrier_cooldown_skips_total, 0);
    assert_eq!(disabled.certificate_carrier_half_open_attempts_total, 0);

    // [CERTIFICATE-CARRIER-CIRCUIT 2026-07-29 by Codex] Certificate
    // scheduling aggregates are additive but remain independent from the
    // block-page circuit's gauge and counters.
    storage.configure_record_commitment_sync(false, true);
    storage.record_commitment_block_carrier_circuit_observation(1, 2, 1);
    storage.record_commitment_certificate_carrier_circuit_observation(2, 3, 1);
    storage.record_commitment_certificate_carrier_circuit_observation(1, 4, 2);
    let follower = storage.record_commitment_sync_status();
    assert_eq!(follower.block_carrier_cooling_slots, 1);
    assert_eq!(follower.block_carrier_cooldown_skips_total, 2);
    assert_eq!(follower.block_carrier_half_open_attempts_total, 1);
    assert_eq!(follower.certificate_carrier_cooling_slots, 1);
    assert_eq!(follower.certificate_carrier_cooldown_skips_total, 7);
    assert_eq!(follower.certificate_carrier_half_open_attempts_total, 3);

    storage.configure_record_commitment_sync(true, false);
    storage.record_commitment_certificate_carrier_circuit_observation(3, 8, 5);
    let coordinator = storage.record_commitment_sync_status();
    assert_eq!(coordinator.certificate_carrier_cooling_slots, 0);
    assert_eq!(coordinator.certificate_carrier_cooldown_skips_total, 0);
    assert_eq!(coordinator.certificate_carrier_half_open_attempts_total, 0);
}

#[test]
fn test_certificate_sync_runtime_is_aggregate_monotonic_and_follower_only() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();

    storage.record_commitment_certificate_sync_outcome(
        90,
        RecordCommitmentCertificateSyncDisposition::Coordinator,
        0,
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .certificate_sync_rounds_total,
        0
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_certificate_security_stop_at,
        None
    );
    storage.configure_record_commitment_sync(false, true);
    storage.record_commitment_certificate_sync_outcome(
        100,
        RecordCommitmentCertificateSyncDisposition::Coordinator,
        0,
    );
    storage.record_commitment_certificate_sync_outcome(
        110,
        RecordCommitmentCertificateSyncDisposition::CarrierRecovered,
        2,
    );
    storage.record_commitment_certificate_sync_outcome(
        105,
        RecordCommitmentCertificateSyncDisposition::AvailabilityExhausted,
        1,
    );
    storage.record_commitment_certificate_sync_outcome(
        115,
        RecordCommitmentCertificateSyncDisposition::VerifiedUnpersisted,
        2,
    );
    storage.record_commitment_certificate_sync_outcome(
        120,
        RecordCommitmentCertificateSyncDisposition::SecurityStopped,
        1,
    );
    storage.record_commitment_certificate_sync_outcome(
        130,
        RecordCommitmentCertificateSyncDisposition::Coordinator,
        0,
    );

    let status = storage.record_commitment_sync_status();
    assert_eq!(status.last_certificate_sync_at, Some(130));
    assert_eq!(
        status.last_certificate_sync_result.as_deref(),
        Some("coordinator")
    );
    assert_eq!(status.last_certificate_carrier_recovered_at, Some(110));
    assert_eq!(status.last_certificate_security_stop_at, Some(120));
    assert_eq!(status.certificate_sync_rounds_total, 6);
    assert_eq!(status.certificate_coordinator_success_total, 2);
    assert_eq!(status.certificate_carrier_attempts_total, 6);
    assert_eq!(status.certificate_carrier_recoveries_total, 1);
    assert_eq!(status.certificate_verified_unpersisted_total, 1);
    assert_eq!(status.certificate_availability_exhausted_total, 1);
    assert_eq!(status.certificate_security_stops_total, 1);
    assert_eq!(
        status.certificate_sync_rounds_total,
        status
            .certificate_coordinator_success_total
            .saturating_add(status.certificate_carrier_recoveries_total)
            .saturating_add(status.certificate_verified_unpersisted_total)
            .saturating_add(status.certificate_availability_exhausted_total)
            .saturating_add(status.certificate_security_stops_total)
    );

    storage.configure_record_commitment_sync(true, false);
    storage.record_commitment_certificate_sync_outcome(
        130,
        RecordCommitmentCertificateSyncDisposition::Coordinator,
        0,
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .certificate_sync_rounds_total,
        0
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_certificate_security_stop_at,
        None
    );
}

#[test]
fn test_certificate_backfill_runtime_is_atomic_monotonic_and_coordinator_only() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();

    storage.record_commitment_certificate_backfill_outcome(
        90,
        RecordCommitmentCertificateBackfillDisposition::Persisted,
        1,
        1,
        2,
        1,
    );
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .coordinator_certificate_backfill_rounds_total,
        0
    );

    // [CERTIFICATE-BACKFILL-TELEMETRY 2026-07-29 by Codex] Follower
    // retrieval and coordinator backfill are deliberately disjoint even
    // when they observe equivalent transport outcomes.
    storage.configure_record_commitment_sync(false, true);
    storage.record_commitment_certificate_backfill_outcome(
        95,
        RecordCommitmentCertificateBackfillDisposition::SecurityStopped,
        1,
        1,
        1,
        1,
    );
    let follower = storage.record_commitment_sync_status();
    assert_eq!(follower.coordinator_certificate_backfill_rounds_total, 0);
    assert_eq!(follower.certificate_sync_rounds_total, 0);

    storage.configure_record_commitment_sync(true, false);
    storage.record_commitment_certificate_backfill_outcome(
        100,
        RecordCommitmentCertificateBackfillDisposition::Persisted,
        1,
        0,
        0,
        0,
    );
    storage.record_commitment_certificate_backfill_outcome(
        110,
        RecordCommitmentCertificateBackfillDisposition::VerifiedUnpersisted,
        2,
        1,
        3,
        1,
    );
    storage.record_commitment_certificate_backfill_outcome(
        105,
        RecordCommitmentCertificateBackfillDisposition::AvailabilityExhausted,
        3,
        2,
        4,
        2,
    );
    storage.record_commitment_certificate_backfill_outcome(
        120,
        RecordCommitmentCertificateBackfillDisposition::SecurityStopped,
        1,
        1,
        5,
        3,
    );
    storage.record_commitment_certificate_backfill_outcome(
        130,
        RecordCommitmentCertificateBackfillDisposition::Persisted,
        0,
        0,
        0,
        0,
    );

    let status = storage.record_commitment_sync_status();
    assert_eq!(status.last_coordinator_certificate_backfill_at, Some(130));
    assert_eq!(
        status
            .last_coordinator_certificate_backfill_result
            .as_deref(),
        Some("persisted")
    );
    assert_eq!(
        status.last_coordinator_certificate_backfill_security_stop_at,
        Some(120)
    );
    assert_eq!(status.coordinator_certificate_backfill_rounds_total, 5);
    assert_eq!(status.coordinator_certificate_backfill_persisted_total, 2);
    assert_eq!(
        status.coordinator_certificate_backfill_verified_unpersisted_total,
        1
    );
    assert_eq!(
        status.coordinator_certificate_backfill_availability_exhausted_total,
        1
    );
    assert_eq!(
        status.coordinator_certificate_backfill_security_stops_total,
        1
    );
    assert_eq!(
        status.coordinator_certificate_backfill_carrier_attempts_total,
        7
    );
    assert_eq!(
        status.coordinator_certificate_backfill_carrier_cooling_slots,
        0
    );
    assert_eq!(
        status.coordinator_certificate_backfill_carrier_cooldown_skips_total,
        12
    );
    assert_eq!(
        status.coordinator_certificate_backfill_carrier_half_open_attempts_total,
        6
    );
    assert_eq!(
        status.coordinator_certificate_backfill_rounds_total,
        status
            .coordinator_certificate_backfill_persisted_total
            .saturating_add(status.coordinator_certificate_backfill_verified_unpersisted_total)
            .saturating_add(status.coordinator_certificate_backfill_availability_exhausted_total)
            .saturating_add(status.coordinator_certificate_backfill_security_stops_total)
    );
    assert_eq!(status.certificate_sync_rounds_total, 0);
    assert_eq!(status.certificate_carrier_attempts_total, 0);

    storage.configure_record_commitment_sync(false, true);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_coordinator_certificate_backfill_security_stop_at,
        None
    );
}

#[tokio::test]
async fn test_certificate_policy_readiness_requires_exact_local_validation() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let producer = IdentityKeyPair::generate();
    let producer_id = producer.public_key_bytes();
    let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x81, &producer);
    storage
        .append_record_commitment_block(&first, Some(&producer_id))
        .await
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    storage.configure_record_commitment_sync(false, true);

    storage.configure_record_commitment_certificate_policy(0, 1);
    storage.record_commitment_sync_checkpoint_success(90, 1);
    let disabled = storage.record_commitment_sync_status();
    assert_eq!(disabled.state, "current");
    assert_eq!(disabled.follower_readiness_state, "ready");
    assert!(disabled.follower_fully_ready);
    assert_eq!(disabled.certificate_policy_state, "disabled");
    assert!(!disabled.certificate_policy_ready);
    assert_eq!(disabled.certificate_witnesses_configured, 0);
    assert_eq!(disabled.certificate_minimum_signers, 1);

    storage.configure_record_commitment_certificate_policy(2, 2);
    let waiting = storage.record_commitment_sync_status();
    assert_eq!(waiting.state, "current");
    assert_eq!(waiting.follower_readiness_state, "waiting_for_certificate");
    assert!(!waiting.follower_fully_ready);
    assert_eq!(waiting.certificate_policy_state, "waiting_for_convergence");
    assert!(!waiting.certificate_policy_ready);
    assert_eq!(waiting.certificate_witnesses_configured, 2);
    assert_eq!(waiting.certificate_minimum_signers, 2);

    storage.record_commitment_certificate_policy_evaluation(
        100,
        RecordCommitmentCertificatePolicyReadiness::Ready { tip_height: 1 },
    );
    let ready = storage.record_commitment_sync_status();
    assert_eq!(ready.follower_readiness_state, "ready");
    assert!(ready.follower_fully_ready);
    assert_eq!(ready.certificate_policy_state, "ready");
    assert!(ready.certificate_policy_ready);
    assert_eq!(ready.certificate_policy_last_evaluated_at, Some(100));
    assert_eq!(ready.certificate_policy_evaluated_tip_height, Some(1));

    let second = signed_commitment_block(2, first.hash(), 0x82, &producer);
    storage
        .append_record_commitment_block(&second, Some(&producer_id))
        .await
        .unwrap();
    let stale = storage.record_commitment_sync_status();
    assert_eq!(stale.follower_readiness_state, "waiting_for_certificate");
    assert!(!stale.follower_fully_ready);
    assert_eq!(stale.certificate_policy_state, "waiting_for_certificate");
    assert!(!stale.certificate_policy_ready);
    assert_eq!(stale.certificate_policy_evaluated_tip_height, Some(1));

    storage.record_commitment_certificate_policy_evaluation(
        110,
        RecordCommitmentCertificatePolicyReadiness::Ready { tip_height: 2 },
    );
    let refreshed = storage.record_commitment_sync_status();
    assert_eq!(refreshed.follower_readiness_state, "ready");
    assert!(refreshed.follower_fully_ready);
    assert_eq!(refreshed.certificate_policy_state, "ready");
    assert!(refreshed.certificate_policy_ready);
    assert_eq!(refreshed.certificate_policy_evaluated_tip_height, Some(2));

    storage.record_commitment_sync_certified_recovery_success(111, 2);
    let certified_recovered = storage.record_commitment_sync_status();
    assert_eq!(certified_recovered.state, "certified_recovered");
    assert_eq!(
        certified_recovered.follower_readiness_state,
        "certified_recovered"
    );
    assert!(!certified_recovered.follower_fully_ready);
    storage.record_commitment_sync_checkpoint_success(112, 2);

    storage.record_commitment_certificate_policy_evaluation(
        90,
        RecordCommitmentCertificatePolicyReadiness::WaitingForCertificate { tip_height: 2 },
    );
    let invalidated = storage.record_commitment_sync_status();
    assert_eq!(
        invalidated.certificate_policy_state,
        "waiting_for_certificate"
    );
    assert!(!invalidated.certificate_policy_ready);
    assert_eq!(
        invalidated.certificate_policy_last_evaluated_at,
        Some(110),
        "wall-clock rollback must not regress the diagnostic timestamp"
    );

    storage.record_commitment_certificate_policy_evaluation(
        120,
        RecordCommitmentCertificatePolicyReadiness::SourceUnavailable { tip_height: 1 },
    );
    let stale_unavailable = storage.record_commitment_sync_status();
    assert_eq!(
        stale_unavailable.certificate_policy_state, "waiting_for_certificate",
        "a transport result for an older tip must not describe current policy state"
    );
    assert!(!stale_unavailable.certificate_policy_ready);
    assert_eq!(
        stale_unavailable.certificate_policy_evaluated_tip_height,
        Some(1)
    );
    assert_eq!(
        stale_unavailable.follower_readiness_state,
        "waiting_for_certificate"
    );
    assert!(!stale_unavailable.follower_fully_ready);

    storage.record_commitment_certificate_policy_evaluation(
        121,
        RecordCommitmentCertificatePolicyReadiness::SourceUnavailable { tip_height: 2 },
    );
    let unavailable = storage.record_commitment_sync_status();
    assert_eq!(unavailable.certificate_policy_state, "source_unavailable");
    assert_eq!(unavailable.follower_readiness_state, "source_unavailable");
    assert!(!unavailable.follower_fully_ready);

    storage.record_commitment_certificate_policy_evaluation(
        122,
        RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
    );
    let security_stopped = storage.record_commitment_sync_status();
    assert_eq!(
        security_stopped.certificate_policy_state,
        "security_stopped"
    );
    assert_eq!(
        security_stopped.follower_readiness_state,
        "security_stopped"
    );
    assert!(!security_stopped.follower_fully_ready);

    storage.configure_record_commitment_certificate_policy(1, 2);
    let invalid = storage.record_commitment_sync_status();
    assert_eq!(invalid.follower_readiness_state, "configuration_error");
    assert!(!invalid.follower_fully_ready);
    assert_eq!(invalid.certificate_policy_state, "configuration_error");
    assert!(!invalid.certificate_policy_ready);

    storage.configure_record_commitment_sync(true, false);
    storage.configure_record_commitment_certificate_policy(3, 2);
    let coordinator = storage.record_commitment_sync_status();
    assert_eq!(coordinator.follower_readiness_state, "not_applicable");
    assert!(!coordinator.follower_fully_ready);
    assert_eq!(coordinator.certificate_policy_state, "not_applicable");
    assert!(!coordinator.certificate_policy_ready);
    assert_eq!(coordinator.certificate_witnesses_configured, 0);
    assert_eq!(coordinator.certificate_minimum_signers, 0);
}

#[test]
fn test_commitment_sync_runtime_reports_coordinator_without_polling() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.configure_record_commitment_sync(true, false);
    storage.record_commitment_outbound_announcement(100, 9, 3, 2, 1, 0, 1, 1, 0);
    storage.record_commitment_outbound_announcement(101, 10, 3, 1, 1, 1, 2, 0, 1);
    storage.record_commitment_outbound_announcement_superseded(102);
    let superseded = storage.record_commitment_sync_status();
    assert_eq!(superseded.last_outbound_announced_height, None);
    assert_eq!(
        superseded.last_outbound_announcement_result.as_deref(),
        Some("superseded")
    );
    storage.record_commitment_outbound_announcement_skipped(103);
    let status = storage.record_commitment_sync_status();
    assert_eq!(status.role, "coordinator");
    assert_eq!(status.state, "producing");
    assert!(!status.enabled);
    assert!(status.recent_events.is_empty());
    assert_eq!(status.last_outbound_announcement_at, Some(103));
    assert_eq!(status.last_outbound_announced_height, None);
    assert_eq!(
        status.last_outbound_announcement_result.as_deref(),
        Some("skipped")
    );
    assert_eq!(status.outbound_announcement_rounds_total, 4);
    assert_eq!(status.outbound_announcement_rounds_skipped_total, 1);
    assert_eq!(status.outbound_announcement_rounds_superseded_total, 1);
    assert_eq!(status.outbound_announcements_attempted_total, 6);
    assert_eq!(status.outbound_announcements_accepted_total, 3);
    assert_eq!(status.outbound_announcements_stale_total, 2);
    assert_eq!(status.outbound_announcements_failed_total, 1);
    assert_eq!(status.outbound_announcement_retries_attempted_total, 3);
    assert_eq!(status.outbound_announcement_retries_succeeded_total, 1);
    assert_eq!(status.outbound_announcement_retries_exhausted_total, 1);
}

#[test]
fn test_commitment_outbound_announcement_results_are_fail_closed() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.configure_record_commitment_sync(true, false);

    storage.record_commitment_outbound_announcement(100, 9, 3, 3, 0, 0, 0, 0, 0);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_outbound_announcement_result
            .as_deref(),
        Some("all_woken")
    );
    storage.record_commitment_outbound_announcement(101, 10, 3, 2, 1, 0, 0, 0, 0);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_outbound_announcement_result
            .as_deref(),
        Some("delivered")
    );
    storage.record_commitment_outbound_announcement(102, 11, 3, 1, 1, 1, 0, 0, 0);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_outbound_announcement_result
            .as_deref(),
        Some("partial")
    );
    storage.record_commitment_outbound_announcement(103, 12, 3, 0, 0, 3, 2, 0, 1);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_outbound_announcement_result
            .as_deref(),
        Some("failed")
    );
    storage.record_commitment_outbound_announcement(104, 13, 0, 0, 0, 0, 0, 0, 0);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_outbound_announcement_result
            .as_deref(),
        Some("no_targets")
    );
    storage.record_commitment_outbound_announcement(105, 14, 3, 3, 1, 0, 0, 0, 0);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_outbound_announcement_result
            .as_deref(),
        Some("failed")
    );
    storage.record_commitment_outbound_announcement(106, 15, 3, 3, 0, 0, 1, 2, 0);
    assert_eq!(
        storage
            .record_commitment_sync_status()
            .last_outbound_announcement_result
            .as_deref(),
        Some("failed")
    );
}

#[test]
fn test_commitment_sync_error_codes_are_explicitly_allow_listed() {
    assert_eq!(
        privacy_safe_sync_error_code("request_timeout"),
        "request_timeout"
    );
    assert_eq!(
        privacy_safe_sync_error_code("http_status_503"),
        "http_status_error"
    );
    assert_eq!(
        privacy_safe_sync_error_code("checkpoint_http_status_503"),
        "checkpoint_http_status_error"
    );
    assert_eq!(
        privacy_safe_sync_error_code("signed_checkpoint_divergence"),
        "signed_checkpoint_divergence"
    );
    assert_eq!(
        privacy_safe_sync_error_code("unknown_but_well_formed"),
        "internal_sync_error"
    );
    assert_eq!(
        privacy_safe_sync_error_code("http_status_50x"),
        "internal_sync_error"
    );
}

#[test]
fn test_checkpoint_runtime_is_aggregate_and_tracks_proof_lifecycle() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let initial = storage.record_commitment_checkpoint_status();
    assert_eq!(initial.state, "not_checked");
    assert_eq!(initial.proofs_verified_total, 0);
    assert_eq!(initial.last_round_state, "not_checked");
    assert_eq!(initial.last_round_at, None);
    assert_eq!(initial.checkpoint_certificates, 0);
    assert_eq!(initial.latest_certified_height, None);

    storage.record_commitment_checkpoint_failure(100);
    storage.record_commitment_checkpoint_verified(110, "remote_ahead", 3, 4);
    storage.record_commitment_checkpoint_verified(120, "converged", 4, 4);
    storage.record_commitment_checkpoint_served(130);
    storage.record_commitment_checkpoint_witness_round(140, 4, 3, 2, 1, 1, 0, 1, 0);
    let status = storage.record_commitment_checkpoint_status();
    assert_eq!(status.state, "converged");
    assert_eq!(status.last_checked_at, Some(120));
    assert_eq!(status.last_converged_at, Some(120));
    assert_eq!(status.last_divergence_at, None);
    assert_eq!(status.last_failure_at, Some(100));
    assert_eq!(status.last_served_at, Some(130));
    assert_eq!(status.local_tip_height, Some(4));
    assert_eq!(status.remote_tip_height, Some(4));
    assert_eq!(status.proofs_verified_total, 2);
    assert_eq!(status.proofs_failed_total, 1);
    assert_eq!(status.divergences_total, 0);
    assert_eq!(status.requests_served_total, 1);
    assert_eq!(status.last_round_state, "partial");
    assert_eq!(status.last_round_at, Some(140));
    assert_eq!(status.last_round_eligible, 4);
    assert_eq!(status.last_round_attempted, 3);
    assert_eq!(status.last_round_verified, 2);
    assert_eq!(status.last_round_failed, 1);
    assert_eq!(status.last_round_converged, 1);
    assert_eq!(status.last_round_remote_ahead, 0);
    assert_eq!(status.last_round_remote_behind, 1);
    assert_eq!(status.last_round_diverged, 0);
    let value = serde_json::to_value(status).unwrap();
    for forbidden in [
        "peer",
        "block_hash",
        "tip_hash",
        "signature",
        "request_id",
        "evidence_digest",
        "certificate_digest",
        "responder",
        "witness_id",
        "endpoint",
        "owner",
        "payload",
    ] {
        assert!(value.get(forbidden).is_none(), "unexpected {forbidden}");
    }
}

#[test]
fn test_checkpoint_serving_cannot_create_outbound_evidence() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();

    storage.record_commitment_checkpoint_served(50);

    let status = storage.record_commitment_checkpoint_status();
    assert_eq!(status.state, "not_checked");
    assert_eq!(status.last_checked_at, None);
    assert_eq!(status.last_converged_at, None);
    assert_eq!(status.last_divergence_at, None);
    assert_eq!(status.last_failure_at, None);
    assert_eq!(status.last_served_at, Some(50));
    assert_eq!(status.local_tip_height, None);
    assert_eq!(status.remote_tip_height, None);
    assert_eq!(status.proofs_verified_total, 0);
    assert_eq!(status.proofs_failed_total, 0);
    assert_eq!(status.divergences_total, 0);
    assert_eq!(status.requests_served_total, 1);
    assert_eq!(status.observation_freshness, "unavailable");
    assert_eq!(status.observation_age_seconds, None);
    assert_eq!(
        status.freshness_window_seconds,
        CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS
    );
    assert_eq!(status.last_round_state, "not_checked");
    assert_eq!(status.last_round_at, None);
}

#[test]
fn test_checkpoint_witness_round_state_never_overclaims_consensus() {
    assert_eq!(checkpoint_witness_round_state(0, 0, 0, 0, 0), "unavailable");
    assert_eq!(checkpoint_witness_round_state(3, 0, 3, 0, 0), "unverified");
    assert_eq!(checkpoint_witness_round_state(3, 2, 1, 0, 0), "partial");
    assert_eq!(
        checkpoint_witness_round_state(3, 3, 0, 0, 0),
        "shared_prefix"
    );
    assert_eq!(checkpoint_witness_round_state(3, 3, 0, 1, 0), "attention");
    assert_eq!(checkpoint_witness_round_state(3, 3, 0, 0, 1), "attention");
}

#[test]
fn test_block_confirmation_state_never_overclaims_certificate_coverage() {
    assert_eq!(
        commitment_block_confirmation_state(false, 9, Some(9), 3, 2),
        ("not_verified", 0)
    );
    assert_eq!(
        commitment_block_confirmation_state(true, 0, None, 0, 0),
        ("empty", 0)
    );
    assert_eq!(
        commitment_block_confirmation_state(true, 3, None, 0, 0),
        ("uncertified", 3)
    );
    assert_eq!(
        commitment_block_confirmation_state(true, 3, Some(3), 1, 2),
        ("certificate_invalid", 0)
    );
    assert_eq!(
        commitment_block_confirmation_state(true, 3, Some(3), 2, 2),
        ("witness_certified", 0)
    );
    assert_eq!(
        commitment_block_confirmation_state(true, 5, Some(3), 2, 2),
        ("certificate_lagging", 2)
    );
    assert_eq!(
        commitment_block_confirmation_state(true, 3, Some(4), 2, 2),
        ("certificate_ahead", 0)
    );
}

#[test]
fn test_checkpoint_observation_freshness_is_age_bounded() {
    assert_eq!(
        checkpoint_observation_freshness(None, 1_000),
        ("unavailable", None)
    );
    assert_eq!(
        checkpoint_observation_freshness(Some(1_001), 1_000),
        ("unavailable", None)
    );
    assert_eq!(
        checkpoint_observation_freshness(Some(1_000), 1_000),
        ("fresh", Some(0))
    );
    assert_eq!(
        checkpoint_observation_freshness(
            Some(1_000),
            1_000 + CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
        ),
        ("fresh", Some(CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS))
    );
    assert_eq!(
        checkpoint_observation_freshness(
            Some(1_000),
            1_001 + CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
        ),
        ("stale", Some(CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS + 1))
    );
}

#[tokio::test]
async fn test_checkpoint_certificate_requires_two_distinct_pinned_witnesses_and_reopens() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-certificate.db");
    let storage = MemoryStorage::open(&path, None).unwrap();
    let (witnesses, digests) = seed_checkpoint_certificate(&storage, 1_700_500_000).await;

    let uncertified = storage.record_commitment_checkpoint_status();
    assert_eq!(uncertified.block_confirmation_state, "uncertified");
    assert_eq!(uncertified.uncertified_block_count, 1);

    assert!(storage
        .persist_record_commitment_checkpoint_certificate(1_700_500_010, 2, &witnesses, &digests,)
        .await
        .unwrap());
    let audit = storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(audit.checkpoint_certificates, 1);
    assert_eq!(audit.latest_certified_height, Some(1));
    assert_eq!(audit.latest_certificate_signers, 2);
    assert_eq!(audit.latest_certificate_required_signers, 2);
    let status = storage.record_commitment_checkpoint_status();
    assert_eq!(status.checkpoint_certificates, 1);
    assert_eq!(status.latest_certified_height, Some(1));
    assert_eq!(status.latest_certificate_signers, 2);
    assert_eq!(status.block_confirmation_state, "witness_certified");
    assert_eq!(status.uncertified_block_count, 0);
    let (_, tip_hash, tip_height, _) = storage
        .record_commitment_chain_checkpoint(u64::MAX)
        .await
        .unwrap();
    let bundle = storage
        .record_commitment_checkpoint_certificate_bundle(tip_height, &tip_hash)
        .await
        .unwrap()
        .expect("matching audited certificate bundle");
    assert_eq!(bundle.checkpoint_height, 1);
    assert_eq!(bundle.checkpoint_hash, tip_hash);
    assert_eq!(bundle.required_signers, 2);
    assert_eq!(bundle.member_frames.len(), 2);
    assert!(storage
        .record_commitment_checkpoint_certificate_bundle(tip_height, &[0xA5; 32])
        .await
        .unwrap()
        .is_none());

    let third_witness = IdentityKeyPair::generate();
    let third_frame = signed_checkpoint_response_frame(
        &third_witness,
        [0xCF; 16],
        1_700_500_020,
        tip_height,
        tip_hash,
        tip_height,
        tip_hash,
    );
    let third_digest: [u8; 32] = Sha256::digest(&third_frame).into();
    storage
        .persist_record_commitment_checkpoint_evidence_with_witness_policy(
            1_700_500_020,
            "converged",
            tip_height,
            tip_height,
            tip_height,
            &third_digest,
            &third_frame,
            true,
        )
        .await
        .unwrap();
    let three_witnesses = [witnesses[0], witnesses[1], third_witness.public_key_bytes()];
    let three_digests = [digests[0], digests[1], third_digest];
    assert!(!storage
        .persist_record_commitment_checkpoint_certificate(
            1_700_500_021,
            3,
            &three_witnesses,
            &three_digests,
        )
        .await
        .unwrap());
    let rotated_witnesses = [witnesses[0], third_witness.public_key_bytes()];
    let rotated_digests = [digests[0], third_digest];
    assert!(!storage
        .persist_record_commitment_checkpoint_certificate(
            1_700_500_022,
            2,
            &rotated_witnesses,
            &rotated_digests,
        )
        .await
        .unwrap());
    let policy_audit = storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(policy_audit.latest_certificate_required_signers, 2);
    assert_eq!(policy_audit.latest_certificate_signers, 2);

    let proposer = IdentityKeyPair::generate();
    let next_block = signed_commitment_block(2, tip_hash, 0xB2, &proposer);
    storage
        .append_record_commitment_block(&next_block, None)
        .await
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let lagging = storage.record_commitment_checkpoint_status();
    assert_eq!(lagging.block_confirmation_state, "certificate_lagging");
    assert_eq!(lagging.uncertified_block_count, 1);
    drop(storage);

    let reopened = MemoryStorage::open(&path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    let reopened_audit = reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(reopened_audit.checkpoint_certificates, 1);
    assert_eq!(reopened_audit.latest_certified_height, Some(1));
    assert_eq!(reopened_audit.latest_certificate_signers, 2);
    let reopened_status = reopened.record_commitment_checkpoint_status();
    assert_eq!(
        reopened_status.block_confirmation_state,
        "certificate_lagging"
    );
    assert_eq!(reopened_status.uncertified_block_count, 1);
}

#[tokio::test]
async fn test_checkpoint_certificate_rejects_duplicate_witness_frames() {
    let (storage, block) = commitment_audit_fixture().await;
    let witness = IdentityKeyPair::generate();
    let witness_id = witness.public_key_bytes();
    let observed_at = 1_700_500_030u64;
    let mut digests = [[0u8; 32]; 2];
    for (index, digest) in digests.iter_mut().enumerate() {
        let frame = signed_checkpoint_response_frame(
            &witness,
            [0xD0u8.saturating_add(index as u8); 16],
            observed_at.saturating_add(index as u64),
            1,
            block.hash(),
            1,
            block.hash(),
        );
        *digest = Sha256::digest(&frame).into();
        storage
            .persist_record_commitment_checkpoint_evidence(
                observed_at.saturating_add(index as u64),
                "converged",
                1,
                1,
                1,
                digest,
                &frame,
            )
            .await
            .unwrap();
    }

    assert!(!storage
        .persist_record_commitment_checkpoint_certificate(
            observed_at.saturating_add(10),
            2,
            &[witness_id, witness_id],
            &digests,
        )
        .await
        .unwrap());
    assert_eq!(
        storage
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap()
            .checkpoint_certificates,
        0
    );
}

#[tokio::test]
async fn test_checkpoint_certificate_audit_rejects_member_and_digest_tampering() {
    let member_tamper = MemoryStorage::open(":memory:", None).unwrap();
    let (witnesses, digests) = seed_checkpoint_certificate(&member_tamper, 1_700_500_050).await;
    member_tamper
        .persist_record_commitment_checkpoint_certificate(1_700_500_060, 2, &witnesses, &digests)
        .await
        .unwrap();
    {
        let conn = member_tamper.conn_lock().await;
        conn.execute(
            "UPDATE record_checkpoint_certificate_members SET responder=?1
                 WHERE evidence_digest=?2",
            params![[0xEEu8; 32].as_slice(), digests[0].as_slice()],
        )
        .unwrap();
    }
    assert!(member_tamper
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap_err()
        .contains("member claim is invalid"));

    let digest_tamper = MemoryStorage::open(":memory:", None).unwrap();
    let (witnesses, digests) = seed_checkpoint_certificate(&digest_tamper, 1_700_500_070).await;
    digest_tamper
        .persist_record_commitment_checkpoint_certificate(1_700_500_080, 2, &witnesses, &digests)
        .await
        .unwrap();
    {
        let conn = digest_tamper.conn_lock().await;
        conn.execute(
            "UPDATE record_checkpoint_certificates SET certificate_digest=?1",
            params![[0xEFu8; 32].as_slice()],
        )
        .unwrap();
    }
    assert!(digest_tamper
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap_err()
        .contains("certificate digest mismatch"));
}

#[tokio::test]
async fn test_checkpoint_evidence_persists_and_reaudits_signed_frame() {
    let (storage, block) = commitment_audit_fixture().await;
    let responder = IdentityKeyPair::generate();
    let observed_at = 1_700_500_001;
    let frame = signed_checkpoint_response_frame(
        &responder,
        [0x31; 16],
        observed_at,
        1,
        block.hash(),
        1,
        block.hash(),
    );
    let digest: [u8; 32] = Sha256::digest(&frame).into();
    storage
        .persist_record_commitment_checkpoint_evidence(
            observed_at,
            "converged",
            1,
            1,
            1,
            &digest,
            &frame,
        )
        .await
        .unwrap();

    let audit = storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(audit.evidence_records, 1);
    assert_eq!(audit.applicable_evidence_records, 1);
    assert_eq!(audit.deferred_evidence_records, 0);
    assert_eq!(audit.divergence_evidence_records, 0);
    assert_eq!(audit.last_evidence_at, Some(observed_at));
    let status = storage.record_commitment_checkpoint_status();
    assert_eq!(status.evidence_state, "verified");
    assert_eq!(status.evidence_records, 1);
    assert_eq!(status.applicable_evidence_records, 1);
    assert_eq!(status.deferred_evidence_records, 0);
    assert_eq!(status.last_evidence_at, Some(observed_at));
}

#[tokio::test]
async fn test_checkpoint_evidence_audit_fails_closed_after_sqlite_tampering() {
    let (storage, block) = commitment_audit_fixture().await;
    let responder = IdentityKeyPair::generate();
    let observed_at = 1_700_500_010;
    let frame = signed_checkpoint_response_frame(
        &responder,
        [0x32; 16],
        observed_at,
        1,
        block.hash(),
        1,
        block.hash(),
    );
    let digest: [u8; 32] = Sha256::digest(&frame).into();
    storage
        .persist_record_commitment_checkpoint_evidence(
            observed_at,
            "converged",
            1,
            1,
            1,
            &digest,
            &frame,
        )
        .await
        .unwrap();
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "UPDATE record_checkpoint_evidence SET signed_response=?1",
            params![vec![MEMCHAIN_MAGIC, 0x00]],
        )
        .unwrap();
    }
    assert!(storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap_err()
        .contains("digest mismatch"));
    assert_eq!(
        storage.record_commitment_checkpoint_status().evidence_state,
        "invalid"
    );
}

#[tokio::test]
async fn test_checkpoint_evidence_duplicate_conflict_fails_closed() {
    let (storage, block) = commitment_audit_fixture().await;
    let responder = IdentityKeyPair::generate();
    let observed_at = 1_700_500_020;
    let frame = signed_checkpoint_response_frame(
        &responder,
        [0x33; 16],
        observed_at,
        1,
        block.hash(),
        1,
        block.hash(),
    );
    let digest: [u8; 32] = Sha256::digest(&frame).into();
    storage
        .persist_record_commitment_checkpoint_evidence(
            observed_at,
            "converged",
            1,
            1,
            1,
            &digest,
            &frame,
        )
        .await
        .unwrap();
    {
        let conn = storage.conn_lock().await;
        conn.execute(
            "UPDATE record_checkpoint_evidence SET relation='remote_ahead'",
            [],
        )
        .unwrap();
    }
    assert!(storage
        .persist_record_commitment_checkpoint_evidence(
            observed_at,
            "converged",
            1,
            1,
            1,
            &digest,
            &frame,
        )
        .await
        .unwrap_err()
        .contains("conflicts with verified frame"));
    assert_eq!(
        storage
            .record_commitment_checkpoint_status()
            .evidence_persistence_failures_total,
        1
    );
}

#[tokio::test]
async fn test_checkpoint_evidence_is_visible_through_wal_and_survives_restart() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-restart.db");
    let writer = MemoryStorage::open(&path, None).unwrap();
    let observed_at = 1_700_500_100;
    let (block, _, _) = seed_checkpoint_evidence(&writer, observed_at).await;

    // Open a second storage while the writer connection remains alive. This
    // proves the committed proof is visible through SQLite WAL, not merely
    // flushed as a side effect of closing the first process.
    let wal_reader = MemoryStorage::open(&path, None).unwrap();
    let chain = wal_reader.audit_record_commitment_chain().await.unwrap();
    assert_eq!(chain.block_count, 1);
    assert_eq!(chain.tip_height, 1);
    assert_eq!(
        wal_reader.record_commitment_chain_tip().await.1,
        block.hash()
    );
    let wal_audit = wal_reader
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(wal_audit.evidence_records, 1);
    assert_eq!(wal_audit.applicable_evidence_records, 1);
    assert_eq!(wal_audit.deferred_evidence_records, 0);
    assert_eq!(wal_audit.last_evidence_at, Some(observed_at));
    drop(wal_reader);
    drop(writer);

    // A later clean process must independently re-establish both audit
    // baselines instead of trusting runtime state from the prior process.
    let reopened = MemoryStorage::open(&path, None).unwrap();
    assert_eq!(
        reopened.record_commitment_chain_integrity_status().state,
        "not_verified"
    );
    assert_eq!(
        reopened
            .record_commitment_checkpoint_status()
            .evidence_state,
        "not_audited"
    );
    reopened.audit_record_commitment_chain().await.unwrap();
    let reopened_audit = reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(reopened_audit.evidence_records, 1);
    assert_eq!(
        reopened
            .record_commitment_checkpoint_status()
            .evidence_state,
        "verified"
    );
}

#[tokio::test]
async fn test_checkpoint_evidence_is_deferred_during_follower_rollback_recovery() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-follower-rollback.db");
    let original = MemoryStorage::open(&path, None).unwrap();
    let observed_at = 1_700_500_150;
    let (block, _, _) = seed_checkpoint_evidence(&original, observed_at).await;
    drop(original);

    // Model a follower restoring an older local chain snapshot while its
    // append-only evidence vault survives. Records and evidence remain;
    // only the derived commitment chain is reset for canonical resync.
    let rollback = rusqlite::Connection::open(&path).unwrap();
    rollback
        .execute_batch(
            "BEGIN IMMEDIATE;
                 DELETE FROM record_block_commitments;
                 DELETE FROM record_commitment_blocks;
                 DELETE FROM chain_state WHERE key IN (
                    'record_block_chain_id',
                    'record_block_tip_hash',
                    'record_block_tip_height'
                 );
                 COMMIT;",
        )
        .unwrap();
    drop(rollback);

    let recovered = MemoryStorage::open(&path, None).unwrap();
    let chain = recovered.audit_record_commitment_chain().await.unwrap();
    assert_eq!(chain.block_count, 0);
    assert_eq!(chain.tip_height, 0);

    let deferred = recovered
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(deferred.evidence_records, 1);
    assert_eq!(deferred.applicable_evidence_records, 0);
    assert_eq!(deferred.deferred_evidence_records, 1);
    assert_eq!(deferred.divergence_evidence_records, 0);
    assert_eq!(deferred.last_evidence_at, None);
    let deferred_status = recovered.record_commitment_checkpoint_status();
    assert_eq!(deferred_status.evidence_state, "verified");
    assert_eq!(deferred_status.applicable_evidence_records, 0);
    assert_eq!(deferred_status.deferred_evidence_records, 1);
    assert_eq!(deferred_status.observation_freshness, "unavailable");
    assert_eq!(deferred_status.observation_age_seconds, None);

    // Once the canonical prefix is restored, the same immutable frame can
    // be reclassified only after its historical hash relation is verified.
    recovered
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    recovered.audit_record_commitment_chain().await.unwrap();
    let applicable = recovered
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(applicable.evidence_records, 1);
    assert_eq!(applicable.applicable_evidence_records, 1);
    assert_eq!(applicable.deferred_evidence_records, 0);
    assert_eq!(applicable.last_evidence_at, Some(observed_at));
}

#[tokio::test]
async fn test_v8_to_current_file_migration_preserves_commitment_chain() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-migration.db");
    let original = MemoryStorage::open(&path, None).unwrap();
    let (block, _, _) = seed_checkpoint_evidence(&original, 1_700_500_200).await;
    drop(original);

    // Reconstruct the exact relevant v8 boundary: the commitment chain is
    // present, but the v9 evidence table does not yet exist.
    let legacy = rusqlite::Connection::open(&path).unwrap();
    legacy
        .execute_batch(
            "DROP TABLE record_checkpoint_evidence;
                 UPDATE schema_version SET version=8;",
        )
        .unwrap();
    drop(legacy);

    let migrated = MemoryStorage::open(&path, None).unwrap();
    let (version, handover_table_exists): (u32, i64) = {
        let conn = migrated.conn_lock().await;
        (
            conn.query_row("SELECT version FROM schema_version", [], |row| row.get(0))
                .unwrap(),
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_coordinator_handovers'",
                [],
                |row| row.get(0),
            )
            .unwrap(),
        )
    };
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(handover_table_exists, 1);
    let chain = migrated.audit_record_commitment_chain().await.unwrap();
    assert_eq!(chain.block_count, 1);
    assert_eq!(migrated.record_commitment_chain_tip().await.1, block.hash());
    let evidence = migrated
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(evidence.evidence_records, 0);
    assert_eq!(evidence.equivocation_incidents, 0);
}

#[tokio::test]
async fn test_v9_to_v10_file_migration_preserves_checkpoint_evidence() {
    let directory = TempDir::new().unwrap();
    let path = directory
        .path()
        .join("checkpoint-equivocation-migration.db");
    let original = MemoryStorage::open(&path, None).unwrap();
    seed_checkpoint_evidence(&original, 1_700_500_250).await;
    drop(original);

    let legacy = rusqlite::Connection::open(&path).unwrap();
    legacy
        .execute_batch(
            "DROP TABLE record_checkpoint_equivocations;
                 UPDATE schema_version SET version=9;",
        )
        .unwrap();
    drop(legacy);

    let migrated = MemoryStorage::open(&path, None).unwrap();
    let (version, incident_table_exists): (u32, i64) = {
        let conn = migrated.conn_lock().await;
        (
            conn.query_row("SELECT version FROM schema_version", [], |row| row.get(0))
                .unwrap(),
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_equivocations'",
                [],
                |row| row.get(0),
            )
            .unwrap(),
        )
    };
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(incident_table_exists, 1);
    migrated.audit_record_commitment_chain().await.unwrap();
    let evidence = migrated
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(evidence.evidence_records, 1);
    assert_eq!(evidence.equivocation_incidents, 0);
    assert_eq!(evidence.trusted_divergence_incidents, 0);
}

#[tokio::test]
async fn test_v10_to_v11_file_migration_preserves_checkpoint_evidence() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-divergence-migration.db");
    let original = MemoryStorage::open(&path, None).unwrap();
    seed_checkpoint_evidence(&original, 1_700_500_275).await;
    drop(original);

    let legacy = rusqlite::Connection::open(&path).unwrap();
    legacy
        .execute_batch(
            "DROP TABLE record_checkpoint_trusted_divergences;
                 UPDATE schema_version SET version=10;",
        )
        .unwrap();
    drop(legacy);

    let migrated = MemoryStorage::open(&path, None).unwrap();
    let (version, incident_table_exists): (u32, i64) = {
        let conn = migrated.conn_lock().await;
        (
            conn.query_row("SELECT version FROM schema_version", [], |row| row.get(0))
                .unwrap(),
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_trusted_divergences'",
                [],
                |row| row.get(0),
            )
            .unwrap(),
        )
    };
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(incident_table_exists, 1);
    migrated.audit_record_commitment_chain().await.unwrap();
    let evidence = migrated
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(evidence.evidence_records, 1);
    assert_eq!(evidence.trusted_divergence_incidents, 0);
}

#[tokio::test]
async fn test_v11_to_v12_file_migration_adds_certificate_tables_without_data_loss() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-certificate-migration.db");
    let original = MemoryStorage::open(&path, None).unwrap();
    seed_checkpoint_evidence(&original, 1_700_500_290).await;
    drop(original);

    let legacy = rusqlite::Connection::open(&path).unwrap();
    legacy
        .execute_batch(
            "DROP TABLE record_checkpoint_certificate_members;
                 DROP TABLE record_checkpoint_certificates;
                 UPDATE schema_version SET version=11;",
        )
        .unwrap();
    drop(legacy);

    let migrated = MemoryStorage::open(&path, None).unwrap();
    let (version, certificates, members): (u32, i64, i64) = {
        let conn = migrated.conn_lock().await;
        (
            conn.query_row("SELECT version FROM schema_version", [], |row| row.get(0))
                .unwrap(),
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_certificates'",
                [],
                |row| row.get(0),
            )
            .unwrap(),
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_certificate_members'",
                [],
                |row| row.get(0),
            )
            .unwrap(),
        )
    };
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(certificates, 1);
    assert_eq!(members, 1);
    migrated.audit_record_commitment_chain().await.unwrap();
    let evidence = migrated
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(evidence.evidence_records, 1);
    assert_eq!(evidence.checkpoint_certificates, 0);
}

#[tokio::test]
async fn test_reopened_checkpoint_evidence_rejects_disk_tampering() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-tamper.db");
    let original = MemoryStorage::open(&path, None).unwrap();
    seed_checkpoint_evidence(&original, 1_700_500_300).await;
    drop(original);

    let tamper = rusqlite::Connection::open(&path).unwrap();
    tamper
        .execute(
            "UPDATE record_checkpoint_evidence SET signed_response=?1",
            params![vec![MEMCHAIN_MAGIC, 0x00]],
        )
        .unwrap();
    drop(tamper);

    let reopened = MemoryStorage::open(&path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    assert!(reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap_err()
        .contains("digest mismatch"));
    assert_eq!(
        reopened
            .record_commitment_checkpoint_status()
            .evidence_state,
        "invalid"
    );
}

#[tokio::test]
async fn test_checkpoint_evidence_rejects_invalid_input_and_tracks_failure() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let frame = vec![MEMCHAIN_MAGIC, 0x01];
    let wrong_digest = [0u8; 32];
    assert!(storage
        .persist_record_commitment_checkpoint_evidence(
            1,
            "converged",
            0,
            0,
            0,
            &wrong_digest,
            &frame,
        )
        .await
        .unwrap_err()
        .contains("digest mismatch"));
    let digest: [u8; 32] = Sha256::digest(&frame).into();
    assert!(storage
        .persist_record_commitment_checkpoint_evidence(1, "unknown", 0, 0, 0, &digest, &frame,)
        .await
        .unwrap_err()
        .contains("relation is invalid"));
    assert!(storage
        .persist_record_commitment_checkpoint_evidence(1, "converged", 0, 0, 0, &digest, &frame,)
        .await
        .unwrap_err()
        .contains("frame decode failed"));
    let retained_rows: i64 = {
        let conn = storage.conn_lock().await;
        conn.query_row(
            "SELECT COUNT(*) FROM record_checkpoint_evidence",
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    assert_eq!(retained_rows, 0);
    assert_eq!(
        storage
            .record_commitment_checkpoint_status()
            .evidence_persistence_failures_total,
        3
    );
}

#[tokio::test]
async fn test_checkpoint_evidence_capacity_preserves_divergence() {
    let (storage, block) = commitment_audit_fixture().await;
    let responder = IdentityKeyPair::generate();
    let diverged_at = 1_700_510_000;
    let fork_hash = [0xE7; 32];
    let divergent = signed_checkpoint_response_frame(
        &responder,
        [0x40; 16],
        diverged_at,
        1,
        fork_hash,
        1,
        fork_hash,
    );
    let divergent_digest: [u8; 32] = Sha256::digest(&divergent).into();
    storage
        .persist_record_commitment_checkpoint_evidence(
            diverged_at,
            "diverged",
            1,
            1,
            1,
            &divergent_digest,
            &divergent,
        )
        .await
        .unwrap();

    for index in 0..CHECKPOINT_EVIDENCE_CAPACITY + 8 {
        let observed_at = diverged_at + 1 + index as u64;
        let mut request_id = [0u8; 16];
        request_id[..8].copy_from_slice(&(index as u64).to_le_bytes());
        let frame = signed_checkpoint_response_frame(
            &responder,
            request_id,
            observed_at,
            1,
            block.hash(),
            1,
            block.hash(),
        );
        let digest: [u8; 32] = Sha256::digest(&frame).into();
        storage
            .persist_record_commitment_checkpoint_evidence(
                observed_at,
                "converged",
                1,
                1,
                1,
                &digest,
                &frame,
            )
            .await
            .unwrap();
    }
    let audit = storage
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(audit.evidence_records, CHECKPOINT_EVIDENCE_CAPACITY as u64);
    assert_eq!(audit.divergence_evidence_records, 1);
}

#[tokio::test]
async fn test_trusted_witness_equivocation_is_durable_and_sticky() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("checkpoint-equivocation.db");
    let storage = MemoryStorage::open(&path, None).unwrap();
    let proposer = IdentityKeyPair::generate();
    let block = signed_commitment_block(1, GENESIS_PREV_HASH, 0xC1, &proposer);
    storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();

    let witness = IdentityKeyPair::generate();
    let first_at = 1_700_520_000;
    let first = signed_checkpoint_response_frame(
        &witness,
        [0x51; 16],
        first_at,
        1,
        block.hash(),
        1,
        block.hash(),
    );
    let first_digest: [u8; 32] = Sha256::digest(&first).into();
    assert_eq!(
        storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                first_at,
                "converged",
                1,
                1,
                1,
                &first_digest,
                &first,
                true,
            )
            .await
            .unwrap(),
        RecordCommitmentCheckpointEvidencePersistOutcome::Stored
    );

    let fork_hash = [0xE1; 32];
    let second_at = first_at + 1;
    let second = signed_checkpoint_response_frame(
        &witness, [0x52; 16], second_at, 1, fork_hash, 1, fork_hash,
    );
    let second_digest: [u8; 32] = Sha256::digest(&second).into();
    assert_eq!(
        storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                second_at,
                "diverged",
                1,
                1,
                1,
                &second_digest,
                &second,
                true,
            )
            .await
            .unwrap(),
        RecordCommitmentCheckpointEvidencePersistOutcome::EquivocationDetected
    );

    // A later honest-looking response cannot overwrite or erase the
    // already established conflict at the same signed height.
    let third_at = second_at + 1;
    let third = signed_checkpoint_response_frame(
        &witness,
        [0x53; 16],
        third_at,
        1,
        block.hash(),
        1,
        block.hash(),
    );
    let third_digest: [u8; 32] = Sha256::digest(&third).into();
    assert_eq!(
        storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                third_at,
                "converged",
                1,
                1,
                1,
                &third_digest,
                &third,
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
    assert_eq!(audit.evidence_records, 3);
    assert_eq!(audit.equivocation_incidents, 1);
    assert_eq!(audit.trusted_divergence_incidents, 1);
    assert!(storage.record_commitment_production_halted());
    assert_eq!(
        storage
            .count_record_commitment_checkpoint_equivocations_for_witnesses(&[
                witness.public_key_bytes(),
            ])
            .await
            .unwrap(),
        1
    );
    drop(storage);

    let reopened = MemoryStorage::open(&path, None).unwrap();
    reopened.audit_record_commitment_chain().await.unwrap();
    let reopened_audit = reopened
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    assert_eq!(reopened_audit.equivocation_incidents, 1);
    assert_eq!(reopened_audit.trusted_divergence_incidents, 1);
    assert_eq!(
        reopened
            .record_commitment_checkpoint_status()
            .equivocation_incidents,
        1
    );
}
