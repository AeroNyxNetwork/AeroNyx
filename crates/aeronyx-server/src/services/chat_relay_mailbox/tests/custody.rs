// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn create_consumes_ticket_once_and_exact_retry_survives_restart() {
    let context = TestContext::new();
    let mailbox = [0x11; 32];
    let request = context.lease(mailbox, [0x21; 16], 2, 32, NOW + 1_000);
    let store = context.open();
    assert!(matches!(
        store.create(&request, NOW).unwrap(),
        AnonymousMailboxCreateOutcome::Created(_)
    ));
    assert!(matches!(
        store.create(&request, NOW + 1).unwrap(),
        AnonymousMailboxCreateOutcome::Existing(_)
    ));
    drop(store);
    let reopened = context.open();
    assert!(matches!(
        reopened.create(&request, NOW + 301).unwrap(),
        AnonymousMailboxCreateOutcome::Existing(_)
    ));

    let conflicting = context.lease([0x12; 32], [0x21; 16], 2, 32, NOW + 1_000);
    assert_eq!(
        reopened.create(&conflicting, NOW + 301).unwrap(),
        AnonymousMailboxCreateOutcome::Conflict
    );
}

#[test]
fn expired_issued_ticket_is_not_served_and_cleanup_is_bounded() {
    let context = TestContext::new();
    let request = context.ticket_issue([0xa1; 16], [0xa2; 16], [0xa3; 32], NOW + 1_000, NOW + 10);
    let store = context.open_with_ticket_issuer();
    assert!(matches!(
        store.issue_ticket(&request, NOW).unwrap(),
        AnonymousMailboxTicketIssueOutcome::Issued(_)
    ));
    assert!(matches!(
        store.issue_ticket(&request, NOW + 11),
        Err(AnonymousMailboxStoreError::Rejected)
    ));
    let report = store.cleanup(NOW + 11).unwrap();
    assert_eq!(report.issued_tickets_removed, 1);
}

#[test]
fn invalid_ticket_issue_proof_is_rejected_without_persistence() {
    let context = TestContext::new();
    let valid = context.ticket_issue([0xb1; 16], [0xb2; 16], [0xb3; 32], NOW + 1_000, NOW + 300);
    let invalid = (valid.proof_nonce.saturating_add(1)..u64::MAX)
        .find_map(|nonce| {
            let candidate = AnonymousMailboxTicketIssueV1::new(
                valid.request_id,
                valid.ticket_id,
                valid.target_node_id,
                valid.lease_claims_commitment,
                valid.issued_at,
                valid.expires_at,
                nonce,
            )
            .unwrap();
            (candidate.proof_digest().unwrap()[0] & 0x80 != 0).then_some(candidate)
        })
        .expect("invalid one-bit proof");
    let store = context.open_with_ticket_issuer();
    let before_changes: i64 = store
        .connection
        .lock()
        .query_row("SELECT total_changes()", [], |row| row.get(0))
        .expect("pre-validation changes");
    assert!(matches!(
        store.issue_ticket(&invalid, NOW),
        Ok(AnonymousMailboxTicketIssueOutcome::PreWriteRejected)
    ));
    let connection = store.connection.lock();
    let after_changes: i64 = connection
        .query_row("SELECT total_changes()", [], |row| row.get(0))
        .expect("post-validation changes");
    assert_eq!(before_changes, after_changes, "zero SQL writes");
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM anonymous_mailbox_issued_tickets",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
}

#[test]
fn v1_store_migrates_additively_and_reopens_with_ticket_journal() {
    let context = TestContext::new();
    drop(context.open());
    let connection = Connection::open(&context.config.db_path).unwrap();
    connection
        .execute_batch(
            "DROP INDEX anonymous_mailbox_pull_replay_expiry;
                 DROP TABLE anonymous_mailbox_pull_replays;
                 DROP INDEX anonymous_mailbox_issued_ticket_expiry;
                 DROP TABLE anonymous_mailbox_issued_tickets;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_rows;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_bytes;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN outstanding_tickets;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN issuance_window_started_at;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN issues_in_window;
                 UPDATE anonymous_mailbox_meta SET schema_version = 1;
                 PRAGMA user_version = 1;",
        )
        .unwrap();
    drop(connection);

    let reopened = context.open_with_ticket_issuer();
    let connection = reopened.connection.lock();
    assert_eq!(schema_user_version(&connection), 4);
    assert_eq!(
        connection
            .query_row(
                "SELECT schema_version FROM anonymous_mailbox_meta WHERE singleton = 1",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        4
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'table' AND name = 'anonymous_mailbox_issued_tickets'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}

#[test]
fn v3_unknown_acked_item_expiry_migrates_without_serving_ciphertext() {
    let context = TestContext::new();
    let mailbox = [0xA7; 32];
    let expires_at = NOW + 1_000;
    let pull = context.pull(mailbox, Vec::new(), NOW);
    let store = context.open();
    store
        .create(&context.lease(mailbox, [0xA8; 16], 1, 32, expires_at), NOW)
        .unwrap();
    store
        .put(
            &context.put(mailbox, [0xA9; 16], b"opaque", expires_at),
            NOW,
        )
        .unwrap();
    let AnonymousMailboxPullOutcome::Item(item) = store.pull_one(&pull, NOW).unwrap() else {
        panic!("item pull must return opaque ciphertext")
    };
    let ack = AnonymousMailboxAckV1::new(
        mailbox,
        [0xAA; 16],
        item.item_id,
        item.sealed_commitment,
        NOW + 1,
        &context.reader,
    )
    .unwrap();
    store.ack(&ack, NOW + 1).unwrap();
    drop(store);

    // Reproduce the previous durable schema with an ACKed replay whose
    // source item is already gone. The migration must not borrow an
    // expiry from another row or extend its old serving authority.
    let connection = Connection::open(&context.config.db_path).unwrap();
    connection
        .execute_batch(
            "ALTER TABLE anonymous_mailbox_pull_replays DROP COLUMN item_expires_at;
                 UPDATE anonymous_mailbox_meta SET schema_version = 3 WHERE singleton = 1;
                 PRAGMA user_version = 3;",
        )
        .unwrap();
    drop(connection);

    let reopened = context.open();
    let connection = reopened.connection.lock();
    assert_eq!(schema_user_version(&connection), 4);
    let preserved: (i64, i64, i64) = connection
        .query_row(
            "SELECT (SELECT COUNT(*) FROM anonymous_mailbox_acks),
                        (SELECT COUNT(*) FROM anonymous_mailbox_pull_replays),
                        (SELECT pull_replay_bytes FROM anonymous_mailbox_meta WHERE singleton = 1)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(preserved, (1, 1, 6));
    drop(connection);
    let changed_same_id = context.pull(mailbox, Vec::new(), NOW + 1);
    assert_eq!(
        reopened.pull_one(&changed_same_id, NOW + 2),
        Err(AnonymousMailboxStoreError::Rejected)
    );
    assert_eq!(
        reopened.pull_one(&pull, NOW + 2),
        Err(AnonymousMailboxStoreError::Rejected),
        "missing legacy item expiry cannot authorize ciphertext replay"
    );
}

#[test]
fn put_exact_retry_precedes_quota_and_changed_bytes_conflict() {
    let context = TestContext::new();
    let mailbox = [0x31; 32];
    let lease = context.lease(mailbox, [0x32; 16], 1, 4, NOW + 1_000);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    let put = context.put(mailbox, [0x33; 16], b"four", NOW + 100);
    assert!(matches!(
        store.put(&put, NOW).unwrap(),
        AnonymousMailboxPutOutcome::Stored(_)
    ));
    assert!(matches!(
        store.put(&put, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::Existing(_)
    ));
    let changed = context.put(mailbox, [0x33; 16], b"diff", NOW + 100);
    assert_eq!(
        store.put(&changed, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::Conflict
    );
    let over = context.put(mailbox, [0x34; 16], b"x", NOW + 100);
    assert_eq!(
        store.put(&over, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::AtCapacity
    );
}

#[test]
fn expired_items_are_never_pulled() {
    let context = TestContext::new();
    let mailbox = [0x45; 32];
    let lease = context.lease(mailbox, [0x46; 16], 2, 64, NOW + 1_000);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    store
        .put(&context.put(mailbox, [0x47; 16], b"opaque", NOW + 2), NOW)
        .unwrap();
    assert_eq!(
        store
            .pull_one(&context.pull(mailbox, Vec::new(), NOW + 3), NOW + 3)
            .unwrap(),
        AnonymousMailboxPullOutcome::Empty
    );
}

#[test]
fn ack_binds_commitment_and_exact_tombstone_is_restart_idempotent() {
    let context = TestContext::new();
    let mailbox = [0x51; 32];
    let lease = context.lease(mailbox, [0x52; 16], 2, 64, NOW + 200_000);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    let put = context.put(mailbox, [0x53; 16], b"opaque", NOW + 100_000);
    store.put(&put, NOW).unwrap();
    let ack = AnonymousMailboxAckV1::new(
        mailbox,
        [0x54; 16],
        put.item_id,
        put.sealed_commitment(),
        NOW + 1,
        &context.reader,
    )
    .unwrap();
    assert_eq!(
        store.ack(&ack, NOW + 1).unwrap(),
        AnonymousMailboxAckOutcome::Acknowledged
    );
    drop(store);
    let reopened = context.open();
    let boundary = reopened.cleanup(put.expires_at).unwrap();
    assert_eq!(boundary.acknowledgements_removed, 0);
    assert_eq!(
        reopened.ack(&ack, put.expires_at + 1).unwrap(),
        AnonymousMailboxAckOutcome::AlreadyAcknowledged
    );
    let wrong = AnonymousMailboxAckV1::new(
        mailbox,
        [0x55; 16],
        put.item_id,
        [0xEE; 32],
        NOW + 1,
        &context.reader,
    )
    .unwrap();
    assert_eq!(
        reopened.ack(&wrong, put.expires_at + 1).unwrap(),
        AnonymousMailboxAckOutcome::Conflict
    );
    let retain_until: i64 = reopened
        .connection
        .lock()
        .query_row(
            "SELECT retain_until FROM anonymous_mailbox_acks
                 WHERE mailbox_id = ?1 AND item_id = ?2",
            params![&mailbox[..], &put.item_id[..]],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(as_u64(retain_until).unwrap(), put.expires_at);
    let cleanup = reopened.cleanup(put.expires_at + 1).unwrap();
    assert_eq!(cleanup.acknowledgements_removed, 1);
    assert_eq!(cleanup.leases_removed, 0);
}

#[test]
fn lease_and_ticket_projection_tampering_fails_closed_without_mutation() {
    let context = TestContext::new();
    let mailbox = [0x57; 32];
    let request = context.lease(mailbox, [0x58; 16], 2, 64, NOW + 1_000);
    let store = context.open();
    store.create(&request, NOW).unwrap();
    let baseline: (i64, i64, i64) = store
        .connection
        .lock()
        .query_row(
            "SELECT total_leases, total_items, total_bytes FROM anonymous_mailbox_meta",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    for mutation in [
        "UPDATE anonymous_mailbox_leases SET deposit_verifier = zeroblob(32)",
        "UPDATE anonymous_mailbox_leases SET read_verifier = zeroblob(32)",
        "UPDATE anonymous_mailbox_leases SET max_items = max_items + 1",
        "UPDATE anonymous_mailbox_leases SET max_bytes = max_bytes + 1",
        "UPDATE anonymous_mailbox_leases SET expires_at = expires_at + 1",
        "UPDATE anonymous_mailbox_leases SET claims_commitment = zeroblob(32)",
        "UPDATE anonymous_mailbox_leases SET ticket_id = zeroblob(16)",
        "UPDATE anonymous_mailbox_tickets SET mailbox_id = zeroblob(32)",
        "UPDATE anonymous_mailbox_tickets SET claims_commitment = zeroblob(32)",
    ] {
        store.connection.lock().execute(mutation, []).unwrap();
        assert_eq!(
            store.create(&request, NOW + 1),
            Err(AnonymousMailboxStoreError::Corrupt)
        );
        let claims = request.claims_commitment();
        store
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_leases
                     SET ticket_id = ?1, claims_commitment = ?2, deposit_verifier = ?3,
                         read_verifier = ?4, max_items = ?5, max_bytes = ?6,
                         created_at = ?7, expires_at = ?8",
                params![
                    &request.admission.ticket_id[..],
                    &claims[..],
                    &request.deposit_verifier[..],
                    &request.read_verifier[..],
                    i64::from(request.max_items),
                    as_i64(request.max_bytes).unwrap(),
                    as_i64(request.issued_at).unwrap(),
                    as_i64(request.expires_at).unwrap(),
                ],
            )
            .unwrap();
        store
            .connection
            .lock()
            .execute(
                "UPDATE anonymous_mailbox_tickets
                     SET mailbox_id = ?1, claims_commitment = ?2
                     WHERE ticket_id = ?3",
                params![
                    &request.mailbox_id[..],
                    &claims[..],
                    &request.admission.ticket_id[..],
                ],
            )
            .unwrap();
        let observed: (i64, i64, i64) = store
            .connection
            .lock()
            .query_row(
                "SELECT total_leases, total_items, total_bytes FROM anonymous_mailbox_meta",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(observed, baseline);
    }
}

#[test]
fn global_item_row_cap_preserves_exact_put_retry_priority() {
    let mut context = TestContext::new();
    context.config.max_items_total = usize::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
    context.config.max_bytes_total = 2 * 1024 * 1024;
    let store = context.open();
    let full_mailbox = [0xA1; 32];
    let fallback_mailbox = [0xA2; 32];
    store
        .create(
            &context.lease(
                full_mailbox,
                [0xA3; 16],
                MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE,
                2 * 1024,
                NOW + 1_000,
            ),
            NOW,
        )
        .unwrap();
    store
        .create(
            &context.lease(fallback_mailbox, [0xA4; 16], 2, 64, NOW + 1_000),
            NOW,
        )
        .unwrap();
    let exact = context.put(full_mailbox, [0xA5; 16], b"x", NOW + 100);
    store.put(&exact, NOW).unwrap();

    // One transaction installs a commitment-valid large fixture without
    // turning this resource-bound regression into 1,023 FULL fsyncs.
    let commitment: [u8; 32] = Sha256::digest(b"x").into();
    let mut connection = store.connection.lock();
    let transaction = connection.transaction().unwrap();
    for sequence in 2_u64..=u64::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE) {
        let mut item_id = [0_u8; 16];
        item_id[..8].copy_from_slice(&sequence.to_le_bytes());
        transaction
            .execute(
                "INSERT INTO anonymous_mailbox_items
                     (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                      sealed_envelope, stored_at, expires_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    &full_mailbox[..],
                    &item_id[..],
                    as_i64(sequence).unwrap(),
                    &[0_u8; 32][..],
                    &commitment[..],
                    &b"x"[..],
                    as_i64(NOW).unwrap(),
                    as_i64(NOW + 100).unwrap(),
                ],
            )
            .unwrap();
    }
    let item_cap = u64::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
    transaction
        .execute(
            "UPDATE anonymous_mailbox_leases
                 SET current_items = ?1, current_bytes = ?1, next_sequence = ?2
                 WHERE mailbox_id = ?3",
            params![
                as_i64(item_cap).unwrap(),
                as_i64(item_cap + 1).unwrap(),
                &full_mailbox[..],
            ],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE anonymous_mailbox_meta SET total_items = ?1, total_bytes = ?1",
            params![as_i64(item_cap).unwrap()],
        )
        .unwrap();
    transaction.commit().unwrap();
    drop(connection);

    assert!(matches!(
        store.put(&exact, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::Existing(_)
    ));
    let blocked = context.put(fallback_mailbox, [0xA6; 16], b"y", NOW + 100);
    assert_eq!(
        store.put(&blocked, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::AtCapacity
    );
    let blocked_request = blocked.request_commitment().unwrap();
    let blocked_sealed = blocked.sealed_commitment();
    let mut connection = store.connection.lock();
    let transaction = connection.transaction().unwrap();
    transaction
        .execute(
            "INSERT INTO anonymous_mailbox_items
                 (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                  sealed_envelope, stored_at, expires_at)
                 VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7)",
            params![
                &fallback_mailbox[..],
                &blocked.item_id[..],
                &blocked_request[..],
                &blocked_sealed[..],
                &blocked.sealed_envelope,
                as_i64(NOW).unwrap(),
                as_i64(blocked.expires_at).unwrap(),
            ],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE anonymous_mailbox_leases
                 SET current_items = 1, current_bytes = 1, next_sequence = 2
                 WHERE mailbox_id = ?1",
            params![&fallback_mailbox[..]],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE anonymous_mailbox_meta
                 SET total_items = total_items + 1, total_bytes = total_bytes + 1",
            [],
        )
        .unwrap();
    transaction.commit().unwrap();
    drop(connection);
    drop(store);
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(
            context.config.clone(),
            context.target.public_key_bytes(),
            CURSOR_SECRET,
        ),
        Err(AnonymousMailboxStoreError::Corrupt)
    ));
}

#[test]
fn full_owner_storage_reopens_and_releases_capacity_without_counter_drift() {
    let mut context = TestContext::new();
    let item_cap = u64::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
    context.config.max_items_total = usize::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE);
    context.config.max_bytes_total = item_cap;
    context.config.cleanup_batch_size = 1;

    let mailbox = [0xB1; 32];
    let lease = context.lease(
        mailbox,
        [0xB2; 16],
        MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE,
        item_cap,
        NOW + 1_000,
    );
    let store = context.open();
    assert!(matches!(
        store.create(&lease, NOW).unwrap(),
        AnonymousMailboxCreateOutcome::Created(_)
    ));
    let exact = context.put(mailbox, [0xB3; 16], b"x", NOW + 100);
    assert!(matches!(
        store.put(&exact, NOW).unwrap(),
        AnonymousMailboxPutOutcome::Stored(_)
    ));

    // Production Put establishes the lease and first opaque row. Populate
    // the remaining one-byte rows in one transaction so this bounded
    // restart regression does not perform 1,023 FULL fsyncs.
    let sealed_commitment: [u8; 32] = Sha256::digest(b"x").into();
    let mut connection = store.connection.lock();
    let transaction = connection.transaction().unwrap();
    for sequence in 2..=item_cap {
        let mut item_id = [0_u8; 16];
        item_id[..8].copy_from_slice(&sequence.to_le_bytes());
        let expires_at = if sequence == 2 { NOW + 2 } else { NOW + 100 };
        transaction
            .execute(
                "INSERT INTO anonymous_mailbox_items
                     (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                      sealed_envelope, stored_at, expires_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    &mailbox[..],
                    &item_id[..],
                    as_i64(sequence).unwrap(),
                    &[0_u8; 32][..],
                    &sealed_commitment[..],
                    &b"x"[..],
                    as_i64(NOW).unwrap(),
                    as_i64(expires_at).unwrap(),
                ],
            )
            .unwrap();
    }
    transaction
        .execute(
            "UPDATE anonymous_mailbox_leases
                 SET current_items = ?1, current_bytes = ?1, next_sequence = ?2
                 WHERE mailbox_id = ?3",
            params![
                as_i64(item_cap).unwrap(),
                as_i64(item_cap + 1).unwrap(),
                &mailbox[..],
            ],
        )
        .unwrap();
    transaction
        .execute(
            "UPDATE anonymous_mailbox_meta SET total_items = ?1, total_bytes = ?1",
            params![as_i64(item_cap).unwrap()],
        )
        .unwrap();
    transaction.commit().unwrap();
    drop(connection);
    assert_item_accounting(&store, &mailbox, item_cap, item_cap);
    drop(store);

    let reopened = context.open();
    assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);
    assert!(matches!(
        reopened.put(&exact, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::Existing(_)
    ));
    let changed = context.put(mailbox, exact.item_id, b"y", exact.expires_at);
    assert_eq!(
        reopened.put(&changed, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::Conflict
    );
    let blocked = context.put(mailbox, [0xB4; 16], b"z", NOW + 100);
    assert_eq!(
        reopened.put(&blocked, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::AtCapacity
    );
    assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);

    let wrong_reader = IdentityKeyPair::generate();
    let wrong_ack = AnonymousMailboxAckV1::new(
        mailbox,
        [0xB5; 16],
        exact.item_id,
        exact.sealed_commitment(),
        NOW + 1,
        &wrong_reader,
    )
    .unwrap();
    assert_eq!(
        reopened.ack(&wrong_ack, NOW + 1),
        Err(AnonymousMailboxStoreError::Rejected)
    );
    assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);

    let ack = AnonymousMailboxAckV1::new(
        mailbox,
        [0xB6; 16],
        exact.item_id,
        exact.sealed_commitment(),
        NOW + 1,
        &context.reader,
    )
    .unwrap();
    assert_eq!(
        reopened.ack(&ack, NOW + 1).unwrap(),
        AnonymousMailboxAckOutcome::Acknowledged
    );
    assert_item_accounting(&reopened, &mailbox, item_cap - 1, item_cap - 1);
    assert!(matches!(
        reopened.put(&blocked, NOW + 1).unwrap(),
        AnonymousMailboxPutOutcome::Stored(_)
    ));
    assert_item_accounting(&reopened, &mailbox, item_cap, item_cap);
    drop(reopened);

    let after_ack_restart = context.open();
    assert_item_accounting(&after_ack_restart, &mailbox, item_cap, item_cap);
    let first_visible = after_ack_restart
        .pull_one(&context.pull(mailbox, Vec::new(), NOW + 3), NOW + 3)
        .unwrap();
    let mut first_unexpired_item_id = [0_u8; 16];
    first_unexpired_item_id[..8].copy_from_slice(&3_u64.to_le_bytes());
    assert!(matches!(
        first_visible,
        AnonymousMailboxPullOutcome::Item(ref item)
            if item.item_id == first_unexpired_item_id
    ));
    let cleanup = after_ack_restart.cleanup(NOW + 3).unwrap();
    assert_eq!(cleanup.items_removed, 1);
    assert_eq!(cleanup.bytes_removed, 1);
    assert_eq!(cleanup.acknowledgements_removed, 0);
    assert_item_accounting(&after_ack_restart, &mailbox, item_cap - 1, item_cap - 1);

    let after_expiry = context.put(mailbox, [0xB7; 16], b"q", NOW + 100);
    assert!(matches!(
        after_ack_restart.put(&after_expiry, NOW + 3).unwrap(),
        AnonymousMailboxPutOutcome::Stored(_)
    ));
    assert_item_accounting(&after_ack_restart, &mailbox, item_cap, item_cap);
    drop(after_ack_restart);

    let final_reopen = context.open();
    assert_item_accounting(&final_reopen, &mailbox, item_cap, item_cap);
    assert!(matches!(
        final_reopen.put(&after_expiry, NOW + 4).unwrap(),
        AnonymousMailboxPutOutcome::Existing(_)
    ));
}

#[test]
fn cleanup_is_bounded_and_failure_rolls_back_without_counter_repair() {
    let mut context = TestContext::new();
    context.config.cleanup_batch_size = 1;
    let mailbox = [0x71; 32];
    let lease = context.lease(mailbox, [0x72; 16], 2, 64, NOW + 1_000);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    store
        .put(&context.put(mailbox, [0x73; 16], b"one", NOW + 2), NOW)
        .unwrap();
    store
        .put(&context.put(mailbox, [0x74; 16], b"two", NOW + 2), NOW)
        .unwrap();
    let report = store.cleanup(NOW + 3).unwrap();
    assert_eq!(report.items_removed, 1);
    assert_eq!(
        count(
            &store.connection.lock().transaction().unwrap(),
            "SELECT COUNT(*) FROM anonymous_mailbox_items"
        )
        .unwrap(),
        1
    );

    store
        .connection
        .lock()
        .execute_batch(
            "CREATE TRIGGER anonymous_mailbox_test_cleanup_abort
                 BEFORE DELETE ON anonymous_mailbox_items
                 BEGIN SELECT RAISE(ABORT, 'test'); END;",
        )
        .unwrap();
    assert_eq!(
        store.cleanup(NOW + 3),
        Err(AnonymousMailboxStoreError::Unavailable)
    );
    let remaining: i64 = store
        .connection
        .lock()
        .query_row("SELECT COUNT(*) FROM anonymous_mailbox_items", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(remaining, 1);

    store
        .connection
        .lock()
        .execute_batch(
            "DROP TRIGGER anonymous_mailbox_test_cleanup_abort;
                 UPDATE anonymous_mailbox_meta SET total_bytes = total_bytes + 1;",
        )
        .unwrap();
    assert_eq!(
        store.cleanup(NOW + 3),
        Err(AnonymousMailboxStoreError::Corrupt)
    );
    let after_corrupt: i64 = store
        .connection
        .lock()
        .query_row("SELECT COUNT(*) FROM anonymous_mailbox_items", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(after_corrupt, 1);
}
