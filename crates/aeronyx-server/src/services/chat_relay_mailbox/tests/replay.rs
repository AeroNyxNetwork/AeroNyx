// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn exact_pull_replay_survives_ack_and_restart_without_advancing_to_next_item() {
    let context = TestContext::new();
    let mailbox = [0xC1; 32];
    let lease = context.lease(mailbox, [0xC2; 16], 2, 32, NOW + 1_000);
    let first_put = context.put(mailbox, [0xC3; 16], b"opaque-a", NOW + 900);
    let second_put = context.put(mailbox, [0xC4; 16], b"opaque-b", NOW + 900);
    let pull = context.pull(mailbox, Vec::new(), NOW);

    let store = context.open();
    store.create(&lease, NOW).unwrap();
    store.put(&first_put, NOW).unwrap();
    store.put(&second_put, NOW).unwrap();
    let first = store.pull_one(&pull, NOW).unwrap();
    let AnonymousMailboxPullOutcome::Item(first_item) = &first else {
        panic!("first pull must return an opaque item")
    };
    assert_eq!(first_item.item_id, first_put.item_id);
    let ack = AnonymousMailboxAckV1::new(
        mailbox,
        [0xC5; 16],
        first_item.item_id,
        first_item.sealed_commitment,
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
    assert_eq!(reopened.pull_one(&pull, NOW + 600).unwrap(), first);
    let changed_same_id = context.pull(mailbox, Vec::new(), NOW + 1);
    assert_eq!(
        reopened.pull_one(&changed_same_id, NOW + 1),
        Err(AnonymousMailboxStoreError::Rejected)
    );
    let fresh_id =
        AnonymousMailboxPullOneV1::new(mailbox, [0xC6; 16], Vec::new(), NOW + 1, &context.reader)
            .unwrap();
    assert!(matches!(
        reopened.pull_one(&fresh_id, NOW + 1).unwrap(),
        AnonymousMailboxPullOutcome::Item(ref item) if item.item_id == second_put.item_id
    ));
}

#[test]
fn exact_empty_pull_replay_survives_later_put_and_restart() {
    let context = TestContext::new();
    let mailbox = [0xD2; 32];
    let lease = context.lease(mailbox, [0xD3; 16], 1, 32, NOW + 1_000);
    let pull = context.pull(mailbox, Vec::new(), NOW);
    let put = context.put(mailbox, [0xD4; 16], b"opaque-later", NOW + 900);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    assert_eq!(
        store.pull_one(&pull, NOW).unwrap(),
        AnonymousMailboxPullOutcome::Empty
    );
    store.put(&put, NOW + 1).unwrap();
    drop(store);

    let reopened = context.open();
    assert_eq!(
        reopened.pull_one(&pull, NOW + 600).unwrap(),
        AnonymousMailboxPullOutcome::Empty
    );
    let fresh =
        AnonymousMailboxPullOneV1::new(mailbox, [0xD5; 16], Vec::new(), NOW + 1, &context.reader)
            .unwrap();
    assert!(matches!(
        reopened.pull_one(&fresh, NOW + 1).unwrap(),
        AnonymousMailboxPullOutcome::Item(ref item) if item.item_id == put.item_id
    ));
}

#[test]
fn pull_replay_retention_boundary_fails_closed_before_cleanup() {
    let context = TestContext::new();
    let item_mailbox = [0xE1; 32];
    let empty_mailbox = [0xE2; 32];
    let expires_at = NOW + 1_000;
    let item_pull = context.pull(item_mailbox, Vec::new(), NOW);
    let empty_pull = context.pull(empty_mailbox, Vec::new(), NOW);
    let store = context.open();
    store
        .create(
            &context.lease(item_mailbox, [0xE3; 16], 1, 32, expires_at),
            NOW,
        )
        .unwrap();
    store
        .create(
            &context.lease(empty_mailbox, [0xE4; 16], 1, 32, expires_at),
            NOW,
        )
        .unwrap();
    let put = context.put(item_mailbox, [0xE5; 16], b"opaque", expires_at);
    store.put(&put, NOW).unwrap();
    let first = store.pull_one(&item_pull, NOW).unwrap();
    assert_eq!(
        store.pull_one(&empty_pull, NOW).unwrap(),
        AnonymousMailboxPullOutcome::Empty
    );
    let AnonymousMailboxPullOutcome::Item(item) = &first else {
        panic!("item pull must return an opaque item")
    };
    let ack = AnonymousMailboxAckV1::new(
        item_mailbox,
        [0xE6; 16],
        item.item_id,
        item.sealed_commitment,
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
    assert_eq!(reopened.pull_one(&item_pull, expires_at).unwrap(), first);
    assert_eq!(
        reopened.pull_one(&empty_pull, expires_at).unwrap(),
        AnonymousMailboxPullOutcome::Empty
    );
    for pull in [&item_pull, &empty_pull] {
        assert_eq!(
            reopened.pull_one(pull, expires_at + 1),
            Err(AnonymousMailboxStoreError::Rejected),
            "retention must end even before cleanup removes the row"
        );
    }
    let changed_same_id = context.pull(item_mailbox, Vec::new(), NOW + 1);
    assert_eq!(
        reopened.pull_one(&changed_same_id, expires_at),
        Err(AnonymousMailboxStoreError::Rejected)
    );
}

#[test]
fn item_expiry_bounds_exact_replay_after_ack_and_restart_without_changing_empty() {
    let context = TestContext::new();
    let item_mailbox = [0xA1; 32];
    let empty_mailbox = [0xA2; 32];
    let item_expires_at = NOW + 10;
    let lease_expires_at = NOW + 1_000;
    let item_pull = context.pull(item_mailbox, Vec::new(), NOW);
    let empty_pull = context.pull(empty_mailbox, Vec::new(), NOW);
    let store = context.open();
    store
        .create(
            &context.lease(item_mailbox, [0xA3; 16], 1, 32, lease_expires_at),
            NOW,
        )
        .unwrap();
    store
        .create(
            &context.lease(empty_mailbox, [0xA4; 16], 1, 32, lease_expires_at),
            NOW,
        )
        .unwrap();
    store
        .put(
            &context.put(item_mailbox, [0xA5; 16], b"opaque", item_expires_at),
            NOW,
        )
        .unwrap();
    let first = store.pull_one(&item_pull, NOW).unwrap();
    assert_eq!(
        store.pull_one(&empty_pull, NOW).unwrap(),
        AnonymousMailboxPullOutcome::Empty
    );
    let AnonymousMailboxPullOutcome::Item(item) = &first else {
        panic!("item pull must return opaque ciphertext")
    };
    let ack = AnonymousMailboxAckV1::new(
        item_mailbox,
        [0xA6; 16],
        item.item_id,
        item.sealed_commitment,
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
    assert_eq!(
        reopened.pull_one(&item_pull, item_expires_at).unwrap(),
        first
    );
    let changed_same_id = context.pull(item_mailbox, Vec::new(), NOW + 1);
    assert_eq!(
        reopened.pull_one(&changed_same_id, item_expires_at + 1),
        Err(AnonymousMailboxStoreError::Rejected),
        "same id with different signed content remains a conflict first"
    );
    assert_eq!(
        reopened.pull_one(&item_pull, item_expires_at + 1),
        Err(AnonymousMailboxStoreError::Rejected)
    );
    assert_eq!(
        reopened.pull_one(&empty_pull, item_expires_at + 1).unwrap(),
        AnonymousMailboxPullOutcome::Empty,
        "an Empty result has no item expiry to inherit"
    );
}

#[test]
fn pull_replay_budget_and_cleanup_are_bounded() {
    let context = TestContext::new();
    let mailbox = [0xC7; 32];
    let lease = context.lease(mailbox, [0xC8; 16], 2, 32, NOW + 1_000);
    let mut store = context.open();
    store.config.max_items_total = 2;
    store.config.cleanup_batch_size = 1;
    store.create(&lease, NOW).unwrap();
    for request_id in [[0xC9; 16], [0xCA; 16]] {
        let request =
            AnonymousMailboxPullOneV1::new(mailbox, request_id, Vec::new(), NOW, &context.reader)
                .unwrap();
        assert_eq!(
            store.pull_one(&request, NOW).unwrap(),
            AnonymousMailboxPullOutcome::Empty
        );
    }
    let over_cap =
        AnonymousMailboxPullOneV1::new(mailbox, [0xCB; 16], Vec::new(), NOW, &context.reader)
            .unwrap();
    assert_eq!(
        store.pull_one(&over_cap, NOW),
        Err(AnonymousMailboxStoreError::Busy)
    );
    let report = store.cleanup(NOW + PULL_REPLAY_RETENTION_SECS + 1).unwrap();
    assert_eq!(report.pull_replays_removed, 1);
    let connection = store.connection.lock();
    assert_eq!(
        connection
            .query_row(
                "SELECT pull_replay_rows FROM anonymous_mailbox_meta WHERE singleton = 1",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}

#[test]
fn pull_replay_ciphertext_corruption_fails_closed_on_restart() {
    let context = TestContext::new();
    let mailbox = [0xCC; 32];
    let lease = context.lease(mailbox, [0xCD; 16], 1, 32, NOW + 1_000);
    let put = context.put(mailbox, [0xCE; 16], b"opaque", NOW + 900);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    store.put(&put, NOW).unwrap();
    store
        .pull_one(&context.pull(mailbox, Vec::new(), NOW), NOW)
        .unwrap();
    drop(store);

    let connection = Connection::open(&context.config.db_path).unwrap();
    connection
        .execute(
            "UPDATE anonymous_mailbox_pull_replays SET sealed_envelope = ?1",
            params![b"tampered-opaque".as_slice()],
        )
        .unwrap();
    drop(connection);
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
fn oversized_replay_blobs_fail_closed_on_lookup_and_startup() {
    for oversize_cursor in [false, true] {
        let context = TestContext::new();
        let mailbox = [0xE7; 32];
        let pull = context.pull(mailbox, Vec::new(), NOW);
        let store = context.open();
        store
            .create(&context.lease(mailbox, [0xE8; 16], 1, 32, NOW + 1_000), NOW)
            .unwrap();
        store
            .put(&context.put(mailbox, [0xE9; 16], b"opaque", NOW + 900), NOW)
            .unwrap();
        assert!(matches!(
            store.pull_one(&pull, NOW).unwrap(),
            AnonymousMailboxPullOutcome::Item(_)
        ));

        let connection = Connection::open(&context.config.db_path).unwrap();
        // Simulate a physically valid but corrupt older V3 file. Current
        // fresh-schema CHECK constraints reject this normal write.
        connection
            .execute_batch("PRAGMA ignore_check_constraints = ON;")
            .unwrap();
        if oversize_cursor {
            connection
                .execute(
                    "UPDATE anonymous_mailbox_pull_replays SET cursor = zeroblob(?1)",
                    params![i64::try_from(CURSOR_BYTES + 1).unwrap()],
                )
                .unwrap();
        } else {
            connection
                .execute(
                    "UPDATE anonymous_mailbox_pull_replays
                         SET sealed_envelope = zeroblob(?1)",
                    params![i64::try_from(MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES + 1).unwrap()],
                )
                .unwrap();
        }
        drop(connection);
        assert_eq!(
            store.pull_one(&pull, NOW + 1),
            Err(AnonymousMailboxStoreError::Corrupt)
        );
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
}

#[test]
fn ticket_issue_replays_exactly_conflicts_and_survives_restart() {
    let context = TestContext::new();
    let request = context.ticket_issue([0x81; 16], [0x82; 16], [0x83; 32], NOW + 1_000, NOW + 300);
    let store = context.open_with_ticket_issuer();
    let issued = match store.issue_ticket(&request, NOW).unwrap() {
        AnonymousMailboxTicketIssueOutcome::Issued(ticket) => ticket,
        other => panic!("unexpected issue result: {other:?}"),
    };
    assert!(matches!(
        store.issue_ticket(&request, NOW + 1).unwrap(),
        AnonymousMailboxTicketIssueOutcome::Existing(ticket) if ticket == issued
    ));
    let conflict = context.ticket_issue(
        request.request_id,
        [0x84; 16],
        [0x83; 32],
        NOW + 1_000,
        NOW + 300,
    );
    assert_eq!(
        store.issue_ticket(&conflict, NOW + 1).unwrap(),
        AnonymousMailboxTicketIssueOutcome::Conflict
    );
    drop(store);
    let reopened = context.open_with_ticket_issuer();
    assert!(matches!(
        reopened.issue_ticket(&request, NOW + 2).unwrap(),
        AnonymousMailboxTicketIssueOutcome::Existing(ticket) if ticket == issued
    ));
}

#[test]
fn issued_ticket_is_consumed_once_and_capacity_replay_precedes_limits() {
    let mut context = TestContext::new();
    context.config.max_outstanding_tickets = 2;
    context.config.max_ticket_issues_per_window = 1;
    let first = context.ticket_issue([0x91; 16], [0x92; 16], [0x93; 32], NOW + 1_000, NOW + 300);
    let second = context.ticket_issue([0x94; 16], [0x95; 16], [0x96; 32], NOW + 1_000, NOW + 300);
    let store = context.open_with_ticket_issuer();
    let ticket = match store.issue_ticket(&first, NOW).unwrap() {
        AnonymousMailboxTicketIssueOutcome::Issued(ticket) => ticket,
        other => panic!("unexpected issue result: {other:?}"),
    };
    assert_eq!(
        store.issue_ticket(&second, NOW + 1).unwrap(),
        AnonymousMailboxTicketIssueOutcome::AtCapacity
    );
    assert!(matches!(
        store.issue_ticket(&first, NOW + 1).unwrap(),
        AnonymousMailboxTicketIssueOutcome::Existing(existing) if existing == ticket
    ));
    let lease = context.lease_for_issued_ticket([0x93; 32], NOW + 1_000, ticket.clone());
    assert!(matches!(
        store.create(&lease, NOW + 2).unwrap(),
        AnonymousMailboxCreateOutcome::Created(_)
    ));
    assert!(matches!(
        store.create(&lease, NOW + 3).unwrap(),
        AnonymousMailboxCreateOutcome::Existing(_)
    ));
    assert!(matches!(
        store.issue_ticket(&first, NOW + 3).unwrap(),
        AnonymousMailboxTicketIssueOutcome::Existing(existing) if existing == ticket
    ));
    assert!(matches!(
        store.issue_ticket(&second, NOW + 60).unwrap(),
        AnonymousMailboxTicketIssueOutcome::Issued(_)
    ));
}

#[test]
fn pull_is_single_padded_snapshot_and_cursor_is_restart_stable_and_authenticated() {
    let context = TestContext::new();
    let mailbox = [0x41; 32];
    let lease = context.lease(mailbox, [0x42; 16], 3, 64, NOW + 1_000);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    store
        .put(&context.put(mailbox, [1; 16], b"one", NOW + 100), NOW)
        .unwrap();
    store
        .put(&context.put(mailbox, [2; 16], b"two", NOW + 100), NOW)
        .unwrap();
    let first = match store
        .pull_one(&context.pull(mailbox, Vec::new(), NOW), NOW)
        .unwrap()
    {
        AnonymousMailboxPullOutcome::Item(item) => item,
        other => panic!("unexpected pull outcome: {other:?}"),
    };
    assert_eq!(first.item_id, [1; 16]);
    assert_eq!(first.sealed_length, 3);
    assert_eq!(
        first.padded_sealed_envelope.len(),
        MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
    );
    assert_eq!(&first.padded_sealed_envelope[..3], b"one");
    assert_eq!(first.cursor.len(), CURSOR_BYTES);

    let mut tampered = first.cursor.clone();
    tampered[10] ^= 1;
    assert_eq!(
        store.pull_one(&context.pull(mailbox, tampered, NOW + 1), NOW + 1),
        Err(AnonymousMailboxStoreError::Rejected)
    );
    drop(store);

    let reopened = context.open();
    // [ANONYMOUS-MAILBOX-ITEM-EXPIRY 2026-09-24 by Codex] A changed
    // cursor is a new signed Pull transcript, not an exact retry. Reusing
    // TestContext::pull's fixed request id would test conflict instead.
    let next_pull = AnonymousMailboxPullOneV1::new(
        mailbox,
        [0x52; 16],
        first.cursor.clone(),
        NOW + 1,
        &context.reader,
    )
    .unwrap();
    let second = reopened.pull_one(&next_pull, NOW + 1).unwrap();
    match second {
        AnonymousMailboxPullOutcome::Item(item) => assert_eq!(item.item_id, [2; 16]),
        other => panic!("unexpected pull outcome: {other:?}"),
    }
    drop(reopened);

    let wrong_key = SqliteAnonymousMailboxStore::open(
        context.config.clone(),
        context.target.public_key_bytes(),
        [0x5A; 32],
    )
    .unwrap();
    let wrong_key_pull =
        AnonymousMailboxPullOneV1::new(mailbox, [0x53; 16], first.cursor, NOW + 1, &context.reader)
            .unwrap();
    assert_eq!(
        wrong_key.pull_one(&wrong_key_pull, NOW + 1),
        Err(AnonymousMailboxStoreError::Rejected)
    );
}
