// [ARCH-SPLIT 2026-10-02]
// Schema verification and exact counter audits before custody mutations.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn initialize_or_verify_schema(
    connection: &mut Connection,
    target_node_id: &[u8; 32],
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let user_version: i64 = transaction
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if user_version == 0 {
        // [ANONYMOUS-MAILBOX-SCHEMA-OWNERSHIP 2026-09-02 by Codex]
        // Claim only an otherwise-empty reserved database so unrelated
        // permanent schema objects never cohabit this node-blind store.
        let foreign_schema_objects: i64 = transaction
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type IN ('table', 'index', 'view', 'trigger')
                   AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if foreign_schema_objects != 0 {
            return Err(AnonymousMailboxStoreError::UnsupportedSchema);
        }
        transaction
            .execute_batch(
                "CREATE TABLE anonymous_mailbox_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL,
                    total_leases INTEGER NOT NULL CHECK (total_leases >= 0),
                    total_items INTEGER NOT NULL CHECK (total_items >= 0),
                    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
                    outstanding_tickets INTEGER NOT NULL CHECK (outstanding_tickets >= 0),
                    issuance_window_started_at INTEGER NOT NULL CHECK (issuance_window_started_at >= 0),
                    issues_in_window INTEGER NOT NULL CHECK (issues_in_window >= 0),
                    pull_replay_rows INTEGER NOT NULL CHECK (pull_replay_rows >= 0),
                    pull_replay_bytes INTEGER NOT NULL CHECK (pull_replay_bytes >= 0)
                 );
                 INSERT INTO anonymous_mailbox_meta VALUES (1, 4, 0, 0, 0, 0, 0, 0, 0, 0);
                 CREATE TABLE anonymous_mailbox_tickets (
                    ticket_id BLOB PRIMARY KEY CHECK (length(ticket_id) = 16),
                    ticket_commitment BLOB NOT NULL CHECK (length(ticket_commitment) = 32),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    lease_request_commitment BLOB NOT NULL CHECK (length(lease_request_commitment) = 32),
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    consumed_at INTEGER NOT NULL CHECK (consumed_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= consumed_at)
                 );
                 CREATE TABLE anonymous_mailbox_leases (
                    mailbox_id BLOB PRIMARY KEY CHECK (length(mailbox_id) = 32),
                    ticket_id BLOB NOT NULL UNIQUE CHECK (length(ticket_id) = 16),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    deposit_verifier BLOB NOT NULL CHECK (length(deposit_verifier) = 32),
                    read_verifier BLOB NOT NULL CHECK (length(read_verifier) = 32),
                    max_items INTEGER NOT NULL CHECK (max_items > 0),
                    max_bytes INTEGER NOT NULL CHECK (max_bytes > 0),
                    current_items INTEGER NOT NULL CHECK (current_items >= 0 AND current_items <= max_items),
                    current_bytes INTEGER NOT NULL CHECK (current_bytes >= 0 AND current_bytes <= max_bytes),
                    next_sequence INTEGER NOT NULL CHECK (next_sequence > 0),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= created_at)
                 );
                 CREATE TABLE anonymous_mailbox_items (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    item_id BLOB NOT NULL CHECK (length(item_id) = 16),
                    sequence INTEGER NOT NULL CHECK (sequence > 0),
                    put_commitment BLOB NOT NULL CHECK (length(put_commitment) = 32),
                    sealed_commitment BLOB NOT NULL CHECK (length(sealed_commitment) = 32),
                    sealed_envelope BLOB NOT NULL CHECK (length(sealed_envelope) > 0),
                    stored_at INTEGER NOT NULL CHECK (stored_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= stored_at),
                    PRIMARY KEY (mailbox_id, item_id),
                    UNIQUE (mailbox_id, sequence),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE
                 );
                 CREATE TABLE anonymous_mailbox_acks (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    item_id BLOB NOT NULL CHECK (length(item_id) = 16),
                    sealed_commitment BLOB NOT NULL CHECK (length(sealed_commitment) = 32),
                    ack_request_commitment BLOB NOT NULL CHECK (length(ack_request_commitment) = 32),
                    acknowledged_at INTEGER NOT NULL CHECK (acknowledged_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= acknowledged_at),
                    PRIMARY KEY (mailbox_id, item_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE
                 );
                 CREATE TABLE anonymous_mailbox_issued_tickets (
                    request_id BLOB PRIMARY KEY CHECK (length(request_id) = 16),
                    ticket_id BLOB NOT NULL UNIQUE CHECK (length(ticket_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    target_node_id BLOB NOT NULL CHECK (length(target_node_id) = 32),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    requested_at INTEGER NOT NULL CHECK (requested_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= requested_at),
                    proof_nonce BLOB NOT NULL CHECK (length(proof_nonce) = 8),
                    ticket_commitment BLOB NOT NULL CHECK (length(ticket_commitment) = 32),
                    ticket_signature BLOB NOT NULL CHECK (length(ticket_signature) = 64),
                    consumed_at INTEGER CHECK (consumed_at >= requested_at)
                 );
                 CREATE TABLE anonymous_mailbox_pull_replays (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    request_id BLOB NOT NULL CHECK (length(request_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    outcome INTEGER NOT NULL CHECK (outcome IN (0, 1)),
                    item_id BLOB CHECK (item_id IS NULL OR length(item_id) = 16),
                    sealed_commitment BLOB CHECK (sealed_commitment IS NULL OR length(sealed_commitment) = 32),
                    sealed_envelope BLOB,
                    cursor BLOB CHECK (cursor IS NULL OR (typeof(cursor) = 'blob' AND length(cursor) = 57)),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= created_at),
                    item_expires_at INTEGER,
                    PRIMARY KEY (mailbox_id, request_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE,
                    CHECK ((outcome = 0 AND item_id IS NULL AND sealed_commitment IS NULL
                            AND sealed_envelope IS NULL AND cursor IS NULL)
                        OR (outcome = 1 AND item_id IS NOT NULL AND sealed_commitment IS NOT NULL
                            AND sealed_envelope IS NOT NULL AND typeof(sealed_envelope) = 'blob'
                            AND length(sealed_envelope) BETWEEN 1 AND 162816
                            AND cursor IS NOT NULL))
                 );
                 CREATE INDEX anonymous_mailbox_lease_expiry
                    ON anonymous_mailbox_leases(expires_at, mailbox_id);
                 CREATE INDEX anonymous_mailbox_item_pull
                    ON anonymous_mailbox_items(mailbox_id, sequence, expires_at);
                 CREATE INDEX anonymous_mailbox_item_expiry
                    ON anonymous_mailbox_items(expires_at, mailbox_id, item_id);
                 CREATE INDEX anonymous_mailbox_ack_expiry
                    ON anonymous_mailbox_acks(retain_until, mailbox_id, item_id);
                 CREATE INDEX anonymous_mailbox_issued_ticket_expiry
                    ON anonymous_mailbox_issued_tickets(expires_at, request_id);
                 CREATE INDEX anonymous_mailbox_pull_replay_expiry
                    ON anonymous_mailbox_pull_replays(retain_until, mailbox_id, request_id);
                 PRAGMA user_version = 4;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    } else if user_version == 1 {
        // [ANONYMOUS-MAILBOX-TICKET-ISSUER 2026-09-03 by Codex] The v2
        // journal is additive: existing consumed tickets and leases retain
        // their frozen rows while only future target-issued authorities gain
        // exact replay evidence.
        transaction
            .execute_batch(
                "ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN outstanding_tickets INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN issuance_window_started_at INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN issues_in_window INTEGER NOT NULL DEFAULT 0;
                 CREATE TABLE anonymous_mailbox_issued_tickets (
                    request_id BLOB PRIMARY KEY CHECK (length(request_id) = 16),
                    ticket_id BLOB NOT NULL UNIQUE CHECK (length(ticket_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    target_node_id BLOB NOT NULL CHECK (length(target_node_id) = 32),
                    claims_commitment BLOB NOT NULL CHECK (length(claims_commitment) = 32),
                    requested_at INTEGER NOT NULL CHECK (requested_at >= 0),
                    expires_at INTEGER NOT NULL CHECK (expires_at >= requested_at),
                    proof_nonce BLOB NOT NULL CHECK (length(proof_nonce) = 8),
                    ticket_commitment BLOB NOT NULL CHECK (length(ticket_commitment) = 32),
                    ticket_signature BLOB NOT NULL CHECK (length(ticket_signature) = 64),
                    consumed_at INTEGER CHECK (consumed_at >= requested_at)
                 );
                 CREATE INDEX anonymous_mailbox_issued_ticket_expiry
                    ON anonymous_mailbox_issued_tickets(expires_at, request_id);
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_rows INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_bytes INTEGER NOT NULL DEFAULT 0;
                 CREATE TABLE anonymous_mailbox_pull_replays (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    request_id BLOB NOT NULL CHECK (length(request_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    outcome INTEGER NOT NULL CHECK (outcome IN (0, 1)),
                    item_id BLOB CHECK (item_id IS NULL OR length(item_id) = 16),
                    sealed_commitment BLOB CHECK (sealed_commitment IS NULL OR length(sealed_commitment) = 32),
                    sealed_envelope BLOB,
                    cursor BLOB CHECK (cursor IS NULL OR (typeof(cursor) = 'blob' AND length(cursor) = 57)),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= created_at),
                    item_expires_at INTEGER,
                    PRIMARY KEY (mailbox_id, request_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE,
                    CHECK ((outcome = 0 AND item_id IS NULL AND sealed_commitment IS NULL
                            AND sealed_envelope IS NULL AND cursor IS NULL)
                        OR (outcome = 1 AND item_id IS NOT NULL AND sealed_commitment IS NOT NULL
                            AND sealed_envelope IS NOT NULL AND typeof(sealed_envelope) = 'blob'
                            AND length(sealed_envelope) BETWEEN 1 AND 162816
                            AND cursor IS NOT NULL))
                 );
                 CREATE INDEX anonymous_mailbox_pull_replay_expiry
                    ON anonymous_mailbox_pull_replays(retain_until, mailbox_id, request_id);
                 UPDATE anonymous_mailbox_meta SET schema_version = 4 WHERE singleton = 1;
                 PRAGMA user_version = 4;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    } else if user_version == 2 {
        // [ANONYMOUS-MAILBOX-PULL-REPLAY 2026-09-24 by Codex] V3 adds only
        // node-blind, bounded replay evidence; all V1/V2 leases, tickets,
        // items, and ACK tombstones remain byte-for-byte readable.
        transaction
            .execute_batch(
                "ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_rows INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE anonymous_mailbox_meta
                    ADD COLUMN pull_replay_bytes INTEGER NOT NULL DEFAULT 0;
                 CREATE TABLE anonymous_mailbox_pull_replays (
                    mailbox_id BLOB NOT NULL CHECK (length(mailbox_id) = 32),
                    request_id BLOB NOT NULL CHECK (length(request_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    outcome INTEGER NOT NULL CHECK (outcome IN (0, 1)),
                    item_id BLOB CHECK (item_id IS NULL OR length(item_id) = 16),
                    sealed_commitment BLOB CHECK (sealed_commitment IS NULL OR length(sealed_commitment) = 32),
                    sealed_envelope BLOB,
                    cursor BLOB CHECK (cursor IS NULL OR (typeof(cursor) = 'blob' AND length(cursor) = 57)),
                    created_at INTEGER NOT NULL CHECK (created_at >= 0),
                    retain_until INTEGER NOT NULL CHECK (retain_until >= created_at),
                    item_expires_at INTEGER,
                    PRIMARY KEY (mailbox_id, request_id),
                    FOREIGN KEY (mailbox_id) REFERENCES anonymous_mailbox_leases(mailbox_id) ON DELETE CASCADE,
                    CHECK ((outcome = 0 AND item_id IS NULL AND sealed_commitment IS NULL
                            AND sealed_envelope IS NULL AND cursor IS NULL)
                        OR (outcome = 1 AND item_id IS NOT NULL AND sealed_commitment IS NOT NULL
                            AND sealed_envelope IS NOT NULL AND typeof(sealed_envelope) = 'blob'
                            AND length(sealed_envelope) BETWEEN 1 AND 162816
                            AND cursor IS NOT NULL))
                 );
                 CREATE INDEX anonymous_mailbox_pull_replay_expiry
                    ON anonymous_mailbox_pull_replays(retain_until, mailbox_id, request_id);
                 UPDATE anonymous_mailbox_meta SET schema_version = 4 WHERE singleton = 1;
                 PRAGMA user_version = 4;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    } else if user_version == 3 {
        // [ANONYMOUS-MAILBOX-ITEM-EXPIRY 2026-09-24 by Codex] Add only the
        // missing expiry column under the same IMMEDIATE transaction. An old
        // Item with no expiry stays unreadable even if its source item still
        // exists; ACK may have deleted that source before this migration.
        transaction
            .execute_batch(
                "ALTER TABLE anonymous_mailbox_pull_replays
                    ADD COLUMN item_expires_at INTEGER;
                 UPDATE anonymous_mailbox_meta SET schema_version = 4 WHERE singleton = 1;
                 PRAGMA user_version = 4;",
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    } else if user_version != SCHEMA_VERSION {
        return Err(AnonymousMailboxStoreError::UnsupportedSchema);
    }
    let version: i64 = transaction
        .query_row(
            "SELECT schema_version FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if version != SCHEMA_VERSION {
        return Err(AnonymousMailboxStoreError::UnsupportedSchema);
    }
    transaction
        .commit()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    audit_counters(&transaction, target_node_id, config)?;
    transaction
        .commit()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)
}

pub(super) fn load_totals(
    transaction: &Transaction<'_>,
) -> Result<StoreTotals, AnonymousMailboxStoreError> {
    let (leases, items, bytes): (i64, i64, i64) = transaction
        .query_row(
            "SELECT total_leases, total_items, total_bytes
             FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    Ok(StoreTotals {
        leases: as_u64(leases)?,
        items: as_u64(items)?,
        bytes: as_u64(bytes)?,
    })
}

pub(super) fn load_pull_replay_totals(
    transaction: &Transaction<'_>,
) -> Result<PullReplayTotals, AnonymousMailboxStoreError> {
    let (rows, bytes): (i64, i64) = transaction
        .query_row(
            "SELECT pull_replay_rows, pull_replay_bytes
             FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    Ok(PullReplayTotals {
        rows: as_u64(rows)?,
        bytes: as_u64(bytes)?,
    })
}

pub(super) fn validate_pull_replay_totals(
    totals: PullReplayTotals,
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    if totals.rows
        > u64::try_from(config.max_items_total).map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        || totals.bytes > config.max_bytes_total
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

pub(super) fn update_pull_replay_totals_exact(
    transaction: &Transaction<'_>,
    old: PullReplayTotals,
    new: PullReplayTotals,
) -> Result<(), AnonymousMailboxStoreError> {
    let affected = transaction
        .execute(
            "UPDATE anonymous_mailbox_meta
             SET pull_replay_rows = ?1, pull_replay_bytes = ?2
             WHERE singleton = 1 AND pull_replay_rows = ?3 AND pull_replay_bytes = ?4",
            params![
                as_i64(new.rows)?,
                as_i64(new.bytes)?,
                as_i64(old.rows)?,
                as_i64(old.bytes)?,
            ],
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    execute_exactly_one(transaction, affected)
}

pub(super) fn validate_totals_limits(
    totals: &StoreTotals,
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let maximum_leases =
        u64::try_from(config.max_leases_total).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let maximum_items =
        u64::try_from(config.max_items_total).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if totals.leases > maximum_leases
        || totals.items > maximum_items
        || totals.bytes > config.max_bytes_total
        || (totals.leases == 0 && (totals.items != 0 || totals.bytes != 0))
        || (totals.items == 0) != (totals.bytes == 0)
        || totals.bytes < totals.items
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

pub(super) fn update_totals_exact(
    transaction: &Transaction<'_>,
    old: StoreTotals,
    new: StoreTotals,
) -> Result<(), AnonymousMailboxStoreError> {
    let affected = transaction
        .execute(
            "UPDATE anonymous_mailbox_meta
             SET total_leases = ?1, total_items = ?2, total_bytes = ?3
             WHERE singleton = 1 AND total_leases = ?4
               AND total_items = ?5 AND total_bytes = ?6",
            params![
                as_i64(new.leases)?,
                as_i64(new.items)?,
                as_i64(new.bytes)?,
                as_i64(old.leases)?,
                as_i64(old.items)?,
                as_i64(old.bytes)?,
            ],
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    execute_exactly_one(transaction, affected)
}

pub(super) fn verify_totals_exact(
    transaction: &Transaction<'_>,
    expected: StoreTotals,
) -> Result<(), AnonymousMailboxStoreError> {
    if load_totals(transaction)? != expected {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

pub(super) fn execute_exactly_one(
    _transaction: &Transaction<'_>,
    affected: usize,
) -> Result<(), AnonymousMailboxStoreError> {
    if affected != 1 {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

pub(super) fn audit_counters(
    transaction: &Transaction<'_>,
    target_node_id: &[u8; 32],
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    #[cfg(test)]
    FULL_AUDIT_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));
    let totals = load_totals(transaction)?;

    // [ANONYMOUS-MAILBOX-STORE 2026-09-02 by Codex] Recompute every
    // aggregate with checked Rust arithmetic. SQLite SUM overflow and stale
    // counters are both corruption; cleanup must never turn either into an
    // implicit repair.
    let mut leases = HashMap::new();
    let mut lease_rows = transaction
        .prepare(
            "SELECT mailbox_id, current_items, current_bytes, next_sequence
             FROM anonymous_mailbox_leases",
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut lease_query = lease_rows
        .query([])
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut lease_count = 0_u64;
    while let Some(row) = lease_query
        .next()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        let mailbox_id = fixed::<32>(
            &row.get::<_, Vec<u8>>(0)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let audit = LeaseCounterAudit {
            stored_items: as_u64(
                row.get::<_, i64>(1)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            )?,
            stored_bytes: as_u64(
                row.get::<_, i64>(2)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            )?,
            next_sequence: as_u64(
                row.get::<_, i64>(3)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            )?,
            observed_items: 0,
            observed_bytes: 0,
            observed_max_sequence: 0,
        };
        if audit.next_sequence == 0 || leases.insert(mailbox_id, audit).is_some() {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        lease_count = lease_count
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
    }
    drop(lease_query);
    drop(lease_rows);

    let mut item_rows = transaction
        .prepare(
            "SELECT mailbox_id, item_id, sequence, length(sealed_envelope)
             FROM anonymous_mailbox_items",
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut item_query = item_rows
        .query([])
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut item_count = 0_u64;
    let mut byte_count = 0_u64;
    let mut item_keys = Vec::new();
    let maximum_item_bytes = u64::try_from(MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES)
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    while let Some(row) = item_query
        .next()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        let mailbox_id = fixed::<32>(
            &row.get::<_, Vec<u8>>(0)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let item_id = fixed::<16>(
            &row.get::<_, Vec<u8>>(1)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let sequence = as_u64(
            row.get::<_, i64>(2)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let length = as_u64(
            row.get::<_, i64>(3)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        if sequence == 0 || length == 0 || length > maximum_item_bytes {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        let lease = leases
            .get_mut(&mailbox_id)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        lease.observed_items = lease
            .observed_items
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        lease.observed_bytes = lease
            .observed_bytes
            .checked_add(length)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        lease.observed_max_sequence = lease.observed_max_sequence.max(sequence);
        item_count = item_count
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        byte_count = byte_count
            .checked_add(length)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        item_keys.push((mailbox_id, item_id));
    }
    drop(item_query);
    drop(item_rows);

    if totals.leases != lease_count || totals.items != item_count || totals.bytes != byte_count {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    for lease in leases.values() {
        if lease.stored_items != lease.observed_items
            || lease.stored_bytes != lease.observed_bytes
            || lease.next_sequence <= lease.observed_max_sequence
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
    }
    for mailbox_id in leases.keys() {
        load_lease(transaction, mailbox_id)?.ok_or(AnonymousMailboxStoreError::Corrupt)?;
    }
    for (mailbox_id, item_id) in item_keys {
        load_item(transaction, &mailbox_id, &item_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
    }
    audit_pull_replays(transaction, config)?;
    audit_issued_tickets(transaction, target_node_id, config)?;
    Ok(())
}

pub(super) fn audit_pull_replays(
    transaction: &Transaction<'_>,
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let stored = load_pull_replay_totals(transaction)?;
    validate_pull_replay_totals(stored, config)?;
    let mut statement = transaction
        .prepare(
            "SELECT r.rowid, r.outcome, r.created_at, r.retain_until, l.expires_at,
                    r.item_expires_at,
                    typeof(r.request_commitment), length(r.request_commitment),
                    typeof(r.item_id), length(r.item_id),
                    typeof(r.sealed_commitment), length(r.sealed_commitment),
                    typeof(r.sealed_envelope), length(r.sealed_envelope),
                    typeof(r.cursor), length(r.cursor)
             FROM anonymous_mailbox_pull_replays r
             LEFT JOIN anonymous_mailbox_leases l ON l.mailbox_id = r.mailbox_id",
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut rows = statement
        .query([])
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let mut observed = PullReplayTotals { rows: 0, bytes: 0 };
    let mut item_rows = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        let rowid = row
            .get::<_, i64>(0)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        let outcome = row
            .get::<_, i64>(1)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        validate_pull_replay_retention(
            row.get::<_, i64>(2)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            row.get::<_, i64>(3)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            row.get::<_, Option<i64>>(4)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
            outcome,
            row.get::<_, Option<i64>>(5)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let bytes = PullReplayShape::from_row(row, 6)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            .checked_envelope_bytes(outcome)?;
        if outcome == 1 {
            item_rows.push((rowid, bytes));
        }
        observed.rows = observed
            .rows
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        observed.bytes = observed
            .bytes
            .checked_add(bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if observed.rows > stored.rows || observed.bytes > stored.bytes {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
    }
    drop(rows);
    drop(statement);
    if observed != stored {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    // The metadata scan admitted every BLOB length before the second pass.
    // A transaction snapshot keeps these reads tied to the audited rows.
    for (rowid, admitted_bytes) in item_rows {
        let (commitment, envelope): (Vec<u8>, Vec<u8>) = transaction
            .query_row(
                "SELECT sealed_commitment, sealed_envelope
                 FROM anonymous_mailbox_pull_replays WHERE rowid = ?1",
                params![rowid],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if u64::try_from(envelope.len()).map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            != admitted_bytes
            || <[u8; 32]>::from(Sha256::digest(&envelope)) != fixed::<32>(&commitment)?
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
    }
    Ok(())
}

pub(super) fn audit_issued_tickets(
    transaction: &Transaction<'_>,
    target_node_id: &[u8; 32],
    config: &AnonymousMailboxStoreConfig,
) -> Result<(), AnonymousMailboxStoreError> {
    let meta = load_ticket_issue_meta(transaction)?;
    if meta.outstanding
        > u64::try_from(config.max_outstanding_tickets)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        || meta.issues_in_window
            > u64::try_from(config.max_ticket_issues_per_window)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let mut statement = transaction
        .prepare("SELECT request_id FROM anonymous_mailbox_issued_tickets")
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let rows = statement
        .query_map([], |row| row.get::<_, Vec<u8>>(0))
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let request_ids = rows
        .map(|row| {
            let bytes = row.map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            fixed::<16>(&bytes)
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(statement);
    let mut outstanding = 0_u64;
    for request_id in request_ids {
        let record = load_issued_ticket_by_request(transaction, &request_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        // The configured difficulty may increase after issuance. Requiring
        // one bit here proves a target-bound non-zero work image while durable
        // issuance admission, not startup policy drift, remains authoritative.
        record
            .request
            .verify_for_target(target_node_id, record.request.issued_at, 1)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        validate_issued_ticket(&record, &record.request, target_node_id)?;
        if record.consumed_at.is_none() {
            outstanding = outstanding
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
    }
    if outstanding != meta.outstanding {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn count(
    transaction: &Transaction<'_>,
    sql: &str,
) -> Result<u64, AnonymousMailboxStoreError> {
    let raw: i64 = transaction
        .query_row(sql, [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    as_u64(raw)
}
