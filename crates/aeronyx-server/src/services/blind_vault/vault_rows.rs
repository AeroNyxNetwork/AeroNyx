// [ARCH-SPLIT 2026-10-02]
// Schema init and the row loaders the service methods share.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn init_schema(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS blind_vault_leases (
            lease_id              BLOB PRIMARY KEY CHECK(length(lease_id) = 32),
            request_id            BLOB NOT NULL UNIQUE CHECK(length(request_id) = 16),
            write_verifying_key   BLOB NOT NULL CHECK(length(write_verifying_key) = 32),
            admin_verifying_key   BLOB NOT NULL CHECK(length(admin_verifying_key) = 32),
            read_capability_tag   BLOB NOT NULL CHECK(length(read_capability_tag) = 32),
            created_at_ms         INTEGER NOT NULL CHECK(created_at_ms >= 0),
            expires_at_ms         INTEGER NOT NULL CHECK(expires_at_ms > created_at_ms),
            object_count          INTEGER NOT NULL DEFAULT 0 CHECK(object_count >= 0),
            byte_count            INTEGER NOT NULL DEFAULT 0 CHECK(byte_count >= 0)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS blind_vault_objects (
            sequence              INTEGER PRIMARY KEY AUTOINCREMENT,
            lease_id              BLOB NOT NULL CHECK(length(lease_id) = 32),
            object_id             BLOB NOT NULL CHECK(length(object_id) = 32),
            request_id            BLOB NOT NULL CHECK(length(request_id) = 16),
            ciphertext            BLOB NOT NULL,
            ciphertext_commitment BLOB NOT NULL CHECK(length(ciphertext_commitment) = 32),
            created_at_ms         INTEGER NOT NULL CHECK(created_at_ms >= 0),
            expires_at_ms         INTEGER NOT NULL CHECK(expires_at_ms > created_at_ms),
            UNIQUE(lease_id, object_id),
            UNIQUE(lease_id, request_id),
            FOREIGN KEY(lease_id) REFERENCES blind_vault_leases(lease_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_blind_vault_objects_pull
          ON blind_vault_objects(lease_id, sequence, expires_at_ms);
        CREATE INDEX IF NOT EXISTS idx_blind_vault_objects_expiry
          ON blind_vault_objects(expires_at_ms, sequence);

        CREATE TABLE IF NOT EXISTS blind_vault_capacity_state (
            state_id                    INTEGER PRIMARY KEY CHECK(state_id = 1),
            committed_lease_count       INTEGER NOT NULL CHECK(committed_lease_count >= 0),
            committed_ciphertext_bytes  INTEGER NOT NULL CHECK(committed_ciphertext_bytes >= 0)
        );
        INSERT OR IGNORE INTO blind_vault_capacity_state
          (state_id, committed_lease_count, committed_ciphertext_bytes)
          VALUES (1, 0, 0);

        CREATE TRIGGER IF NOT EXISTS blind_vault_capacity_lease_insert
        AFTER INSERT ON blind_vault_leases
        BEGIN
          UPDATE blind_vault_capacity_state
          SET committed_lease_count = committed_lease_count + 1
          WHERE state_id = 1;
        END;
        CREATE TRIGGER IF NOT EXISTS blind_vault_capacity_lease_delete
        AFTER DELETE ON blind_vault_leases
        BEGIN
          UPDATE blind_vault_capacity_state
          SET committed_lease_count = committed_lease_count - 1
          WHERE state_id = 1;
        END;
        CREATE TRIGGER IF NOT EXISTS blind_vault_capacity_object_insert
        AFTER INSERT ON blind_vault_objects
        BEGIN
          UPDATE blind_vault_capacity_state
          SET committed_ciphertext_bytes = committed_ciphertext_bytes + length(NEW.ciphertext)
          WHERE state_id = 1;
        END;
        CREATE TRIGGER IF NOT EXISTS blind_vault_capacity_object_delete
        AFTER DELETE ON blind_vault_objects
        BEGIN
          UPDATE blind_vault_capacity_state
          SET committed_ciphertext_bytes = committed_ciphertext_bytes - length(OLD.ciphertext)
          WHERE state_id = 1;
        END;

        CREATE TABLE IF NOT EXISTS blind_vault_tombstones (
            lease_id              BLOB NOT NULL CHECK(length(lease_id) = 32),
            object_id             BLOB NOT NULL CHECK(length(object_id) = 32),
            ciphertext_commitment BLOB NOT NULL CHECK(length(ciphertext_commitment) = 32),
            deleted_at_ms         INTEGER NOT NULL CHECK(deleted_at_ms >= 0),
            expires_at_ms         INTEGER NOT NULL CHECK(expires_at_ms > deleted_at_ms),
            PRIMARY KEY(lease_id, object_id),
            FOREIGN KEY(lease_id) REFERENCES blind_vault_leases(lease_id) ON DELETE CASCADE
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_blind_vault_tombstones_expiry
          ON blind_vault_tombstones(expires_at_ms);

        CREATE TABLE IF NOT EXISTS blind_vault_lease_tombstones (
            lease_id                 BLOB PRIMARY KEY CHECK(length(lease_id) = 32),
            request_id               BLOB NOT NULL CHECK(length(request_id) = 16),
            admin_verifying_key      BLOB NOT NULL CHECK(length(admin_verifying_key) = 32),
            retired_at_ms            INTEGER NOT NULL CHECK(retired_at_ms >= 0),
            request_commitment       BLOB NOT NULL CHECK(length(request_commitment) = 32),
            deleted_object_count     INTEGER NOT NULL CHECK(deleted_object_count >= 0),
            deleted_ciphertext_bytes INTEGER NOT NULL CHECK(deleted_ciphertext_bytes >= 0),
            expires_at_ms            INTEGER NOT NULL CHECK(expires_at_ms > retired_at_ms)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_blind_vault_lease_tombstones_expiry
          ON blind_vault_lease_tombstones(expires_at_ms);

        CREATE TABLE IF NOT EXISTS blind_vault_lease_renewals (
            lease_id               BLOB NOT NULL CHECK(length(lease_id) = 32),
            request_id             BLOB NOT NULL CHECK(length(request_id) = 16),
            admission_spend_id     BLOB NOT NULL CHECK(length(admission_spend_id) = 32),
            request_commitment     BLOB NOT NULL CHECK(length(request_commitment) = 32),
            previous_expires_at_ms INTEGER NOT NULL CHECK(previous_expires_at_ms >= 0),
            renewed_expires_at_ms  INTEGER NOT NULL CHECK(renewed_expires_at_ms > previous_expires_at_ms),
            renewed_at_ms          INTEGER NOT NULL CHECK(renewed_at_ms < previous_expires_at_ms),
            expires_at_ms          INTEGER NOT NULL CHECK(expires_at_ms > renewed_at_ms),
            PRIMARY KEY(lease_id, request_id),
            FOREIGN KEY(lease_id) REFERENCES blind_vault_leases(lease_id) ON DELETE CASCADE
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_blind_vault_lease_renewals_expiry
          ON blind_vault_lease_renewals(expires_at_ms);

        CREATE TABLE IF NOT EXISTS blind_vault_admission_spends (
            token_id       BLOB PRIMARY KEY CHECK(length(token_id) = 32),
            consumed_at_ms INTEGER NOT NULL CHECK(consumed_at_ms >= 0),
            expires_at_ms  INTEGER NOT NULL CHECK(expires_at_ms > consumed_at_ms)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_blind_vault_admission_spends_expiry
          ON blind_vault_admission_spends(expires_at_ms);

        CREATE TABLE IF NOT EXISTS blind_vault_blind_issuer_state (
            state_id      INTEGER PRIMARY KEY CHECK(state_id = 1),
            generation    INTEGER NOT NULL CHECK(generation > 0),
            digest        BLOB NOT NULL CHECK(length(digest) = 32),
            updated_at_ms INTEGER NOT NULL CHECK(updated_at_ms >= 0)
        );

        CREATE TABLE IF NOT EXISTS blind_vault_blind_issuer_epochs (
            issuer_key_id       BLOB PRIMARY KEY CHECK(length(issuer_key_id) = 32),
            admission_version   INTEGER NOT NULL CHECK(admission_version > 0),
            public_key_der      BLOB NOT NULL,
            not_before_ms       INTEGER NOT NULL CHECK(not_before_ms >= 0),
            expires_at_ms       INTEGER NOT NULL CHECK(expires_at_ms > not_before_ms),
            max_lease_ttl_ms    INTEGER NOT NULL CHECK(max_lease_ttl_ms > 0)
        ) WITHOUT ROWID;

        -- [BLIND-VAULT-NODE-CAPACITY 2026-08-28 by Codex] Rebuild the compact
        -- ledger from durable truth on every process start. Runtime mutations
        -- then stay O(1) and transactionally coherent through triggers,
        -- including foreign-key cascades during lease cleanup or retirement.
        UPDATE blind_vault_capacity_state
        SET committed_lease_count = (SELECT COUNT(*) FROM blind_vault_leases),
            committed_ciphertext_bytes =
              (SELECT COALESCE(SUM(length(ciphertext)), 0) FROM blind_vault_objects)
        WHERE state_id = 1;",
    )
}

pub(super) fn existing_lease_outcome(
    transaction: &Transaction<'_>,
    request: &BlindVaultLeaseCreateRequest,
    read_capability_tag: &[u8; 32],
) -> Result<Option<BlindVaultLeaseProvisionOutcome>, BlindVaultServiceError> {
    let Some(existing) = load_lease_provisioning(transaction, &request.lease_id)? else {
        return Ok(None);
    };
    let exact = existing.request_id == request.request_id
        && existing.write_verifying_key == request.write_verifying_key
        && existing.admin_verifying_key == request.admin_verifying_key
        && existing.read_capability_tag == *read_capability_tag
        && existing.expires_at_ms == request.expires_at_ms;
    if exact {
        Ok(Some(BlindVaultLeaseProvisionOutcome::Existing))
    } else {
        Err(BlindVaultServiceError::LeaseConflict)
    }
}

pub(super) fn ensure_lease_request_available(
    transaction: &Transaction<'_>,
    request_id: &[u8; 16],
) -> Result<(), BlindVaultServiceError> {
    let conflict: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM blind_vault_leases WHERE request_id = ?1)",
        params![&request_id[..]],
        |row| row.get(0),
    )?;
    if conflict {
        Err(BlindVaultServiceError::RequestConflict)
    } else {
        Ok(())
    }
}

pub(super) fn ensure_node_admission_capacity(
    transaction: &Transaction<'_>,
    max_live_leases: u64,
    max_total_ciphertext_bytes: u64,
) -> Result<(), BlindVaultServiceError> {
    // [BLIND-VAULT-NODE-CAPACITY 2026-08-28 by Codex] Read one trigger-backed
    // singleton instead of scanning every lease on each anonymous admission.
    let (committed_leases, committed_bytes): (i64, i64) = transaction.query_row(
        "SELECT committed_lease_count, committed_ciphertext_bytes
         FROM blind_vault_capacity_state WHERE state_id = 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if non_negative_u64(committed_leases)? >= max_live_leases
        || non_negative_u64(committed_bytes)? >= max_total_ciphertext_bytes
    {
        return Err(BlindVaultServiceError::NodeCapacityExceeded);
    }
    Ok(())
}

pub(super) fn ensure_node_ciphertext_capacity(
    transaction: &Transaction<'_>,
    additional_bytes: u64,
    max_total_ciphertext_bytes: u64,
) -> Result<(), BlindVaultServiceError> {
    // [BLIND-VAULT-NODE-CAPACITY 2026-08-28 by Codex] The immediate writer
    // transaction makes this O(1) check and the following insert indivisible.
    let committed_bytes: i64 = transaction.query_row(
        "SELECT committed_ciphertext_bytes
         FROM blind_vault_capacity_state WHERE state_id = 1",
        [],
        |row| row.get(0),
    )?;
    let next_bytes = non_negative_u64(committed_bytes)?
        .checked_add(additional_bytes)
        .ok_or(BlindVaultServiceError::NodeCapacityExceeded)?;
    if next_bytes > max_total_ciphertext_bytes {
        return Err(BlindVaultServiceError::NodeCapacityExceeded);
    }
    Ok(())
}

pub(super) fn insert_lease_row(
    transaction: &Transaction<'_>,
    request: &BlindVaultLeaseCreateRequest,
    read_capability_tag: &[u8; 32],
    created_at_ms: i64,
    expires_at_ms: i64,
) -> Result<(), BlindVaultServiceError> {
    transaction.execute(
        "INSERT INTO blind_vault_leases
         (lease_id, request_id, write_verifying_key, admin_verifying_key,
          read_capability_tag, created_at_ms, expires_at_ms, object_count, byte_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 0)",
        params![
            &request.lease_id[..],
            &request.request_id[..],
            &request.write_verifying_key[..],
            &request.admin_verifying_key[..],
            &read_capability_tag[..],
            created_at_ms,
            expires_at_ms,
        ],
    )?;
    Ok(())
}

pub(super) fn load_lease_provisioning(
    transaction: &Transaction<'_>,
    lease_id: &[u8; 32],
) -> Result<Option<LeaseProvisioningRow>, BlindVaultServiceError> {
    let row: Option<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, i64)> = transaction
        .query_row(
            "SELECT request_id, write_verifying_key, admin_verifying_key,
                    read_capability_tag, expires_at_ms
             FROM blind_vault_leases WHERE lease_id = ?1",
            params![&lease_id[..]],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()?;
    row.map(|(request_id, write, admin, read_tag, expires)| {
        Ok(LeaseProvisioningRow {
            request_id: fixed_array(&request_id)?,
            write_verifying_key: fixed_array(&write)?,
            admin_verifying_key: fixed_array(&admin)?,
            read_capability_tag: fixed_array(&read_tag)?,
            expires_at_ms: u64::try_from(expires)
                .map_err(|_| BlindVaultServiceError::CorruptState)?,
        })
    })
    .transpose()
}

pub(super) fn load_lease_runtime(
    connection: &Connection,
    lease_id: &[u8; 32],
) -> Result<Option<LeaseRuntimeRow>, BlindVaultServiceError> {
    let row: Option<(Vec<u8>, Vec<u8>, i64, i64, i64)> = connection
        .query_row(
            "SELECT write_verifying_key, admin_verifying_key, expires_at_ms,
                    object_count, byte_count
             FROM blind_vault_leases WHERE lease_id = ?1",
            params![&lease_id[..]],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()?;
    row.map(|(write, admin, expires, count, bytes)| {
        Ok(LeaseRuntimeRow {
            write_verifying_key: fixed_array(&write)?,
            admin_verifying_key: fixed_array(&admin)?,
            expires_at_ms: non_negative_u64(expires)?,
            object_count: non_negative_u64(count)?,
            byte_count: non_negative_u64(bytes)?,
        })
    })
    .transpose()
}

pub(super) fn load_replica_job_authority(
    connection: &Connection,
    lease_id: &[u8; 32],
    object_id: &[u8; 32],
) -> Result<Option<ReplicaJobAuthorityRow>, BlindVaultReplicaJobAuthorizationError> {
    // [BLIND-VAULT-REPLICA-AUTH 2026-09-01 by Codex] Keep this projection
    // intentionally narrow: ciphertext bytes, read capabilities, request IDs,
    // usage counters, and unrelated lease metadata never cross this boundary.
    let row: Option<(Vec<u8>, i64, Vec<u8>, i64)> = connection
        .query_row(
            "SELECT l.admin_verifying_key, l.expires_at_ms,
                    o.ciphertext_commitment, o.expires_at_ms
             FROM blind_vault_leases l
             JOIN blind_vault_objects o ON o.lease_id = l.lease_id
             WHERE l.lease_id = ?1 AND o.object_id = ?2",
            params![&lease_id[..], &object_id[..]],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()
        .map_err(|_| BlindVaultReplicaJobAuthorizationError::Unavailable)?;
    row.map(|(admin, lease_expires, commitment, object_expires)| {
        Ok(ReplicaJobAuthorityRow {
            admin_verifying_key: admin
                .as_slice()
                .try_into()
                .map_err(|_| BlindVaultReplicaJobAuthorizationError::Unavailable)?,
            lease_expires_at_ms: u64::try_from(lease_expires)
                .map_err(|_| BlindVaultReplicaJobAuthorizationError::Unavailable)?,
            ciphertext_commitment: commitment
                .as_slice()
                .try_into()
                .map_err(|_| BlindVaultReplicaJobAuthorizationError::Unavailable)?,
            object_expires_at_ms: u64::try_from(object_expires)
                .map_err(|_| BlindVaultReplicaJobAuthorizationError::Unavailable)?,
        })
    })
    .transpose()
}

pub(super) fn load_live_lease_status(
    connection: &Connection,
    lease_id: &[u8; 32],
    now_ms: u64,
) -> Result<Option<LeaseStatusObservation>, BlindVaultServiceError> {
    let now = sqlite_i64(now_ms)?;
    let row: Option<(Vec<u8>, i64, i64, i64)> = connection
        .query_row(
            "SELECT l.admin_verifying_key, l.expires_at_ms,
                    COUNT(o.sequence), COALESCE(SUM(length(o.ciphertext)), 0)
             FROM blind_vault_leases AS l
             LEFT JOIN blind_vault_objects AS o
               ON o.lease_id = l.lease_id AND o.expires_at_ms > ?2
             WHERE l.lease_id = ?1
             GROUP BY l.lease_id, l.admin_verifying_key, l.expires_at_ms",
            params![&lease_id[..], now],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    row.map(|(admin_key, expires_at, object_count, ciphertext_bytes)| {
        Ok(LeaseStatusObservation {
            admin_verifying_key: fixed_array(&admin_key)?,
            expires_at_ms: non_negative_u64(expires_at)?,
            live_object_count: non_negative_u64(object_count)?,
            live_ciphertext_bytes: non_negative_u64(ciphertext_bytes)?,
        })
    })
    .transpose()
}

pub(super) fn load_live_lease_inventory(
    connection: &Connection,
    lease_id: &[u8; 32],
    now_ms: u64,
    lease_expires_at_ms: u64,
) -> Result<BlindVaultInventoryCommitmentSummary, BlindVaultServiceError> {
    let now = sqlite_i64(now_ms)?;
    let mut statement = connection.prepare(
        "SELECT object_id, ciphertext_commitment, expires_at_ms, length(ciphertext)
         FROM blind_vault_objects
         WHERE lease_id = ?1 AND expires_at_ms > ?2
         ORDER BY object_id ASC",
    )?;
    let rows = statement.query_map(params![&lease_id[..], now], |row| {
        Ok((
            row.get::<_, Vec<u8>>(0)?,
            row.get::<_, Vec<u8>>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
        ))
    })?;
    let mut builder = BlindVaultInventoryCommitmentBuilder::new(*lease_id)
        .map_err(|_| BlindVaultServiceError::CorruptState)?;
    for row in rows {
        let (object_id, commitment, expires_at, ciphertext_bytes) = row?;
        let expires_at_ms = non_negative_u64(expires_at)?;
        if expires_at_ms > lease_expires_at_ms {
            return Err(BlindVaultServiceError::CorruptState);
        }
        builder
            .push(BlindVaultInventoryCommitmentEntry {
                object_id: fixed_array(&object_id)?,
                ciphertext_commitment: fixed_array(&commitment)?,
                expires_at_ms,
                ciphertext_bytes: non_negative_u64(ciphertext_bytes)?,
            })
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
    }
    Ok(builder.finish())
}

pub(super) fn load_lease_object_usage(
    connection: &Connection,
    lease_id: &[u8; 32],
) -> Result<LeaseObjectUsage, BlindVaultServiceError> {
    let (object_count, ciphertext_bytes): (i64, i64) = connection.query_row(
        "SELECT COUNT(*), COALESCE(SUM(length(ciphertext)), 0)
         FROM blind_vault_objects WHERE lease_id = ?1",
        params![&lease_id[..]],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok(LeaseObjectUsage {
        object_count: non_negative_u64(object_count)?,
        ciphertext_bytes: non_negative_u64(ciphertext_bytes)?,
    })
}

pub(super) fn load_active_lease_retirement(
    connection: &Connection,
    lease_id: &[u8; 32],
    now_ms: u64,
) -> Result<Option<LeaseRetirementRow>, BlindVaultServiceError> {
    let now = sqlite_i64(now_ms)?;
    let row: Option<(Vec<u8>, Vec<u8>, Vec<u8>, i64, i64, i64)> = connection
        .query_row(
            "SELECT request_id, admin_verifying_key, request_commitment, retired_at_ms,
                    deleted_object_count, deleted_ciphertext_bytes
             FROM blind_vault_lease_tombstones
             WHERE lease_id = ?1 AND expires_at_ms > ?2",
            params![&lease_id[..], now],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            },
        )
        .optional()?;
    row.map(
        |(
            request_id,
            admin_key,
            request_commitment,
            retired_at,
            object_count,
            ciphertext_bytes,
        )| {
            Ok(LeaseRetirementRow {
                request_id: fixed_array(&request_id)?,
                admin_verifying_key: fixed_array(&admin_key)?,
                request_commitment: fixed_array(&request_commitment)?,
                retired_at_ms: non_negative_u64(retired_at)?,
                deleted_object_count: non_negative_u64(object_count)?,
                deleted_ciphertext_bytes: non_negative_u64(ciphertext_bytes)?,
            })
        },
    )
    .transpose()
}

pub(super) fn load_active_lease_renewal(
    connection: &Connection,
    lease_id: &[u8; 32],
    request_id: &[u8; 16],
    now_ms: u64,
) -> Result<Option<LeaseRenewalRow>, BlindVaultServiceError> {
    let now = sqlite_i64(now_ms)?;
    let row: Option<(Vec<u8>, Vec<u8>, i64, i64, i64)> = connection
        .query_row(
            "SELECT admission_spend_id, request_commitment,
                    previous_expires_at_ms, renewed_expires_at_ms, renewed_at_ms
             FROM blind_vault_lease_renewals
             WHERE lease_id = ?1 AND request_id = ?2 AND expires_at_ms > ?3",
            params![&lease_id[..], &request_id[..], now],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()?;
    row.map(
        |(spend_id, commitment, previous_expires, renewed_expires, renewed_at)| {
            Ok(LeaseRenewalRow {
                admission_spend_id: fixed_array(&spend_id)?,
                request_commitment: fixed_array(&commitment)?,
                previous_expires_at_ms: non_negative_u64(previous_expires)?,
                renewed_expires_at_ms: non_negative_u64(renewed_expires)?,
                renewed_at_ms: non_negative_u64(renewed_at)?,
            })
        },
    )
    .transpose()
}

pub(super) fn ensure_exact_lease_renewal(
    existing: &LeaseRenewalRow,
    request: &BlindVaultBlindLeaseRenewalRequest,
) -> Result<(), BlindVaultServiceError> {
    if existing.admission_spend_id != request.admission.spend_id()
        || existing.request_commitment != request.renewal.commitment()
        || existing.previous_expires_at_ms != request.renewal.expected_expires_at_ms
        || existing.renewed_expires_at_ms != request.renewal.requested_expires_at_ms
    {
        return Err(BlindVaultServiceError::RequestConflict);
    }
    Ok(())
}

pub(super) fn load_existing_object(
    transaction: &Transaction<'_>,
    request: &BlindVaultPutRequest,
) -> Result<Option<ExistingObjectRow>, BlindVaultServiceError> {
    let row: Option<(Vec<u8>, Vec<u8>, i64, i64)> = transaction
        .query_row(
            "SELECT request_id, ciphertext_commitment, created_at_ms, expires_at_ms
             FROM blind_vault_objects
             WHERE lease_id = ?1 AND object_id = ?2",
            params![&request.lease_id[..], &request.object_id[..]],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    row.map(|(request_id, commitment, created, expires)| {
        Ok(ExistingObjectRow {
            request_id: fixed_array(&request_id)?,
            ciphertext_commitment: fixed_array(&commitment)?,
            created_at_ms: non_negative_u64(created)?,
            expires_at_ms: non_negative_u64(expires)?,
        })
    })
    .transpose()
}

pub(super) fn select_expired_objects(
    transaction: &Transaction<'_>,
    now: i64,
) -> Result<Vec<ExpiredObjectRow>, BlindVaultServiceError> {
    let mut statement = transaction.prepare(
        "SELECT sequence, lease_id, object_id, ciphertext_commitment, length(ciphertext)
         FROM blind_vault_objects WHERE expires_at_ms <= ?1
         ORDER BY expires_at_ms ASC, sequence ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![now, CLEANUP_OBJECT_BATCH as i64], |row| {
        Ok(ExpiredObjectRow {
            sequence: row.get(0)?,
            lease_id: row.get(1)?,
            object_id: row.get(2)?,
            ciphertext_commitment: row.get(3)?,
            ciphertext_bytes: row.get(4)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub(super) fn select_expired_leases(
    transaction: &Transaction<'_>,
    now: i64,
) -> Result<Vec<Vec<u8>>, BlindVaultServiceError> {
    let mut statement = transaction.prepare(
        "SELECT lease_id FROM blind_vault_leases
         WHERE expires_at_ms <= ?1 ORDER BY expires_at_ms ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![now, CLEANUP_LEASE_BATCH as i64], |row| row.get(0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub(super) fn fixed_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], BlindVaultServiceError> {
    bytes
        .try_into()
        .map_err(|_| BlindVaultServiceError::CorruptState)
}

pub(super) fn sqlite_i64(value: u64) -> Result<i64, BlindVaultServiceError> {
    i64::try_from(value).map_err(|_| BlindVaultServiceError::TimestampOutOfRange)
}

pub(super) fn non_negative_u64(value: i64) -> Result<u64, BlindVaultServiceError> {
    u64::try_from(value).map_err(|_| BlindVaultServiceError::CorruptState)
}
