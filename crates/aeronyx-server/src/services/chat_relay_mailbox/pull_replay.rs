// [ARCH-SPLIT 2026-10-02]
// Durable PullOne replay journal load and insert.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn validate_pull_replay_retention(
    created_at: i64,
    retain_until: i64,
    lease_expires_at: Option<i64>,
    outcome: i64,
    item_expires_at: Option<i64>,
) -> Result<Option<u64>, AnonymousMailboxStoreError> {
    let created_at = as_u64(created_at)?;
    let retain_until = as_u64(retain_until)?;
    let lease_expires_at = as_u64(lease_expires_at.ok_or(AnonymousMailboxStoreError::Corrupt)?)?;
    let maximum = lease_expires_at.min(created_at.saturating_add(PULL_REPLAY_RETENTION_SECS));
    match (outcome, item_expires_at) {
        (0, None) if retain_until == maximum => Ok(Some(retain_until)),
        (1, Some(item_expires_at)) => {
            let item_expires_at = as_u64(item_expires_at)?;
            if item_expires_at < created_at
                || item_expires_at > lease_expires_at
                || retain_until != maximum.min(item_expires_at)
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            Ok(Some(retain_until))
        }
        // [ANONYMOUS-MAILBOX-ITEM-EXPIRY 2026-09-24 by Codex] V3 could
        // retain an Item after ACK removed its source row. Its signed expiry
        // cannot be reconstructed, so keep the bounded row for cleanup but
        // never authorize replay of its opaque ciphertext.
        (1, None) if retain_until == maximum => Ok(None),
        _ => Err(AnonymousMailboxStoreError::Corrupt),
    }
}

pub(super) fn load_pull_replay(
    transaction: &Transaction<'_>,
    request: &AnonymousMailboxPullOneV1,
    request_commitment: &[u8; 32],
    now: u64,
) -> Result<Option<AnonymousMailboxPullOutcome>, AnonymousMailboxStoreError> {
    let metadata = transaction
        .query_row(
            "SELECT r.rowid, r.outcome, r.created_at, r.retain_until, l.expires_at,
                    r.item_expires_at,
                    typeof(r.request_commitment), length(r.request_commitment),
                    typeof(r.item_id), length(r.item_id),
                    typeof(r.sealed_commitment), length(r.sealed_commitment),
                    typeof(r.sealed_envelope), length(r.sealed_envelope),
                    typeof(r.cursor), length(r.cursor)
             FROM anonymous_mailbox_pull_replays r
             LEFT JOIN anonymous_mailbox_leases l ON l.mailbox_id = r.mailbox_id
             WHERE r.mailbox_id = ?1 AND r.request_id = ?2",
            params![&request.mailbox_id[..], &request.request_id[..]],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                    PullReplayShape::from_row(row, 6)?,
                ))
            },
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let Some((rowid, outcome, created_at, retain_until, lease_expires_at, item_expires_at, shape)) =
        metadata
    else {
        return Ok(None);
    };
    if !shape.request.is_exact_blob(32) {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let stored_request: Vec<u8> = transaction
        .query_row(
            "SELECT request_commitment FROM anonymous_mailbox_pull_replays WHERE rowid = ?1",
            params![rowid],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if stored_request != request_commitment[..] {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    shape.checked_envelope_bytes(outcome)?;
    let retain_until = validate_pull_replay_retention(
        created_at,
        retain_until,
        lease_expires_at,
        outcome,
        item_expires_at,
    )?
    .ok_or(AnonymousMailboxStoreError::Rejected)?;
    // [ANONYMOUS-MAILBOX-PULL-BOUNDS 2026-09-24 by Codex] Expired exact
    // replays cannot outlive their lease or the 24-hour journal window merely
    // because cleanup has not reached their row yet.
    if now > retain_until {
        return Err(AnonymousMailboxStoreError::Rejected);
    }
    let stored = if outcome == 0 {
        StoredPullReplay::Empty
    } else {
        let (item_id, commitment, sealed_envelope, cursor): (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) =
            transaction
                .query_row(
                    "SELECT item_id, sealed_commitment, sealed_envelope, cursor
                     FROM anonymous_mailbox_pull_replays WHERE rowid = ?1",
                    params![rowid],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        StoredPullReplay::Item {
            item_id: fixed::<16>(&item_id)?,
            sealed_commitment: fixed::<32>(&commitment)?,
            sealed_envelope,
            cursor,
            item_expires_at: as_u64(item_expires_at.ok_or(AnonymousMailboxStoreError::Corrupt)?)?,
        }
    };
    match stored {
        StoredPullReplay::Empty => Ok(Some(AnonymousMailboxPullOutcome::Empty)),
        StoredPullReplay::Item {
            item_id,
            sealed_commitment,
            sealed_envelope,
            cursor,
            item_expires_at: _,
        } => {
            if sealed_envelope.is_empty()
                || sealed_envelope.len() > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
                || cursor.len() != CURSOR_BYTES
                || <[u8; 32]>::from(Sha256::digest(&sealed_envelope)) != sealed_commitment
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            let mut padded = vec![0_u8; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES];
            padded[..sealed_envelope.len()].copy_from_slice(&sealed_envelope);
            Ok(Some(AnonymousMailboxPullOutcome::Item(
                AnonymousMailboxPulledItem {
                    item_id,
                    sealed_commitment,
                    sealed_length: u32::try_from(sealed_envelope.len())
                        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
                    padded_sealed_envelope: padded,
                    cursor,
                },
            )))
        }
    }
}

pub(super) fn insert_pull_replay(
    transaction: &Transaction<'_>,
    config: &AnonymousMailboxStoreConfig,
    request: &AnonymousMailboxPullOneV1,
    request_commitment: &[u8; 32],
    stored: StoredPullReplay,
    now: u64,
    retain_until: u64,
) -> Result<(), AnonymousMailboxStoreError> {
    let old = load_pull_replay_totals(transaction)?;
    validate_pull_replay_totals(old, config)?;
    let bytes = match &stored {
        StoredPullReplay::Empty => 0,
        StoredPullReplay::Item {
            sealed_envelope, ..
        } => {
            u64::try_from(sealed_envelope.len()).map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        }
    };
    let new = PullReplayTotals {
        rows: old
            .rows
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?,
        bytes: old
            .bytes
            .checked_add(bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?,
    };
    if validate_pull_replay_totals(new, config).is_err() {
        return Err(AnonymousMailboxStoreError::Busy);
    }
    let (outcome, item_id, commitment, envelope, cursor, item_expires_at) = match stored {
        StoredPullReplay::Empty => (0_i64, None, None, None, None, None),
        StoredPullReplay::Item {
            item_id,
            sealed_commitment,
            sealed_envelope,
            cursor,
            item_expires_at,
        } => (
            1_i64,
            Some(item_id.to_vec()),
            Some(sealed_commitment.to_vec()),
            Some(sealed_envelope),
            Some(cursor),
            Some(as_i64(item_expires_at)?),
        ),
    };
    let lease_expires_at: i64 = transaction
        .query_row(
            "SELECT expires_at FROM anonymous_mailbox_leases WHERE mailbox_id = ?1",
            params![&request.mailbox_id[..]],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    validate_pull_replay_retention(
        as_i64(now)?,
        as_i64(retain_until)?,
        Some(lease_expires_at),
        outcome,
        item_expires_at,
    )?
    .ok_or(AnonymousMailboxStoreError::Corrupt)?;
    execute_exactly_one(
        transaction,
        transaction
            .execute(
                "INSERT INTO anonymous_mailbox_pull_replays
                 (mailbox_id, request_id, request_commitment, outcome, item_id,
                  sealed_commitment, sealed_envelope, cursor, created_at, retain_until,
                  item_expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    &request.mailbox_id[..],
                    &request.request_id[..],
                    &request_commitment[..],
                    outcome,
                    item_id,
                    commitment,
                    envelope,
                    cursor,
                    as_i64(now)?,
                    as_i64(retain_until)?,
                    item_expires_at,
                ],
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
    )?;
    update_pull_replay_totals_exact(transaction, old, new)?;
    if load_pull_replay_totals(transaction)? != new {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}
