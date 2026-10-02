// [ARCH-SPLIT 2026-10-02]
// Lease, item, and ACK row loads shared by the custody operations.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn load_lease(
    transaction: &Transaction<'_>,
    mailbox_id: &[u8; 32],
) -> Result<Option<AnonymousMailboxLeaseProjection>, AnonymousMailboxStoreError> {
    let row = transaction
        .query_row(
            "SELECT mailbox_id, ticket_id, claims_commitment, deposit_verifier, read_verifier,
                    max_items, max_bytes, current_items, current_bytes, next_sequence,
                    created_at, expires_at
             FROM anonymous_mailbox_leases WHERE mailbox_id = ?1",
            params![&mailbox_id[..]],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                ))
            },
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let ticket_id = fixed::<16>(&row.1)?;
    let stored_claims = fixed::<32>(&row.2)?;
    let lease = AnonymousMailboxLeaseProjection {
        mailbox_id: fixed::<32>(&row.0)?,
        deposit_verifier: fixed::<32>(&row.3)?,
        read_verifier: fixed::<32>(&row.4)?,
        max_items: u16::try_from(as_u64(row.5)?)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        max_bytes: as_u64(row.6)?,
        current_items: u16::try_from(as_u64(row.7)?)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        current_bytes: as_u64(row.8)?,
        next_sequence: as_u64(row.9)?,
        created_at: as_u64(row.10)?,
        expires_at: as_u64(row.11)?,
    };
    let calculated_claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
        &lease.mailbox_id,
        &lease.deposit_verifier,
        &lease.read_verifier,
        lease.max_items,
        lease.max_bytes,
        lease.created_at,
        lease.expires_at,
    );
    if stored_claims != calculated_claims {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let ticket_link = transaction
        .query_row(
            "SELECT mailbox_id, claims_commitment FROM anonymous_mailbox_tickets
             WHERE ticket_id = ?1",
            params![&ticket_id[..]],
            |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
        .ok_or(AnonymousMailboxStoreError::Corrupt)?;
    if fixed::<32>(&ticket_link.0)? != lease.mailbox_id
        || fixed::<32>(&ticket_link.1)? != stored_claims
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(Some(lease))
}

pub(super) fn load_item(
    transaction: &Transaction<'_>,
    mailbox_id: &[u8; 32],
    item_id: &[u8; 16],
) -> Result<Option<AnonymousMailboxItemProjection>, AnonymousMailboxStoreError> {
    let metadata = transaction
        .query_row(
            "SELECT mailbox_id, item_id, sequence, sealed_commitment,
                    length(sealed_envelope), stored_at, expires_at
             FROM anonymous_mailbox_items WHERE mailbox_id = ?1 AND item_id = ?2",
            params![&mailbox_id[..], &item_id[..]],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            },
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let admitted_length =
        usize::try_from(metadata.4).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    if admitted_length == 0 || admitted_length > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let sealed_envelope: Vec<u8> = transaction
        .query_row(
            "SELECT sealed_envelope FROM anonymous_mailbox_items
             WHERE mailbox_id = ?1 AND item_id = ?2",
            params![&mailbox_id[..], &item_id[..]],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    if sealed_envelope.len() != admitted_length {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    let sealed_commitment = fixed::<32>(&metadata.3)?;
    let calculated_commitment: [u8; 32] = Sha256::digest(&sealed_envelope).into();
    if sealed_commitment != calculated_commitment {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(Some(AnonymousMailboxItemProjection {
        mailbox_id: fixed::<32>(&metadata.0)?,
        item_id: fixed::<16>(&metadata.1)?,
        sequence: as_u64(metadata.2)?,
        sealed_commitment,
        sealed_envelope,
        stored_at: as_u64(metadata.5)?,
        expires_at: as_u64(metadata.6)?,
    }))
}

pub(super) fn load_ack_commitment(
    transaction: &Transaction<'_>,
    mailbox_id: &[u8; 32],
    item_id: &[u8; 16],
) -> Result<Option<[u8; 32]>, AnonymousMailboxStoreError> {
    transaction
        .query_row(
            "SELECT sealed_commitment FROM anonymous_mailbox_acks
             WHERE mailbox_id = ?1 AND item_id = ?2",
            params![&mailbox_id[..], &item_id[..]],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        .map(|bytes| fixed::<32>(&bytes))
        .transpose()
}

pub(super) fn select_expired_items(
    transaction: &Transaction<'_>,
    now: u64,
    limit: u64,
) -> Result<Vec<(Vec<u8>, Vec<u8>, u64)>, AnonymousMailboxStoreError> {
    let mut statement = transaction
        .prepare(
            "SELECT mailbox_id, item_id, length(sealed_envelope)
             FROM anonymous_mailbox_items WHERE expires_at < ?1
             ORDER BY expires_at, mailbox_id, item_id LIMIT ?2",
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let rows = statement
        .query_map(params![as_i64(now)?, as_i64(limit)?], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let mut result = Vec::new();
    for row in rows {
        let (mailbox_id, item_id, bytes) =
            row.map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let bytes = as_u64(bytes)?;
        if bytes == 0 || bytes > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES as u64 {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        result.push((mailbox_id, item_id, bytes));
    }
    Ok(result)
}

pub(super) fn as_i64(value: u64) -> Result<i64, AnonymousMailboxStoreError> {
    i64::try_from(value).map_err(|_| AnonymousMailboxStoreError::Rejected)
}

pub(super) fn as_u64(value: i64) -> Result<u64, AnonymousMailboxStoreError> {
    u64::try_from(value).map_err(|_| AnonymousMailboxStoreError::Corrupt)
}

pub(super) fn fixed<const N: usize>(bytes: &[u8]) -> Result<[u8; N], AnonymousMailboxStoreError> {
    bytes
        .try_into()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)
}
