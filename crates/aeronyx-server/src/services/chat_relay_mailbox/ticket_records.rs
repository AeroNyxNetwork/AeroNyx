// [ARCH-SPLIT 2026-10-02]
// Issued-ticket lookup, validation, and expiry purge.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn load_ticket_issue_meta(
    transaction: &Transaction<'_>,
) -> Result<TicketIssueMeta, AnonymousMailboxStoreError> {
    let (outstanding, window_started_at, issues_in_window): (i64, i64, i64) = transaction
        .query_row(
            "SELECT outstanding_tickets, issuance_window_started_at, issues_in_window
             FROM anonymous_mailbox_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    Ok(TicketIssueMeta {
        outstanding: as_u64(outstanding)?,
        window_started_at: as_u64(window_started_at)?,
        issues_in_window: as_u64(issues_in_window)?,
    })
}

pub(super) fn update_ticket_issue_meta_exact(
    transaction: &Transaction<'_>,
    old: TicketIssueMeta,
    new: TicketIssueMeta,
) -> Result<(), AnonymousMailboxStoreError> {
    let affected = transaction
        .execute(
            "UPDATE anonymous_mailbox_meta
             SET outstanding_tickets = ?1, issuance_window_started_at = ?2, issues_in_window = ?3
             WHERE singleton = 1 AND outstanding_tickets = ?4
               AND issuance_window_started_at = ?5 AND issues_in_window = ?6",
            params![
                as_i64(new.outstanding)?,
                as_i64(new.window_started_at)?,
                as_i64(new.issues_in_window)?,
                as_i64(old.outstanding)?,
                as_i64(old.window_started_at)?,
                as_i64(old.issues_in_window)?,
            ],
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    execute_exactly_one(transaction, affected)
}

pub(super) fn verify_ticket_issue_meta(
    transaction: &Transaction<'_>,
    expected: TicketIssueMeta,
) -> Result<(), AnonymousMailboxStoreError> {
    if load_ticket_issue_meta(transaction)? != expected {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    Ok(())
}

pub(super) fn load_issued_ticket_by_request(
    transaction: &Transaction<'_>,
    request_id: &[u8; 16],
) -> Result<Option<IssuedTicketRecord>, AnonymousMailboxStoreError> {
    load_issued_ticket(transaction, "request_id = ?1", params![&request_id[..]])
}

pub(super) fn load_issued_ticket_by_ticket(
    transaction: &Transaction<'_>,
    ticket_id: &[u8; 16],
) -> Result<Option<IssuedTicketRecord>, AnonymousMailboxStoreError> {
    load_issued_ticket(transaction, "ticket_id = ?1", params![&ticket_id[..]])
}

pub(super) fn load_issued_ticket<P>(
    transaction: &Transaction<'_>,
    predicate: &str,
    params: P,
) -> Result<Option<IssuedTicketRecord>, AnonymousMailboxStoreError>
where
    P: rusqlite::Params,
{
    let sql = format!(
        "SELECT request_id, ticket_id, request_commitment, target_node_id,
                claims_commitment, requested_at, expires_at, proof_nonce,
                ticket_commitment, ticket_signature, consumed_at
         FROM anonymous_mailbox_issued_tickets WHERE {predicate}"
    );
    let row = transaction
        .query_row(&sql, params, |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, Vec<u8>>(7)?,
                row.get::<_, Vec<u8>>(8)?,
                row.get::<_, Vec<u8>>(9)?,
                row.get::<_, Option<i64>>(10)?,
            ))
        })
        .optional()
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let request = AnonymousMailboxTicketIssueV1 {
        version: aeronyx_core::protocol::anonymous_mailbox::ANONYMOUS_MAILBOX_VERSION_V1,
        request_id: fixed::<16>(&row.0)?,
        ticket_id: fixed::<16>(&row.1)?,
        target_node_id: fixed::<32>(&row.3)?,
        lease_claims_commitment: fixed::<32>(&row.4)?,
        issued_at: as_u64(row.5)?,
        expires_at: as_u64(row.6)?,
        proof_nonce: u64::from_le_bytes(fixed::<8>(&row.7)?),
    };
    let ticket = AnonymousMailboxAdmissionTicketV1 {
        version: aeronyx_core::protocol::anonymous_mailbox::ANONYMOUS_MAILBOX_VERSION_V1,
        ticket_id: request.ticket_id,
        target_node_id: request.target_node_id,
        lease_claims_commitment: request.lease_claims_commitment,
        issued_at: request.issued_at,
        expires_at: request.expires_at,
        signature: fixed::<64>(&row.9)?,
    };
    Ok(Some(IssuedTicketRecord {
        request_id: request.request_id,
        request_commitment: fixed::<32>(&row.2)?,
        request,
        ticket,
        ticket_commitment: fixed::<32>(&row.8)?,
        consumed_at: row.10.map(as_u64).transpose()?,
    }))
}

pub(super) fn validate_issued_ticket(
    record: &IssuedTicketRecord,
    request: &AnonymousMailboxTicketIssueV1,
    target_node_id: &[u8; 32],
) -> Result<(), AnonymousMailboxStoreError> {
    if record.request_id != request.request_id
        || record.request != *request
        || record.ticket.target_node_id != *target_node_id
        || record
            .ticket
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            != record.ticket_commitment
    {
        return Err(AnonymousMailboxStoreError::Corrupt);
    }
    record
        .ticket
        .verify_at(
            target_node_id,
            &request.lease_claims_commitment,
            request.issued_at,
        )
        .map_err(|_| AnonymousMailboxStoreError::Corrupt)
}

pub(super) fn purge_expired_issued_tickets(
    transaction: &Transaction<'_>,
    now: u64,
    limit: u64,
) -> Result<ExpiredIssuedTicketPurge, AnonymousMailboxStoreError> {
    let mut statement = transaction
        .prepare(
            "SELECT request_id, consumed_at FROM anonymous_mailbox_issued_tickets
             WHERE expires_at < ?1 ORDER BY expires_at, request_id LIMIT ?2",
        )
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let rows = statement
        .query_map(params![as_i64(now)?, as_i64(limit)?], |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Option<i64>>(1)?))
        })
        .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
    let mut result = ExpiredIssuedTicketPurge::default();
    for row in rows {
        let (request_id, consumed_at) = row.map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let request_id = fixed::<16>(&request_id)?;
        execute_exactly_one(
            transaction,
            transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_issued_tickets WHERE request_id = ?1",
                    params![&request_id[..]],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        result.removed = result
            .removed
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if consumed_at.is_none() {
            result.unconsumed = result
                .unconsumed
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
    }
    Ok(result)
}
