// [ARCH-SPLIT 2026-10-02]
// Ticket issue, create, put, pull, ack, and cleanup. These are the repository trait methods.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl AnonymousMailboxCustodyRepository for SqliteAnonymousMailboxStore {
    fn issue_ticket(
        &self,
        request: &AnonymousMailboxTicketIssueV1,
        now: u64,
    ) -> Result<AnonymousMailboxTicketIssueOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        // [ANONYMOUS-MAILBOX-TICKET-ISSUER 2026-09-03 by Codex] Exact
        // durable replay precedes proof-of-work and every capacity check, but
        // an expired authority is never served again.
        if let Some(existing) = load_issued_ticket_by_request(&transaction, &request.request_id)? {
            if existing.request_commitment != request_commitment {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxTicketIssueOutcome::Conflict);
            }
            if existing.ticket.expires_at < now {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Err(AnonymousMailboxStoreError::Rejected);
            }
            validate_issued_ticket(&existing, request, &self.target_node_id)?;
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxTicketIssueOutcome::Existing(
                existing.ticket,
            ));
        }
        if load_issued_ticket_by_ticket(&transaction, &request.ticket_id)?.is_some() {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxTicketIssueOutcome::Conflict);
        }

        // [ANONYMOUS-MAILBOX-POLICY-TERMINAL 2026-09-24 by Codex] This
        // classification is valid only after exact replay lookup and before
        // purge/meta/issuance writes. A later error (including Rejected) has
        // no such provenance and remains an ambiguous repository failure.
        if request
            .verify_for_target(
                &self.target_node_id,
                now,
                self.config.ticket_issue_work_bits,
            )
            .is_err()
        {
            return Ok(AnonymousMailboxTicketIssueOutcome::PreWriteRejected);
        }
        let issuer = self
            .ticket_issuer
            .as_ref()
            .ok_or(AnonymousMailboxStoreError::Unavailable)?;

        let before = load_ticket_issue_meta(&transaction)?;
        let removed = purge_expired_issued_tickets(
            &transaction,
            now,
            u64::try_from(self.config.cleanup_batch_size)
                .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
        )?;
        let after_purge = TicketIssueMeta {
            outstanding: before
                .outstanding
                .checked_sub(removed.unconsumed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            ..before
        };
        if after_purge != before {
            update_ticket_issue_meta_exact(&transaction, before, after_purge)?;
        }
        let mut admission = after_purge;
        if now.saturating_sub(admission.window_started_at)
            >= self.config.ticket_issuance_window_secs
        {
            admission.window_started_at = now;
            admission.issues_in_window = 0;
        }
        let maximum_outstanding = u64::try_from(self.config.max_outstanding_tickets)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        let maximum_window = u64::try_from(self.config.max_ticket_issues_per_window)
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if admission.outstanding >= maximum_outstanding
            || admission.issues_in_window >= maximum_window
        {
            if admission != after_purge {
                update_ticket_issue_meta_exact(&transaction, after_purge, admission)?;
            }
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxTicketIssueOutcome::AtCapacity);
        }
        let ticket = AnonymousMailboxAdmissionTicketV1::issue(
            request.ticket_id,
            request.lease_claims_commitment,
            request.issued_at,
            request.expires_at,
            issuer,
        )
        .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let ticket_commitment = ticket
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_issued_tickets
                     (request_id, ticket_id, request_commitment, target_node_id,
                      claims_commitment, requested_at, expires_at, proof_nonce,
                      ticket_commitment, ticket_signature, consumed_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL)",
                    params![
                        &request.request_id[..],
                        &request.ticket_id[..],
                        &request_commitment[..],
                        &request.target_node_id[..],
                        &request.lease_claims_commitment[..],
                        as_i64(request.issued_at)?,
                        as_i64(request.expires_at)?,
                        &request.proof_nonce.to_le_bytes()[..],
                        &ticket_commitment[..],
                        &ticket.signature[..],
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let issued = TicketIssueMeta {
            outstanding: admission
                .outstanding
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            issues_in_window: admission
                .issues_in_window
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            ..admission
        };
        update_ticket_issue_meta_exact(&transaction, after_purge, issued)?;
        verify_ticket_issue_meta(&transaction, issued)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxTicketIssueOutcome::Issued(ticket))
    }

    fn create(
        &self,
        request: &AnonymousMailboxLeaseCreateV1,
        now: u64,
    ) -> Result<AnonymousMailboxCreateOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let ticket_commitment = request
            .admission
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let claims_commitment = request.claims_commitment();
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        if let Some((stored_ticket, stored_claims, stored_request, mailbox)) = transaction
            .query_row(
                "SELECT ticket_commitment, claims_commitment, lease_request_commitment, mailbox_id
                   FROM anonymous_mailbox_tickets WHERE ticket_id = ?1",
                params![&request.admission.ticket_id[..]],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        {
            let exact_authority =
                stored_ticket == ticket_commitment && stored_request == request_commitment;
            if !exact_authority {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxCreateOutcome::Conflict);
            }
            if stored_claims != claims_commitment || mailbox != request.mailbox_id {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            let lease = load_lease(&transaction, &request.mailbox_id)?
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            if lease.mailbox_id != request.mailbox_id
                || lease.deposit_verifier != request.deposit_verifier
                || lease.read_verifier != request.read_verifier
                || lease.max_items != request.max_items
                || lease.max_bytes != request.max_bytes
                || lease.created_at != request.issued_at
                || lease.expires_at != request.expires_at
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxCreateOutcome::Existing(lease));
        }

        request
            .verify_for_target(&self.target_node_id, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        if totals.leases >= self.config.max_leases_total as u64 {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxCreateOutcome::AtCapacity);
        }
        if load_lease(&transaction, &request.mailbox_id)?.is_some() {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxCreateOutcome::Conflict);
        }
        let issued_ticket =
            load_issued_ticket_by_ticket(&transaction, &request.admission.ticket_id)?;
        if let Some(record) = &issued_ticket {
            if record.ticket != request.admission
                || record.consumed_at.is_some()
                || record.ticket.expires_at < now
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            validate_issued_ticket(&record, &record.request, &self.target_node_id)?;
        }

        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_leases
                 (mailbox_id, ticket_id, claims_commitment, deposit_verifier, read_verifier,
                  max_items, max_bytes, current_items, current_bytes, next_sequence,
                  created_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 0, 1, ?8, ?9)",
                    params![
                        &request.mailbox_id[..],
                        &request.admission.ticket_id[..],
                        &claims_commitment[..],
                        &request.deposit_verifier[..],
                        &request.read_verifier[..],
                        i64::from(request.max_items),
                        as_i64(request.max_bytes)?,
                        as_i64(request.issued_at)?,
                        as_i64(request.expires_at)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_tickets
                 (ticket_id, ticket_commitment, claims_commitment, lease_request_commitment,
                  mailbox_id, consumed_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        &request.admission.ticket_id[..],
                        &ticket_commitment[..],
                        &claims_commitment[..],
                        &request_commitment[..],
                        &request.mailbox_id[..],
                        as_i64(now)?,
                        as_i64(request.admission.expires_at)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        if let Some(record) = issued_ticket {
            let before = load_ticket_issue_meta(&transaction)?;
            let after = TicketIssueMeta {
                outstanding: before
                    .outstanding
                    .checked_sub(1)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                ..before
            };
            execute_exactly_one(
                &transaction,
                transaction
                    .execute(
                        "UPDATE anonymous_mailbox_issued_tickets SET consumed_at = ?1
                         WHERE ticket_id = ?2 AND consumed_at IS NULL",
                        params![as_i64(now)?, &record.ticket.ticket_id[..]],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
            )?;
            update_ticket_issue_meta_exact(&transaction, before, after)?;
            verify_ticket_issue_meta(&transaction, after)?;
        }
        let new_totals = StoreTotals {
            leases: totals
                .leases
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?,
            ..totals
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        let lease = load_lease(&transaction, &request.mailbox_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if lease.current_items != 0 || lease.current_bytes != 0 || lease.next_sequence != 1 {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxCreateOutcome::Created(lease))
    }

    fn put(
        &self,
        request: &AnonymousMailboxPutV1,
        now: u64,
    ) -> Result<AnonymousMailboxPutOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let sealed_commitment = request.sealed_commitment();
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        if let Some(existing) = load_item(&transaction, &request.mailbox_id, &request.item_id)? {
            let stored_request: Vec<u8> = transaction
                .query_row(
                    "SELECT put_commitment FROM anonymous_mailbox_items
                     WHERE mailbox_id = ?1 AND item_id = ?2",
                    params![&request.mailbox_id[..], &request.item_id[..]],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            let outcome = if stored_request == request_commitment
                && existing.sealed_commitment == sealed_commitment
                && existing.sealed_envelope == request.sealed_envelope
            {
                AnonymousMailboxPutOutcome::Existing(existing)
            } else {
                AnonymousMailboxPutOutcome::Conflict
            };
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(outcome);
        }
        if load_ack_commitment(&transaction, &request.mailbox_id, &request.item_id)?.is_some() {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPutOutcome::Conflict);
        }

        let lease = match load_lease(&transaction, &request.mailbox_id)? {
            Some(lease) => lease,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxPutOutcome::LeaseNotFound);
            }
        };
        request
            .verify_at(&lease.deposit_verifier, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        if lease.expires_at < now || request.expires_at > lease.expires_at {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPutOutcome::LeaseExpired);
        }
        let item_bytes = u64::try_from(request.sealed_envelope.len())
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let new_items = u64::from(lease.current_items)
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let new_lease_bytes = lease
            .current_bytes
            .checked_add(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        let new_total_items = totals
            .items
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let new_total_bytes = totals
            .bytes
            .checked_add(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if new_items > u64::from(lease.max_items)
            || new_lease_bytes > lease.max_bytes
            || new_total_items
                > u64::try_from(self.config.max_items_total)
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?
            || new_total_bytes > self.config.max_bytes_total
        {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPutOutcome::AtCapacity);
        }
        let sequence = lease.next_sequence;
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_items
                 (mailbox_id, item_id, sequence, put_commitment, sealed_commitment,
                  sealed_envelope, stored_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        &request.mailbox_id[..],
                        &request.item_id[..],
                        as_i64(sequence)?,
                        &request_commitment[..],
                        &sealed_commitment[..],
                        &request.sealed_envelope,
                        as_i64(now)?,
                        as_i64(request.expires_at)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "UPDATE anonymous_mailbox_leases
                 SET current_items = ?1, current_bytes = ?2, next_sequence = ?3
                 WHERE mailbox_id = ?4 AND current_items = ?5
                   AND current_bytes = ?6 AND next_sequence = ?7",
                    params![
                        as_i64(new_items)?,
                        as_i64(new_lease_bytes)?,
                        as_i64(next_sequence)?,
                        &request.mailbox_id[..],
                        i64::from(lease.current_items),
                        as_i64(lease.current_bytes)?,
                        as_i64(lease.next_sequence)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let new_totals = StoreTotals {
            leases: totals.leases,
            items: new_total_items,
            bytes: new_total_bytes,
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        let item = load_item(&transaction, &request.mailbox_id, &request.item_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let updated_lease = load_lease(&transaction, &request.mailbox_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if u64::from(updated_lease.current_items) != new_items
            || updated_lease.current_bytes != new_lease_bytes
            || updated_lease.next_sequence != next_sequence
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxPutOutcome::Stored(item))
    }

    fn pull_one(
        &self,
        request: &AnonymousMailboxPullOneV1,
        now: u64,
    ) -> Result<AnonymousMailboxPullOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        // [ANONYMOUS-MAILBOX-PULL-REPLAY 2026-09-24 by Codex] Resolve the
        // exact signed request before freshness and live-item lookup. This is
        // the durable authority for a response lost immediately before ACK.
        if let Some(outcome) = load_pull_replay(&transaction, request, &request_commitment, now)? {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(outcome);
        }
        let lease = match load_lease(&transaction, &request.mailbox_id)? {
            Some(lease) => lease,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxPullOutcome::LeaseNotFound);
            }
        };
        request
            .verify_at(&lease.read_verifier, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        if lease.expires_at < now {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPullOutcome::LeaseExpired);
        }
        let state = if request.cursor.is_empty() {
            CursorState {
                snapshot_ceiling: lease.next_sequence.saturating_sub(1),
                next_sequence: 1,
                expires_at: lease.expires_at.min(now.saturating_add(CURSOR_TTL_SECS)),
            }
        } else {
            let state = self.decode_cursor(&request.mailbox_id, &request.cursor, now)?;
            if state.snapshot_ceiling >= lease.next_sequence {
                return Err(AnonymousMailboxStoreError::Rejected);
            }
            state
        };
        let replay_retain_until = lease
            .expires_at
            .min(now.saturating_add(PULL_REPLAY_RETENTION_SECS));
        let row = transaction
            .query_row(
                "SELECT item_id, sequence, sealed_commitment, length(sealed_envelope),
                        stored_at, expires_at
                 FROM anonymous_mailbox_items
                 WHERE mailbox_id = ?1 AND sequence >= ?2 AND sequence <= ?3
                   AND expires_at >= ?4
                 ORDER BY sequence ASC LIMIT 1",
                params![
                    &request.mailbox_id[..],
                    as_i64(state.next_sequence)?,
                    as_i64(state.snapshot_ceiling)?,
                    as_i64(now)?,
                ],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let Some((item_id_bytes, sequence_raw, commitment_bytes, length, _, item_expires_at)) = row
        else {
            insert_pull_replay(
                &transaction,
                &self.config,
                request,
                &request_commitment,
                StoredPullReplay::Empty,
                now,
                replay_retain_until,
            )?;
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxPullOutcome::Empty);
        };
        let sequence = as_u64(sequence_raw)?;
        let item_expires_at = as_u64(item_expires_at)?;
        // [ANONYMOUS-MAILBOX-ITEM-EXPIRY 2026-09-24 by Codex] An exact
        // retry is not an authority to serve ciphertext past its signed item
        // expiry, even after the source item has been ACKed away.
        let item_replay_retain_until = replay_retain_until.min(item_expires_at);
        let admitted_length =
            usize::try_from(length).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        if admitted_length == 0 || admitted_length > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        // The length admission above occurs in the same snapshot before this
        // BLOB is materialized.
        let sealed_envelope: Vec<u8> = transaction
            .query_row(
                "SELECT sealed_envelope FROM anonymous_mailbox_items
                 WHERE mailbox_id = ?1 AND item_id = ?2",
                params![&request.mailbox_id[..], &item_id_bytes],
                |row| row.get(0),
            )
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        if sealed_envelope.len() != admitted_length {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        let item_id = fixed::<16>(&item_id_bytes)?;
        let sealed_commitment = fixed::<32>(&commitment_bytes)?;
        let calculated_commitment: [u8; 32] = Sha256::digest(&sealed_envelope).into();
        if sealed_commitment != calculated_commitment {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        let mut padded = vec![0_u8; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES];
        padded[..sealed_envelope.len()].copy_from_slice(&sealed_envelope);
        let cursor = self.encode_cursor(
            &request.mailbox_id,
            CursorState {
                snapshot_ceiling: state.snapshot_ceiling,
                next_sequence: sequence
                    .checked_add(1)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                expires_at: state.expires_at,
            },
        )?;
        insert_pull_replay(
            &transaction,
            &self.config,
            request,
            &request_commitment,
            StoredPullReplay::Item {
                item_id,
                sealed_commitment,
                sealed_envelope: sealed_envelope.clone(),
                cursor: cursor.clone(),
                item_expires_at,
            },
            now,
            item_replay_retain_until,
        )?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxPullOutcome::Item(
            AnonymousMailboxPulledItem {
                item_id,
                sealed_commitment,
                sealed_length: u32::try_from(sealed_envelope.len())
                    .map_err(|_| AnonymousMailboxStoreError::Corrupt)?,
                padded_sealed_envelope: padded,
                cursor,
            },
        ))
    }

    fn ack(
        &self,
        request: &AnonymousMailboxAckV1,
        now: u64,
    ) -> Result<AnonymousMailboxAckOutcome, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let request_commitment = request
            .request_commitment()
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        // [ANONYMOUS-MAILBOX-STORE 2026-09-02 by Codex] A durable exact ACK
        // replay is decided before request freshness and lease lookup. The
        // tombstone is the retry authority; a restarted caller must not lose
        // idempotency merely because the original signature window elapsed.
        if let Some((stored_commitment, stored_request)) = transaction
            .query_row(
                "SELECT sealed_commitment, ack_request_commitment
                 FROM anonymous_mailbox_acks WHERE mailbox_id = ?1 AND item_id = ?2",
                params![&request.mailbox_id[..], &request.item_id[..]],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?
        {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(
                if stored_commitment == request.sealed_commitment
                    && stored_request == request_commitment
                {
                    AnonymousMailboxAckOutcome::AlreadyAcknowledged
                } else {
                    AnonymousMailboxAckOutcome::Conflict
                },
            );
        }
        let lease = match load_lease(&transaction, &request.mailbox_id)? {
            Some(lease) => lease,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxAckOutcome::NotFound);
            }
        };
        request
            .verify_at(&lease.read_verifier, now)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        if lease.expires_at < now {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxAckOutcome::NotFound);
        }
        let item = match load_item(&transaction, &request.mailbox_id, &request.item_id)? {
            Some(item) => item,
            None => {
                transaction
                    .commit()
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                return Ok(AnonymousMailboxAckOutcome::NotFound);
            }
        };
        if item.sealed_commitment != request.sealed_commitment {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            return Ok(AnonymousMailboxAckOutcome::Conflict);
        }
        let retain_until = lease.expires_at.min(
            now.saturating_add(ACK_TOMBSTONE_RETENTION_SECS)
                .max(item.expires_at),
        );
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_items WHERE mailbox_id = ?1 AND item_id = ?2",
                    params![&request.mailbox_id[..], &request.item_id[..]],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_acks
                 (mailbox_id, item_id, sealed_commitment, ack_request_commitment,
                  acknowledged_at, retain_until)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        &request.mailbox_id[..],
                        &request.item_id[..],
                        &request.sealed_commitment[..],
                        &request_commitment[..],
                        as_i64(now)?,
                        as_i64(retain_until)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let item_bytes = u64::try_from(item.sealed_envelope.len())
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        let current_items = u64::from(lease.current_items)
            .checked_sub(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let current_bytes = lease
            .current_bytes
            .checked_sub(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let total_items = totals
            .items
            .checked_sub(1)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let total_bytes = totals
            .bytes
            .checked_sub(item_bytes)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        execute_exactly_one(
            &transaction,
            transaction
                .execute(
                    "UPDATE anonymous_mailbox_leases SET current_items = ?1, current_bytes = ?2
                 WHERE mailbox_id = ?3 AND current_items = ?4 AND current_bytes = ?5",
                    params![
                        as_i64(current_items)?,
                        as_i64(current_bytes)?,
                        &request.mailbox_id[..],
                        i64::from(lease.current_items),
                        as_i64(lease.current_bytes)?,
                    ],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
        )?;
        let new_totals = StoreTotals {
            leases: totals.leases,
            items: total_items,
            bytes: total_bytes,
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        let updated_lease = load_lease(&transaction, &request.mailbox_id)?
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        if u64::from(updated_lease.current_items) != current_items
            || updated_lease.current_bytes != current_bytes
            || updated_lease.next_sequence != lease.next_sequence
        {
            return Err(AnonymousMailboxStoreError::Corrupt);
        }
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(AnonymousMailboxAckOutcome::Acknowledged)
    }

    fn cleanup(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxCleanupReport, AnonymousMailboxStoreError> {
        let _permit = self.acquire()?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        let totals = load_totals(&transaction)?;
        validate_totals_limits(&totals, &self.config)?;
        let mut report = AnonymousMailboxCleanupReport::default();
        let mut remaining = u64::try_from(self.config.cleanup_batch_size)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;

        let replay_totals = load_pull_replay_totals(&transaction)?;
        validate_pull_replay_totals(replay_totals, &self.config)?;
        if remaining > 0 {
            let (rows, bytes): (i64, i64) = transaction
                .query_row(
                    "SELECT COUNT(*), COALESCE(SUM(length(sealed_envelope)), 0)
                     FROM anonymous_mailbox_pull_replays WHERE rowid IN
                     (SELECT rowid FROM anonymous_mailbox_pull_replays
                      WHERE retain_until < ?1 ORDER BY retain_until, rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            let rows = as_u64(rows)?;
            let bytes = as_u64(bytes)?;
            if rows > 0 {
                let removed = transaction
                    .execute(
                        "DELETE FROM anonymous_mailbox_pull_replays WHERE rowid IN
                         (SELECT rowid FROM anonymous_mailbox_pull_replays
                          WHERE retain_until < ?1 ORDER BY retain_until, rowid LIMIT ?2)",
                        params![as_i64(now)?, as_i64(remaining)?],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
                if u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)? != rows
                {
                    return Err(AnonymousMailboxStoreError::Corrupt);
                }
                let updated = PullReplayTotals {
                    rows: replay_totals
                        .rows
                        .checked_sub(rows)
                        .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                    bytes: replay_totals
                        .bytes
                        .checked_sub(bytes)
                        .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                };
                update_pull_replay_totals_exact(&transaction, replay_totals, updated)?;
                report.pull_replays_removed = rows;
                remaining = remaining
                    .checked_sub(rows)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            }
        }

        let expired_items = select_expired_items(&transaction, now, remaining)?;
        for (mailbox_id, item_id, bytes) in expired_items {
            let mailbox_key = fixed::<32>(&mailbox_id)?;
            let lease = load_lease(&transaction, &mailbox_key)?
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            let updated_items = u64::from(lease.current_items)
                .checked_sub(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            let updated_bytes = lease
                .current_bytes
                .checked_sub(bytes)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            execute_exactly_one(
                &transaction,
                transaction
                    .execute(
                        "DELETE FROM anonymous_mailbox_items
                     WHERE mailbox_id = ?1 AND item_id = ?2",
                        params![&mailbox_id, &item_id],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
            )?;
            execute_exactly_one(
                &transaction,
                transaction
                    .execute(
                        "UPDATE anonymous_mailbox_leases
                     SET current_items = ?1, current_bytes = ?2
                     WHERE mailbox_id = ?3 AND current_items = ?4 AND current_bytes = ?5",
                        params![
                            as_i64(updated_items)?,
                            as_i64(updated_bytes)?,
                            &mailbox_id,
                            i64::from(lease.current_items),
                            as_i64(lease.current_bytes)?,
                        ],
                    )
                    .map_err(|_| AnonymousMailboxStoreError::Unavailable)?,
            )?;
            let updated = load_lease(&transaction, &mailbox_key)?
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            if u64::from(updated.current_items) != updated_items
                || updated.current_bytes != updated_bytes
                || updated.next_sequence != lease.next_sequence
            {
                return Err(AnonymousMailboxStoreError::Corrupt);
            }
            report.items_removed = report
                .items_removed
                .checked_add(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            report.bytes_removed = report
                .bytes_removed
                .checked_add(bytes)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(1)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }

        if remaining > 0 {
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_acks WHERE rowid IN
                     (SELECT rowid FROM anonymous_mailbox_acks
                      WHERE retain_until < ?1 ORDER BY retain_until, rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            report.acknowledgements_removed =
                u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(report.acknowledgements_removed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
        if remaining > 0 {
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_leases WHERE rowid IN
                     (SELECT l.rowid FROM anonymous_mailbox_leases l
                      WHERE l.expires_at < ?1 AND l.current_items = 0 AND l.current_bytes = 0
                        AND NOT EXISTS (SELECT 1 FROM anonymous_mailbox_items i
                                        WHERE i.mailbox_id = l.mailbox_id)
                        AND NOT EXISTS (SELECT 1 FROM anonymous_mailbox_acks a
                                        WHERE a.mailbox_id = l.mailbox_id)
                        AND NOT EXISTS (SELECT 1 FROM anonymous_mailbox_pull_replays r
                                        WHERE r.mailbox_id = l.mailbox_id)
                      ORDER BY l.expires_at, l.rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            report.leases_removed =
                u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(report.leases_removed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
        if remaining > 0 {
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_tickets WHERE rowid IN
                     (SELECT t.rowid FROM anonymous_mailbox_tickets t
                      LEFT JOIN anonymous_mailbox_leases l ON l.mailbox_id = t.mailbox_id
                      WHERE t.expires_at < ?1 AND l.mailbox_id IS NULL
                      ORDER BY t.expires_at, t.rowid LIMIT ?2)",
                    params![as_i64(now)?, as_i64(remaining)?],
                )
                .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
            report.tickets_removed =
                u64::try_from(removed).map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
            remaining = remaining
                .checked_sub(report.tickets_removed)
                .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        }
        if remaining > 0 {
            let before = load_ticket_issue_meta(&transaction)?;
            let purge = purge_expired_issued_tickets(&transaction, now, remaining)?;
            let after = TicketIssueMeta {
                outstanding: before
                    .outstanding
                    .checked_sub(purge.unconsumed)
                    .ok_or(AnonymousMailboxStoreError::Corrupt)?,
                ..before
            };
            if after != before {
                update_ticket_issue_meta_exact(&transaction, before, after)?;
                verify_ticket_issue_meta(&transaction, after)?;
            }
            report.issued_tickets_removed = purge.removed;
        }

        let leases = totals
            .leases
            .checked_sub(report.leases_removed)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let items = totals
            .items
            .checked_sub(report.items_removed)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let bytes = totals
            .bytes
            .checked_sub(report.bytes_removed)
            .ok_or(AnonymousMailboxStoreError::Corrupt)?;
        let new_totals = StoreTotals {
            leases,
            items,
            bytes,
        };
        validate_totals_limits(&new_totals, &self.config)?;
        update_totals_exact(&transaction, totals, new_totals)?;
        verify_totals_exact(&transaction, new_totals)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        Ok(report)
    }
}
