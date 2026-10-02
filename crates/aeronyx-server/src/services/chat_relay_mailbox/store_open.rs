// [ARCH-SPLIT 2026-10-02]
// Open the mailbox database, take the operation permit, and encode cursors.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl SqliteAnonymousMailboxStore {
    pub(super) fn open_inner(
        config: AnonymousMailboxStoreConfig,
        target_node_id: [u8; 32],
        cursor_secret: [u8; 32],
        ticket_issuer: Option<IdentityKeyPair>,
    ) -> Result<Self, AnonymousMailboxStoreError> {
        if !config.enabled {
            return Err(AnonymousMailboxStoreError::Disabled);
        }
        if config.db_path.is_empty()
            || config.db_path == ":memory:"
            || config.max_leases_total == 0
            || i64::try_from(config.max_leases_total).is_err()
            || config.max_items_total < usize::from(MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE)
            || i64::try_from(config.max_items_total).is_err()
            || config.max_bytes_total == 0
            || config.max_bytes_total > i64::MAX as u64
            || config.max_in_flight == 0
            || config.cleanup_batch_size == 0
            || config.max_outstanding_tickets == 0
            || i64::try_from(config.max_outstanding_tickets).is_err()
            || config.max_ticket_issues_per_window == 0
            || i64::try_from(config.max_ticket_issues_per_window).is_err()
            || config.ticket_issuance_window_secs == 0
            || config.ticket_issue_work_bits == 0
            || config.ticket_issue_work_bits
                > aeronyx_core::protocol::anonymous_mailbox::MAX_ANONYMOUS_MAILBOX_TICKET_ISSUE_WORK_BITS
            || cursor_secret == [0; 32]
        {
            return Err(AnonymousMailboxStoreError::Rejected);
        }

        let target = prepare_private_sqlite_target(Path::new(&config.db_path))?;

        // [ANONYMOUS-MAILBOX-STORE 2026-09-02 by Codex] SQLite NOFOLLOW,
        // owner-only mode, startup quick-check, WAL and FULL durability reuse
        // the existing relay custody activation policy.
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        verify_private_file(&target.resolved_path, false)?;
        restrict_private_sqlite_permissions(&target.resolved_path)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        verify_sqlite_physical_integrity(&connection, "anonymous_mailbox_startup_integrity")
            .map_err(|_| AnonymousMailboxStoreError::Corrupt)?;
        configure_full_durability(&connection, MINIMUM_SYNCHRONOUS_LEVEL)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        initialize_or_verify_schema(&mut connection, &target_node_id, &config)?;
        let startup_limits = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;
        validate_totals_limits(&load_totals(&startup_limits)?, &config)?;
        validate_pull_replay_totals(load_pull_replay_totals(&startup_limits)?, &config)?;
        startup_limits
            .commit()
            .map_err(|_| AnonymousMailboxStoreError::Unavailable)?;

        Ok(Self {
            config,
            target_node_id,
            ticket_issuer,
            cursor_secret,
            connection: Mutex::new(connection),
            in_flight: AtomicUsize::new(0),
            #[cfg(unix)]
            _database_parent: target.parent,
        })
    }

    pub(super) fn acquire(&self) -> Result<OperationPermit<'_>, AnonymousMailboxStoreError> {
        self.in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.config.max_in_flight).then_some(current + 1)
            })
            .map_err(|_| AnonymousMailboxStoreError::Busy)?;
        Ok(OperationPermit {
            counter: &self.in_flight,
        })
    }

    pub(super) fn encode_cursor(
        &self,
        mailbox_id: &[u8; 32],
        state: CursorState,
    ) -> Result<Vec<u8>, AnonymousMailboxStoreError> {
        let mut body = Vec::with_capacity(CURSOR_BYTES);
        body.push(CURSOR_VERSION);
        body.extend_from_slice(&state.snapshot_ceiling.to_le_bytes());
        body.extend_from_slice(&state.next_sequence.to_le_bytes());
        body.extend_from_slice(&state.expires_at.to_le_bytes());
        let mut mac = CursorMac::new_from_slice(&self.cursor_secret)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        mac.update(CURSOR_DOMAIN);
        mac.update(mailbox_id);
        mac.update(&body);
        body.extend_from_slice(&mac.finalize().into_bytes());
        Ok(body)
    }

    pub(super) fn decode_cursor(
        &self,
        mailbox_id: &[u8; 32],
        cursor: &[u8],
        now: u64,
    ) -> Result<CursorState, AnonymousMailboxStoreError> {
        if cursor.len() != CURSOR_BYTES || cursor[0] != CURSOR_VERSION {
            return Err(AnonymousMailboxStoreError::Rejected);
        }
        let (body, tag) = cursor.split_at(CURSOR_BODY_BYTES);
        let mut mac = CursorMac::new_from_slice(&self.cursor_secret)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        mac.update(CURSOR_DOMAIN);
        mac.update(mailbox_id);
        mac.update(body);
        mac.verify_slice(tag)
            .map_err(|_| AnonymousMailboxStoreError::Rejected)?;
        let state = CursorState {
            snapshot_ceiling: u64::from_le_bytes(
                body[1..9]
                    .try_into()
                    .map_err(|_| AnonymousMailboxStoreError::Rejected)?,
            ),
            next_sequence: u64::from_le_bytes(
                body[9..17]
                    .try_into()
                    .map_err(|_| AnonymousMailboxStoreError::Rejected)?,
            ),
            expires_at: u64::from_le_bytes(
                body[17..25]
                    .try_into()
                    .map_err(|_| AnonymousMailboxStoreError::Rejected)?,
            ),
        };
        if state.next_sequence == 0
            || state.next_sequence > state.snapshot_ceiling.saturating_add(1)
            || state.expires_at < now
        {
            return Err(AnonymousMailboxStoreError::Rejected);
        }
        Ok(state)
    }
}
