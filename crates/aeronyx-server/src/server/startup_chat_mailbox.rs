// [ARCH-SPLIT 2026-10-02]
// Chat relay and anonymous mailbox startup. Custody witness checks stay in Server::run.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl Server {
    // ============================================
    // MemChain initialization
    // ============================================

    /// Initializes the explicitly configured durable Chat Relay service.
    ///
    /// [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] `enabled=true` is a
    /// required service contract, not a best-effort hint. Return one stable,
    /// aggregate reason bucket so SQLite paths or raw storage diagnostics do
    /// not enter process-health output while the startup transaction fails.
    pub(super) fn init_chat_relay_service(&self) -> Result<Option<Arc<ChatRelayService>>> {
        if !self.config.memchain.is_chat_relay_enabled() {
            if self.config.memchain.is_enabled() {
                info!(
                    "[CHAT_RELAY] Disabled by memchain.chat_relay.enabled=false; chat routes remain unavailable"
                );
            }
            return Ok(None);
        }

        let node_secret = derive_node_secret(&self.identity.to_bytes());
        ChatRelayService::new(self.config.memchain.chat_relay.clone(), node_secret)
            .map(|service| Some(Arc::new(service)))
            .map_err(|error| {
                let reason = error.reason_bucket();
                error!(
                    reason,
                    "[CHAT_RELAY] Required durable service initialization failed"
                );
                ServerError::startup_failed(format!("Chat Relay initialization failed ({reason})"))
            })
    }

    /// Opens the explicitly enabled node-blind mailbox repository. Disabled
    /// configuration performs no filesystem operation and is not advertised.
    pub(super) async fn init_anonymous_mailbox_store(
        &self,
    ) -> Result<Option<Arc<SqliteAnonymousMailboxStore>>> {
        self.init_anonymous_mailbox_store_at(unix_now_secs()).await
    }

    pub(super) async fn init_anonymous_mailbox_store_at(
        &self,
        now: u64,
    ) -> Result<Option<Arc<SqliteAnonymousMailboxStore>>> {
        let config = self.config.memchain.chat_relay.anonymous_mailbox.clone();
        if !config.enabled {
            return Ok(None);
        }
        // [ANONYMOUS-MAILBOX-CLEANUP-RUNTIME 2026-09-13 by Codex] Opening and
        // the first bounded cleanup share one blocking startup job. Readiness
        // cannot become true while expired custody rows remain unaudited, and
        // disabled configuration returns before any key or path operation.
        // [BLIND-RELAY-ANONYMOUS-MAILBOX 2026-09-03 by Codex] The cursor MAC
        // root is stable but private, independently domain-separated from the
        // existing chat relay secret, and never logged or returned.
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx/anonymous-mailbox/cursor-root/v1");
        hasher.update(self.identity.to_bytes());
        let cursor_secret: [u8; 32] = hasher.finalize().into();
        let identity = self.identity.clone();
        // [BLIND-RELAY-ANONYMOUS-MAILBOX-TICKET 2026-09-03 by Codex] The
        // mailbox ticket signer is precisely this node's existing identity;
        // no global/config issuer or additional private key is introduced.
        // Construct it before readiness advertisement so an enabled mailbox
        // cannot claim availability while ticket issuance is unavailable.
        let (store, report) = tokio::task::spawn_blocking(move || {
            let store = Arc::new(SqliteAnonymousMailboxStore::open_with_ticket_issuer(
                config,
                identity,
                cursor_secret,
            )?);
            let report = store.cleanup(now)?;
            Ok::<_, AnonymousMailboxStoreError>((store, report))
        })
        .await
        .map_err(|_| ServerError::startup_failed("Anonymous mailbox initialization failed"))?
        .map_err(|_| ServerError::startup_failed("Anonymous mailbox initialization failed"))?;
        info!(
            leases_removed = report.leases_removed,
            items_removed = report.items_removed,
            bytes_removed = report.bytes_removed,
            acknowledgements_removed = report.acknowledgements_removed,
            tickets_removed = report.tickets_removed,
            issued_tickets_removed = report.issued_tickets_removed,
            "[ANONYMOUS_MAILBOX] Bounded startup cleanup completed"
        );
        Ok(Some(store))
    }

    /// Activates the optional source only after authenticated PeerStore
    /// bootstrap. All filesystem and SQLite work stays off the async startup
    /// worker; an enabled source either opens its own private journal or stops
    /// startup before any VPN route can become ready.
    pub(super) async fn init_anonymous_mailbox_source_coordinator(
        &self,
        peer_store: Arc<PeerStore>,
    ) -> Result<Option<AnonymousMailboxSourceRuntime>> {
        self.init_anonymous_mailbox_source_coordinator_at(peer_store, unix_now_secs())
            .await
    }

    pub(super) async fn init_anonymous_mailbox_source_coordinator_at(
        &self,
        peer_store: Arc<PeerStore>,
        now: u64,
    ) -> Result<Option<AnonymousMailboxSourceRuntime>> {
        let config = self
            .config
            .memchain
            .chat_relay
            .anonymous_mailbox_source
            .clone();
        if !config.enabled {
            return Ok(None);
        }
        // [ANONYMOUS-MAILBOX-SOURCE-WIRING 2026-09-03 by Codex] Source
        // journal encryption is domain-separated from relay custody, mailbox
        // cursor, and every wire signature. It remains local-only and is never
        // advertised or returned through discovery.
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx/anonymous-mailbox/source-journal-key/v1");
        hasher.update(self.identity.to_bytes());
        let journal_key: [u8; 32] = hasher.finalize().into();
        let source_identity = Arc::new(self.identity.clone());
        let (runtime, report) = tokio::task::spawn_blocking(move || {
            let journal = Arc::new(SqliteAnonymousMailboxSourceJournal::open(
                config,
                journal_key,
            )?);
            // [ANONYMOUS-MAILBOX-CLEANUP-RUNTIME 2026-09-13 by Codex]
            // Cleanup must finish before the coordinator can enter API/runtime
            // composition. Merely spawning an immediate interval tick would
            // allow readiness to race expired terminal source state.
            let report = journal.cleanup_terminal_records(now)?;
            let runtime = AnonymousMailboxSourceRuntime {
                coordinator: Arc::new(AnonymousMailboxSourceCoordinator::new(
                    source_identity,
                    peer_store,
                    Arc::clone(&journal),
                )),
                journal,
            };
            Ok::<_, AnonymousMailboxSourceError>((runtime, report))
        })
        .await
        .map_err(|_| ServerError::startup_failed("Anonymous mailbox source initialization failed"))?
        .map_err(|_| {
            ServerError::startup_failed("Anonymous mailbox source initialization failed")
        })?;
        info!(
            rows_removed = report.rows_removed,
            bytes_removed = report.bytes_removed,
            "[ANONYMOUS_MAILBOX] Bounded source startup cleanup completed"
        );
        Ok(Some(runtime))
    }
}
