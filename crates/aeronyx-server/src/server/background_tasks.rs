// [SERVER-BACKGROUND-TASKS-SPLIT 2026-09-25 by Codex]
// Periodic cleanup and background-task ownership remains shutdown aware.
use super::*;

/// Owns the management-plane sender and every long-lived management task
/// until the main runtime adopts them.
// [MANAGEMENT-RUNTIME-OWNERSHIP 2026-07-30 by Codex] A management task must
// never become fire-and-forget: losing heartbeat, policy commands, or session
// reporting while the process remains healthy creates a false operational
// state. The fixed-size array also makes adding a fourth detached task a
// compile-visible architecture change.
pub(super) struct ManagementRuntime {
    pub(super) session_events: SessionEventSender,
    pub(super) tasks: [(&'static str, JoinHandle<()>); 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnonymousMailboxCleanupFailureDisposition {
    Retryable,
    Fatal,
}

#[derive(Debug)]
pub(super) struct AnonymousMailboxCleanupCycleOutcome {
    pub(super) custody:
        Option<std::result::Result<AnonymousMailboxCleanupReport, AnonymousMailboxStoreError>>,
    pub(super) source: Option<
        std::result::Result<AnonymousMailboxSourceCleanupReport, AnonymousMailboxSourceError>,
    >,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnonymousMailboxCleanupLoopDirective {
    Continue,
    StopFatal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnonymousMailboxCleanupLoopExit {
    Shutdown,
    Fatal,
}

pub(super) fn anonymous_mailbox_custody_cleanup_failure_disposition(
    error: AnonymousMailboxStoreError,
) -> AnonymousMailboxCleanupFailureDisposition {
    match error {
        AnonymousMailboxStoreError::Busy | AnonymousMailboxStoreError::Unavailable => {
            AnonymousMailboxCleanupFailureDisposition::Retryable
        }
        AnonymousMailboxStoreError::Corrupt
        | AnonymousMailboxStoreError::UnsupportedSchema
        | AnonymousMailboxStoreError::Disabled
        | AnonymousMailboxStoreError::Rejected => AnonymousMailboxCleanupFailureDisposition::Fatal,
    }
}

pub(super) fn anonymous_mailbox_source_cleanup_failure_disposition(
    error: AnonymousMailboxSourceError,
) -> AnonymousMailboxCleanupFailureDisposition {
    match error {
        AnonymousMailboxSourceError::Unavailable => {
            AnonymousMailboxCleanupFailureDisposition::Retryable
        }
        AnonymousMailboxSourceError::Corrupt
        | AnonymousMailboxSourceError::Disabled
        | AnonymousMailboxSourceError::Rejected
        | AnonymousMailboxSourceError::Conflict
        | AnonymousMailboxSourceError::Ambiguous => {
            AnonymousMailboxCleanupFailureDisposition::Fatal
        }
    }
}

pub(super) async fn run_anonymous_mailbox_cleanup_loop<C, F>(
    interval: Duration,
    mut shutdown_rx: broadcast::Receiver<()>,
    mut cycle: C,
) -> AnonymousMailboxCleanupLoopExit
where
    C: FnMut() -> F,
    F: std::future::Future<Output = AnonymousMailboxCleanupLoopDirective>,
{
    let mut timer = tokio::time::interval(interval);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.recv() => return AnonymousMailboxCleanupLoopExit::Shutdown,
            _ = timer.tick() => {
                if cycle().await == AnonymousMailboxCleanupLoopDirective::StopFatal {
                    return AnonymousMailboxCleanupLoopExit::Fatal;
                }
            }
        }
    }
}

impl Server {
    pub(super) fn spawn_directory_replica_sync_task(
        &self,
        peer_store: Arc<PeerStore>,
        store: Option<Arc<DirectoryReplicaStore>>,
        runtime: Arc<DirectoryReplicaSyncRuntime>,
        directory_http_client: Arc<reqwest::Client>,
    ) -> Result<Option<JoinHandle<()>>> {
        let peers = self
            .config
            .discovery
            .directory_chain_sync_peer_node_id_bytes();
        let full_node_mirror_enabled = self.config.discovery.directory_full_node_mirror_enabled;
        let full_node_mirror_max_producers = self
            .config
            .discovery
            .directory_full_node_mirror_max_producers;
        if peers.is_empty() && !full_node_mirror_enabled {
            info!(
                "[DIRECTORY_REPLICA] Outbound sync disabled; no peers are pinned and mirror mode is off"
            );
            return Ok(None);
        }
        // [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Validation
        // normally binds an enabled synchronization mode to a durable store,
        // but keep the runtime boundary fail-closed for embedders and future
        // configuration paths that construct `ServerConfig` programmatically.
        let store = store.ok_or_else(|| {
            ServerError::startup_failed(
                "Directory replica synchronization requires an initialized replica store",
            )
        })?;
        let interval_secs = self.config.discovery.directory_chain_sync_interval_secs;
        let witness_min_verified = self
            .config
            .discovery
            .directory_observation_witness_min_verified;
        let coordinator = DirectoryReplicaSyncCoordinator::new_with_policy_and_resources(
            peers,
            interval_secs,
            DirectoryReplicaSyncResources {
                store,
                runtime,
                peer_store,
                identity: Arc::new(self.identity.clone()),
                client: directory_http_client.as_ref().clone(),
            },
            DirectoryReplicaSyncPolicy {
                witness_min_verified,
                full_node_mirror_enabled,
                full_node_mirror_max_producers,
            },
        )
        .map_err(|reason| {
            // Keep the startup reason stable and free of producer identities,
            // endpoints, paths, blocks, or request metadata.
            ServerError::startup_failed(format!(
                "Directory replica synchronization initialization failed: {reason}"
            ))
        })?;
        Ok(Some(coordinator.spawn(self.shutdown_tx.subscribe())))
    }

    pub(super) async fn reconcile_directory_chain_once(
        store: Arc<DirectoryChainStore>,
        identity: Arc<IdentityKeyPair>,
        peer_store: &PeerStore,
    ) -> std::result::Result<DirectoryChainAppendReport, String> {
        let produced_at = unix_now_secs();
        let descriptors = peer_store.export_peer_cache_snapshot(produced_at).peers;
        tokio::task::spawn_blocking(move || {
            store.append_descriptors(&descriptors, produced_at, identity.as_ref())
        })
        .await
        .map_err(|error| {
            format!(
                "blocking reconciliation task failed: {}",
                RuntimeTaskJoinFailureKind::classify(&error).blocking_task_reason()
            )
        })?
        .map_err(|error| error.to_string())
    }

    pub(super) fn spawn_directory_chain_persistence_task(
        &self,
        peer_store: Arc<PeerStore>,
        store: Option<Arc<DirectoryChainStore>>,
    ) -> Option<JoinHandle<()>> {
        let store = store?;
        let identity = Arc::new(self.identity.clone());
        let interval_secs = self.config.discovery.peer_cache_write_interval_secs;
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        Some(tokio::spawn(async move {
            let period = Duration::from_secs(interval_secs);
            let start = tokio::time::Instant::now() + period;
            let mut timer = tokio::time::interval_at(start, period);
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                let reason = tokio::select! {
                    _ = shutdown_rx.recv() => "shutdown",
                    _ = timer.tick() => "interval",
                };
                match Self::reconcile_directory_chain_once(
                    Arc::clone(&store),
                    Arc::clone(&identity),
                    &peer_store,
                )
                .await
                {
                    Ok(report) => {
                        debug!(
                            reason,
                            blocks_appended = report.blocks_appended,
                            commitments_appended = report.commitments_appended,
                            tip_height = report.tip_height,
                            "[DIRECTORY_CHAIN] Descriptor commitments reconciled"
                        );
                    }
                    Err(error) => {
                        error!(
                            reason,
                            error = %error,
                            "[DIRECTORY_CHAIN] Descriptor reconciliation failed"
                        );
                    }
                }
                if reason == "shutdown" {
                    break;
                }
            }
        }))
    }

    pub(super) fn run_anonymous_mailbox_cleanup_cycle(
        custody: Option<Arc<SqliteAnonymousMailboxStore>>,
        source: Option<Arc<SqliteAnonymousMailboxSourceJournal>>,
        now: u64,
    ) -> AnonymousMailboxCleanupCycleOutcome {
        // Each repository already bounds one IMMEDIATE transaction. Keeping
        // both calls in this single blocking closure makes overlap impossible
        // while allowing an independent repository to report after its peer
        // returned a coarse error.
        let custody = custody.map(|store| store.cleanup(now));
        let source = source.map(|journal| journal.cleanup_terminal_records(now));
        AnonymousMailboxCleanupCycleOutcome { custody, source }
    }

    pub(super) fn custody_cleanup_reason(error: AnonymousMailboxStoreError) -> &'static str {
        match error {
            AnonymousMailboxStoreError::Busy => "busy",
            AnonymousMailboxStoreError::Unavailable => "unavailable",
            AnonymousMailboxStoreError::Corrupt => "corrupt",
            AnonymousMailboxStoreError::UnsupportedSchema => "unsupported_schema",
            AnonymousMailboxStoreError::Disabled => "disabled",
            AnonymousMailboxStoreError::Rejected => "rejected",
        }
    }

    pub(super) fn source_cleanup_reason(error: AnonymousMailboxSourceError) -> &'static str {
        match error {
            AnonymousMailboxSourceError::Unavailable => "unavailable",
            AnonymousMailboxSourceError::Corrupt => "corrupt",
            AnonymousMailboxSourceError::Disabled => "disabled",
            AnonymousMailboxSourceError::Rejected => "rejected",
            AnonymousMailboxSourceError::Conflict => "conflict",
            AnonymousMailboxSourceError::Ambiguous => "ambiguous",
        }
    }

    pub(super) fn observe_anonymous_mailbox_cleanup_cycle(
        outcome: AnonymousMailboxCleanupCycleOutcome,
    ) -> AnonymousMailboxCleanupLoopDirective {
        let mut fatal = false;
        if let Some(result) = outcome.custody {
            match result {
                Ok(report) => {
                    if report.leases_removed > 0
                        || report.items_removed > 0
                        || report.bytes_removed > 0
                        || report.acknowledgements_removed > 0
                        || report.tickets_removed > 0
                        || report.issued_tickets_removed > 0
                    {
                        info!(
                            component = "custody",
                            leases_removed = report.leases_removed,
                            items_removed = report.items_removed,
                            bytes_removed = report.bytes_removed,
                            acknowledgements_removed = report.acknowledgements_removed,
                            tickets_removed = report.tickets_removed,
                            issued_tickets_removed = report.issued_tickets_removed,
                            "[ANONYMOUS_MAILBOX] Bounded cleanup completed"
                        );
                    }
                }
                Err(error) => {
                    let reason = Self::custody_cleanup_reason(error);
                    match anonymous_mailbox_custody_cleanup_failure_disposition(error) {
                        AnonymousMailboxCleanupFailureDisposition::Retryable => warn!(
                            component = "custody",
                            reason, "[ANONYMOUS_MAILBOX] Cleanup deferred"
                        ),
                        AnonymousMailboxCleanupFailureDisposition::Fatal => {
                            fatal = true;
                            error!(
                                component = "custody",
                                reason, "[ANONYMOUS_MAILBOX] Cleanup invariant failed"
                            );
                        }
                    }
                }
            }
        }
        if let Some(result) = outcome.source {
            match result {
                Ok(report) => {
                    if report.rows_removed > 0 || report.bytes_removed > 0 {
                        info!(
                            component = "source_journal",
                            rows_removed = report.rows_removed,
                            bytes_removed = report.bytes_removed,
                            "[ANONYMOUS_MAILBOX] Bounded cleanup completed"
                        );
                    }
                }
                Err(error) => {
                    let reason = Self::source_cleanup_reason(error);
                    match anonymous_mailbox_source_cleanup_failure_disposition(error) {
                        AnonymousMailboxCleanupFailureDisposition::Retryable => warn!(
                            component = "source_journal",
                            reason, "[ANONYMOUS_MAILBOX] Cleanup deferred"
                        ),
                        AnonymousMailboxCleanupFailureDisposition::Fatal => {
                            fatal = true;
                            error!(
                                component = "source_journal",
                                reason, "[ANONYMOUS_MAILBOX] Cleanup invariant failed"
                            );
                        }
                    }
                }
            }
        }
        if fatal {
            AnonymousMailboxCleanupLoopDirective::StopFatal
        } else {
            AnonymousMailboxCleanupLoopDirective::Continue
        }
    }

    /// Schedules one non-overlapping cleanup cycle for both anonymous stores.
    ///
    /// The task is absent when both features are disabled. A shutdown received
    /// during spawn_blocking is observed immediately after the bounded cycle,
    /// before a further interval can begin.
    pub(super) fn spawn_anonymous_mailbox_cleanup_task(
        &self,
        custody: Option<Arc<SqliteAnonymousMailboxStore>>,
        source: Option<Arc<SqliteAnonymousMailboxSourceJournal>>,
    ) -> Option<JoinHandle<()>> {
        if custody.is_none() && source.is_none() {
            return None;
        }
        let interval_secs = self.config.memchain.chat_relay.cleanup_interval_secs;
        let shutdown_rx = self.shutdown_tx.subscribe();
        Some(tokio::spawn(async move {
            info!(
                interval_secs,
                "[ANONYMOUS_MAILBOX] Supervised bounded cleanup task started"
            );
            let exit = run_anonymous_mailbox_cleanup_loop(
                Duration::from_secs(interval_secs),
                shutdown_rx,
                || {
                    let custody = custody.clone();
                    let source = source.clone();
                    async move {
                        let now = unix_now_secs();
                        match tokio::task::spawn_blocking(move || {
                            Self::run_anonymous_mailbox_cleanup_cycle(custody, source, now)
                        })
                        .await
                        {
                            Ok(outcome) => Self::observe_anonymous_mailbox_cleanup_cycle(outcome),
                            Err(join_error) => {
                                let reason = if join_error.is_panic() {
                                    "cleanup_worker_panicked"
                                } else {
                                    "cleanup_worker_cancelled"
                                };
                                error!(
                                    component = "blocking_worker",
                                    reason, "[ANONYMOUS_MAILBOX] Cleanup worker failed"
                                );
                                AnonymousMailboxCleanupLoopDirective::StopFatal
                            }
                        }
                    }
                },
            )
            .await;
            if exit == AnonymousMailboxCleanupLoopExit::Fatal {
                error!(
                    reason = "fatal_cleanup_invariant",
                    "[ANONYMOUS_MAILBOX] Required cleanup task stopping"
                );
            }
        }))
    }

    /// Schedules durable `ChatRelay` TTL cleanup without blocking Tokio workers.
    ///
    /// The first interval tick runs immediately so a restarted node enforces
    /// retention before waiting a full interval. Slow cycles never overlap;
    /// missed ticks are skipped instead of replayed in a burst.
    pub(super) fn spawn_chat_relay_cleanup_task(
        &self,
        relay: Arc<ChatRelayService>,
    ) -> JoinHandle<()> {
        let interval_secs = relay.config().cleanup_interval_secs;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(interval_secs));
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            info!(
                interval_secs,
                "[CHAT_RELAY] Durable TTL cleanup task started"
            );

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = timer.tick() => {
                        let cleanup_relay = Arc::clone(&relay);
                        match tokio::task::spawn_blocking(move || cleanup_relay.run_cleanup()).await {
                            Ok(Ok(_)) => {}
                            Ok(Err(error)) => {
                                warn!(
                                    reason = error.reason_bucket(),
                                    "[CHAT_RELAY] Durable TTL cleanup failed"
                                );
                            }
                            Err(join_error) => {
                                let reason = if join_error.is_panic() {
                                    "cleanup_worker_panicked"
                                } else {
                                    "cleanup_worker_cancelled"
                                };
                                relay.record_maintenance_worker_failure(reason);
                                error!(reason, "[CHAT_RELAY] Durable TTL cleanup worker failed");
                            }
                        }
                    }
                }
            }
        })
    }

    /// Schedules bounded Blind Vault expiry cleanup on the blocking pool.
    /// Logs contain aggregate counts only and never lease/object identifiers.
    pub(super) fn spawn_blind_vault_cleanup_task(
        &self,
        vault: Arc<BlindVaultService>,
    ) -> JoinHandle<()> {
        let interval_secs = self.config.blind_vault.cleanup_interval_secs;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(interval_secs));
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            info!(interval_secs, "[BLIND_VAULT] Bounded cleanup task started");

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = timer.tick() => {
                        let cleanup_vault = Arc::clone(&vault);
                        let now_ms = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis()
                            .try_into()
                            .unwrap_or(u64::MAX);
                        match tokio::task::spawn_blocking(move || {
                            cleanup_vault.run_cleanup(now_ms)
                        }).await {
                            Ok(Ok(report)) => {
                                if report.objects_removed > 0
                                    || report.leases_removed > 0
                                    || report.tombstones_removed > 0
                                    || report.lease_tombstones_removed > 0
                                    || report.lease_renewals_removed > 0
                                    || report.admission_spends_removed > 0
                                {
                                    info!(
                                        objects_removed = report.objects_removed,
                                        leases_removed = report.leases_removed,
                                        tombstones_removed = report.tombstones_removed,
                                        lease_tombstones_removed = report.lease_tombstones_removed,
                                        lease_renewals_removed = report.lease_renewals_removed,
                                        admission_spends_removed = report.admission_spends_removed,
                                        "[BLIND_VAULT] Bounded cleanup completed"
                                    );
                                }
                            }
                            Ok(Err(error)) => {
                                warn!(reason = %error, "[BLIND_VAULT] Cleanup failed");
                            }
                            Err(join_error) => {
                                let reason = if join_error.is_panic() {
                                    "cleanup_worker_panicked"
                                } else {
                                    "cleanup_worker_cancelled"
                                };
                                error!(reason, "[BLIND_VAULT] Cleanup worker failed");
                            }
                        }
                    }
                }
            }
        })
    }

    pub(super) fn spawn_cleanup_task(
        &self,
        sessions: Arc<SessionManager>,
        ip_pool: Arc<IpPoolService>,
        routing: Arc<RoutingService>,
        events: SessionEventSender,
        chat_relay: Option<Arc<ChatRelayService>>,
        traffic_tracker: Arc<TrafficTracker>,
        deny_list: Arc<DenyList>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }

                        for termination in sessions.cleanup_expired() {
                            // [SESSION-TERMINATION 2026-08-15 by Codex] Timeout
                            // expiry and signed graceful close must finalize the
                            // exact same external resources.
                            Self::finalize_session_termination(
                                termination,
                                &routing,
                                &events,
                                chat_relay.as_deref(),
                                &traffic_tracker,
                            );
                        }

                        for ip in sessions.drain_cooldown_pool() {
                            ip_pool.release(ip);
                        }

                        // Evict expired deny list entries (QuotaExceeded whose
                        // month has rolled over). NoPremiumAccess entries are
                        // permanent and are only removed by handle_membership_response.
                        deny_list.cleanup();
                    }
                }
            }
        })
    }
}
