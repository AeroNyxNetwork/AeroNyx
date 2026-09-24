// [SERVER-API-RUNTIME-SPLIT 2026-09-25 by Codex] Keep API/router assembly,
// required-listener binding/serving, and management task construction in one
// server-local runtime child while preserving startup and shutdown ordering.
use super::*;

impl Server {
    // ============================================
    // Combined API Server
    // ============================================

    /// Binds every required API socket before returning a runtime task.
    ///
    /// [STARTUP-READINESS 2026-07-29 by Codex] The previous implementation
    /// performed `bind()` inside a detached task. A bind failure therefore
    /// logged an error while `Server::run()` still announced successful
    /// startup. Pre-binding makes listener availability part of the startup
    /// transaction and gives `Type=notify` a truthful readiness barrier.
    pub(super) async fn start_combined_api(
        &self,
        listen_addr: std::net::SocketAddr,
        mpi_state: Option<Arc<MpiState>>,
        ip_pool: Arc<IpPoolService>,
        sessions: Arc<SessionManager>,
        node_policy: Arc<NodePolicyRuntime>,
        voucher_verifier: Arc<VoucherVerifier>,
        encrypted_message_counter: Arc<AtomicU64>,
        packet_handler: Arc<PacketHandler>,
        peer_store: Arc<PeerStore>,
        directory_chain_store: Option<Arc<DirectoryChainStore>>,
        directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
        directory_replica_sync_runtime: Arc<DirectoryReplicaSyncRuntime>,
        chat_relay: Option<Arc<ChatRelayService>>,
        blind_vault: Option<Arc<BlindVaultService>>,
        anonymous_mailbox: Option<Arc<SqliteAnonymousMailboxStore>>,
        anonymous_mailbox_source: Option<Arc<AnonymousMailboxSourceCoordinator>>,
        udp: Arc<UdpTransport>,
        peer_http_clients: &PeerHttpClients,
        commitment_sync_tip_notifier: Option<mpsc::Sender<u64>>,
        anonymous_mailbox_cleanup_runtime_supervised: bool,
        anonymous_mailbox_readiness: AnonymousMailboxReadinessProjection,
        critical_failure_tx: mpsc::Sender<CriticalRuntimeFailure>,
    ) -> Result<JoinHandle<()>> {
        if anonymous_mailbox_source.is_some() && mpi_state.is_none() {
            return Err(ServerError::startup_failed(
                "Anonymous mailbox source requires authenticated VPN MPI runtime",
            ));
        }
        let endpoint_evidence =
            open_endpoint_evidence_store(&self.config.discovery, &self.identity).await?;
        let endpoint_attestation_inbox =
            open_endpoint_attestation_inbox(&self.config.discovery).await?;
        let shutdown_rx = self.shutdown_tx.subscribe();
        let shutdown_rx_vpn = self.shutdown_tx.subscribe();
        let shutdown_rx_public = self.shutdown_tx.subscribe();
        let shutdown_rx_supervisor = self.shutdown_tx.subscribe();
        let runtime_shutdown = Arc::clone(&self.shutdown);
        let vpn_listen_addr =
            Self::vpn_client_api_listen_addr(self.config.gateway_ip(), listen_addr);
        let node_listener = Self::bind_required_api_listener("node_api", listen_addr).await?;
        let vpn_listener =
            Self::bind_required_api_listener("vpn_client_api", vpn_listen_addr).await?;
        let public_api_listener = match self.config.discovery.public_api_listen_addr {
            Some(public_addr) => Some((
                public_addr,
                Self::bind_required_api_listener("public_node_api", public_addr).await?,
            )),
            None => None,
        };
        let vpn_health_config = self.config.clone();
        let anonymous_mailbox_configured =
            self.config.memchain.chat_relay.anonymous_mailbox.enabled
                || self
                    .config
                    .memchain
                    .chat_relay
                    .anonymous_mailbox_source
                    .enabled;
        let discovery_api_policy = DiscoveryApiPolicy::from_config(&self.config.discovery);
        let chat_relay_runtime_ready = chat_relay.is_some();
        let local_capability_status = Self::discovery_local_capability_status_for_runtime(
            &vpn_health_config,
            chat_relay_runtime_ready,
        );
        let node_identity = Arc::new(self.identity.clone());
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Open every
        // private evidence gate before any public route or task is exposed.
        // Disabled mode allocates no DB, key, responder, or scheduler.
        let promotion_runtime = if self
            .config
            .discovery
            .permissionless_endpoint_promotion_enabled
        {
            let prefix = self
                .config
                .discovery
                .permissionless_endpoint_promotion_db_prefix
                .clone();
            let inbox = endpoint_attestation_inbox.clone().ok_or_else(|| {
                ServerError::startup_failed("Endpoint promotion requires attestation inbox")
            })?;
            let store = Arc::clone(&peer_store);
            let identity = Arc::clone(&node_identity);
            Some(Arc::new(
                tokio::task::spawn_blocking(move || {
                    PermissionlessPromotionCoordinator::open(&prefix, store, inbox, identity)
                })
                .await
                .map_err(|_| ServerError::startup_failed("Endpoint promotion unavailable"))?
                .map_err(|_| ServerError::startup_failed("Endpoint promotion unavailable"))?,
            ))
        } else {
            None
        };
        let public_promotion_runtime = promotion_runtime.clone();
        let mut promotion_shutdown = self.shutdown_tx.subscribe();
        let peer_http_client = Arc::clone(&peer_http_clients.control);
        let smoke_peer_store = Arc::clone(&peer_store);
        let smoke_node_identity = Arc::clone(&node_identity);
        let smoke_peer_http_client = Arc::clone(&peer_http_client);
        let smoke_local_capability_status = local_capability_status.clone();
        // [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Operator smokes retain
        // their historical 12-second request budget without stretching the
        // replica synchronizer's 10-second failover deadline. `Some` preserves
        // the existing unavailable-reporting route contract.
        let directory_carrier_smoke_http_client =
            Some(Arc::clone(&peer_http_clients.directory_operator));
        let directory_carrier_smoke_store = directory_replica_store.clone();
        let directory_carrier_smoke_peer_store = Arc::clone(&peer_store);
        let directory_carrier_smoke_identity = Arc::clone(&node_identity);
        let directory_carrier_smoke_gate = Arc::new(TokioMutex::new(()));
        let directory_carrier_smoke_last_started_at = Arc::new(AtomicU64::new(0));
        let directory_cold_bootstrap_http_client = directory_carrier_smoke_http_client.clone();
        let directory_cold_bootstrap_peer_store = Arc::clone(&peer_store);
        let directory_cold_bootstrap_identity = Arc::clone(&node_identity);
        let directory_cold_bootstrap_gate = Arc::new(TokioMutex::new(()));
        let directory_cold_bootstrap_last_started_at = Arc::new(AtomicU64::new(0));
        let commitment_storage = mpi_state.as_ref().and_then(|state| state.storage.clone());
        let commitment_lease_authorized_coordinator =
            self.config.memchain.commitment_sync_coordinator_node_id();
        let public_commitment_sync_tip_notifier = commitment_sync_tip_notifier.clone();
        let directory_chain_sync_peer_ids = self
            .config
            .discovery
            .directory_chain_sync_peer_node_id_bytes();
        let directory_cold_bootstrap_producers = Arc::new(directory_chain_sync_peer_ids.clone());
        let directory_observation_witness_min_verified = self
            .config
            .discovery
            .directory_observation_witness_min_verified;
        let directory_observation_witness_maturity_delay_secs =
            self.config.discovery.directory_chain_sync_interval_secs;
        let directory_full_node_mirror_enabled =
            self.config.discovery.directory_full_node_mirror_enabled;
        let directory_full_node_mirror_max_producers = self
            .config
            .discovery
            .directory_full_node_mirror_max_producers;
        let public_directory_chain_store = directory_chain_store.clone();
        let public_directory_chain_sync_peer_ids = directory_chain_sync_peer_ids.clone();
        let public_directory_observation_witness_min_verified =
            directory_observation_witness_min_verified;
        let public_directory_replica_store = directory_replica_store.clone();
        let public_directory_replica_sync_runtime = Arc::clone(&directory_replica_sync_runtime);
        let public_directory_full_node_mirror_enabled = directory_full_node_mirror_enabled;
        let public_directory_full_node_mirror_max_producers =
            directory_full_node_mirror_max_producers;
        // [BLIND-VAULT-API 2026-07-23 by Codex] Storage activation and client
        // exposure are separate fail-closed decisions. Keep the same gate on
        // both listeners so an operator cannot accidentally expose the vault
        // merely by enabling its local maintenance task.
        let blind_vault_public_api_enabled = self.config.blind_vault.public_api_enabled;
        let public_blind_vault = blind_vault.clone();
        let local_blind_vault = blind_vault.clone();
        let public_anonymous_mailbox = anonymous_mailbox.clone();
        let local_anonymous_mailbox = anonymous_mailbox.clone();
        let vpn_anonymous_mailbox_source = anonymous_mailbox_source.clone();
        let anonymous_mailbox_source_config = self
            .config
            .memchain
            .chat_relay
            .anonymous_mailbox_source
            .clone();
        // [BLIND-VAULT-SHARED-ADMISSION 2026-09-01 by Codex] Both listeners
        // expose the same process capability. They must consume one pressure
        // budget rather than multiplying limits by the number of routers.
        let blind_vault_admission = Arc::new(BlindVaultApiAdmissionRuntime::default());
        // [PERMISSIONLESS-ENDPOINT-PROOF 2026-09-24 by Codex] Only the
        // dedicated public listener receives the descriptor-authenticated
        // candidate router; local, VPN, and node-peer apps remain unchanged.
        let public_endpoint_proof_enabled =
            self.config.discovery.permissionless_endpoint_proof_enabled;
        let public_endpoint_proof_max_entries = self
            .config
            .discovery
            .permissionless_endpoint_proof_max_entries;
        let public_endpoint_proof_ttl_secs =
            self.config.discovery.permissionless_endpoint_proof_ttl_secs;

        Ok(tokio::spawn(async move {
            // [RUNTIME-SUPERVISION 2026-07-29 by Codex] Required listeners
            // live in one JoinSet. No listener may outlive or disappear behind
            // a detached task that the process cannot observe.
            let mut listener_tasks = JoinSet::new();
            if let Some((public_addr, public_listener)) = public_api_listener {
                let mut public_app = build_public_node_router(PublicNodeRouterDependencies {
                    peer_store: Arc::clone(&peer_store),
                    discovery_api_policy: discovery_api_policy.clone(),
                    chat_relay: chat_relay.clone(),
                    sessions: Arc::clone(&sessions),
                    udp: Arc::clone(&udp),
                    node_identity: Arc::clone(&node_identity),
                    peer_http_client: Arc::clone(&peer_http_client),
                    local_capability_status: local_capability_status.clone(),
                    directory_chain_store: public_directory_chain_store,
                    directory_replica_store: public_directory_replica_store,
                    directory_replica_sync_runtime: public_directory_replica_sync_runtime,
                    directory_chain_sync_peer_ids: public_directory_chain_sync_peer_ids,
                    directory_observation_witness_min_verified:
                        public_directory_observation_witness_min_verified,
                    directory_observation_witness_maturity_delay_secs,
                    directory_full_node_mirror_enabled: public_directory_full_node_mirror_enabled,
                    directory_full_node_mirror_max_producers:
                        public_directory_full_node_mirror_max_producers,
                    commitment_storage: commitment_storage.clone(),
                    commitment_lease_authorized_coordinator,
                    commitment_sync_tip_notifier: public_commitment_sync_tip_notifier,
                    blind_vault: public_blind_vault,
                    blind_vault_public_api_enabled,
                    blind_vault_admission: Arc::clone(&blind_vault_admission),
                    anonymous_mailbox: public_anonymous_mailbox
                        .map(|store| store as Arc<dyn AnonymousMailboxCustodyRepository>),
                    endpoint_proof_enabled: public_endpoint_proof_enabled,
                    endpoint_proof_max_entries: public_endpoint_proof_max_entries,
                    endpoint_proof_ttl_secs: public_endpoint_proof_ttl_secs,
                    endpoint_evidence,
                    endpoint_attestation_inbox: endpoint_attestation_inbox.clone(),
                });
                if let Some(runtime) = public_promotion_runtime {
                    public_app = public_app.merge(build_endpoint_possession_responder(Arc::clone(
                        &node_identity,
                    )));
                    let probe_store = Arc::clone(&peer_store);
                    let probe_identity = Arc::clone(&node_identity);
                    let probe_http = Arc::clone(&peer_http_client);
                    // One supervised, bounded candidate per cadence. The
                    // coordinator performs all SQLite work in blocking cells;
                    // errors remain coarse and never disclose node or path.
                    listener_tasks.spawn(async move {
                        let mut interval = tokio::time::interval(Duration::from_secs(15));
                        loop {
                            tokio::select! {
                                _ = promotion_shutdown.recv() => break,
                                _ = interval.tick() => {
                                    match runtime.advance_one().await {
                                        Ok(Some(descriptor)) => {
                                            // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex]
                                            // Promotion is not route readiness. Only an existing
                                            // signed blind-relay control probe can open the gate.
                                            // An identity already in route quarantine cannot
                                            // use descriptor rotation to accelerate recovery.
                                            let now = unix_now_secs();
                                            if probe_store.is_route_quarantined_now(&descriptor.node_id(), now) {
                                                continue;
                                            }
                                            let self_id = probe_identity.public_key_bytes();
                                            Self::probe_blind_relay_candidate_descriptor(
                                                probe_http.as_ref(), probe_store.as_ref(),
                                                probe_identity.as_ref(), &self_id, descriptor, now,
                                                true,
                                            ).await;
                                        }
                                        Ok(None) => {}
                                        Err(reason) => debug!(reason = %reason,
                                            "[DISCOVERY] Permissionless promotion deferred"),
    }
}
                            }
                        }
                        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex]
                        // Share the required-listener JoinSet so a premature
                        // worker exit remains observable by the supervisor.
                        RequiredApiListenerExit {
                            role: "permissionless_endpoint_promotion",
                            address: public_addr,
                            result: Ok(()),
                        }
                    });
                }
                listener_tasks.spawn(async move {
                    Self::serve_public_discovery_api(
                        public_addr,
                        public_listener,
                        public_app,
                        shutdown_rx_public,
                    )
                    .await
                });
            }

            // Encrypted media is a client surface. Keep it on the combined
            // loopback/VPN app and out of `build_public_discovery_router()` so
            // Internet peers cannot use the node-peer listener as a blob host.
            let chat_blob_router = chat_relay
                .as_ref()
                .map(|relay| build_chat_router(Arc::clone(relay)))
                .unwrap_or_else(axum::Router::new);
            let blind_vault_router = match (blind_vault_public_api_enabled, local_blind_vault) {
                (true, Some(vault)) => build_blind_vault_router_with_admission_runtime(
                    vault,
                    Arc::clone(&node_identity),
                    Arc::clone(&blind_vault_admission),
                ),
                _ => axum::Router::new(),
            };
            // [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] The status route
            // receives the exact same mount decision and runtime as the peer
            // route, so service activity cannot be inferred from storage alone.
            let witness_carrier_route_enabled =
                directory_chain_store.is_some() && directory_replica_store.is_some();
            let custody_store_opened = local_anonymous_mailbox.is_some();
            let ticket_terminal_router = build_chat_peer_router_with_anonymous_mailbox(
                chat_relay.clone(),
                Arc::clone(&sessions),
                udp,
                Arc::clone(&peer_store),
                Arc::clone(&node_identity),
                Arc::clone(&peer_http_client),
                blind_vault_public_api_enabled
                    .then(|| blind_vault.clone())
                    .flatten(),
                local_anonymous_mailbox
                    .map(|store| store as Arc<dyn AnonymousMailboxCustodyRepository>),
            );
            let ticket_terminal_wired = custody_store_opened;
            let app = axum::Router::new()
                .merge(build_voice_router(Arc::clone(&sessions)))
                .merge(chat_blob_router)
                .merge(blind_vault_router)
                .merge(build_vpn_health_router_with_anonymous_mailbox_readiness(
                    vpn_health_config,
                    Arc::clone(&ip_pool),
                    Arc::clone(&sessions),
                    node_policy,
                    voucher_verifier,
                    encrypted_message_counter,
                    packet_handler,
                    Arc::clone(&peer_store),
                    chat_relay.clone(),
                    anonymous_mailbox_readiness.clone(),
                ))
                .merge(ticket_terminal_router)
                // Local/VPN-only operator smoke trigger. The public discovery API
                // intentionally does not expose this route; it actively sends a
                // synthetic two-hop onion delivery probe and returns aggregate
                // counters only, never route ids, selected hops, receiver keys,
                // endpoints, encrypted payload bytes, or social graph metadata.
                .route(
                    "/api/discovery/smoke/two-hop",
                    axum::routing::post(move || {
                        let peer_store = Arc::clone(&smoke_peer_store);
                        let identity = Arc::clone(&smoke_node_identity);
                        let client = Arc::clone(&smoke_peer_http_client);
                        let local_capabilities = smoke_local_capability_status.clone();
                        async move {
                            let before_at = unix_now_secs();
                            let before_status = peer_store.status(before_at);
                            let before_runtime = blind_relay_runtime_status_value(
                                before_at,
                                &before_status,
                                &local_capabilities,
                            );
                            let self_node_id = identity.public_key_bytes();
                            let outcome = Self::probe_two_hop_blind_relay_path(
                                &client,
                                &peer_store,
                                &identity,
                                &self_node_id,
                                before_at,
                            )
                            .await;
                            let after_at = unix_now_secs();
                            let after_status = peer_store.status(after_at);
                            let after_runtime = blind_relay_runtime_status_value(
                                after_at,
                                &after_status,
                                &local_capabilities,
                            );
                            axum::Json(serde_json::json!({
                                "success": outcome.terminal_delivery_verified,
                                "contract_version": "two_hop_smoke.v2",
                                "source": "rust_local_operator_smoke",
                                "scope": "local_or_vpn_operator_api_only",
                                "probe": {
                                    "type": "two_hop_onion_delivery",
                                    "payload": "synthetic_opaque_ciphertext",
                                    "ack_boundary": "terminal_chat_relay_store_or_online_delivery",
                                },
                                "outcome": {
                                    "attempted": outcome.attempted,
                                    "route_accepted": outcome.route_accepted,
                                    "terminal_delivery_verified": outcome.terminal_delivery_verified,
                                },
                                "before": before_runtime,
                                "after": after_runtime,
                                "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
                                "privacy_boundary": "operator smoke returns aggregate counters only; no endpoints, route ids, selected hops, receiver keys, encrypted payloads, client IPs, destinations, DNS contents, private keys, wallet-level traffic, or social graph metadata",
                            }))
                        }
                    }),
                )
                // [MIRROR-CARRIER-SMOKE 2026-07-25 by Codex] This route is
                // intentionally absent from `build_public_discovery_router`.
                // Single-flight and cooldown bound work even if a VPN client
                // repeatedly calls the operator listener.
                .route(
                    "/api/discovery/directory/carrier-smoke",
                    axum::routing::post(move || {
                        let store = directory_carrier_smoke_store.clone();
                        let peer_store = Arc::clone(&directory_carrier_smoke_peer_store);
                        let identity = Arc::clone(&directory_carrier_smoke_identity);
                        let client = directory_carrier_smoke_http_client.clone();
                        let gate = Arc::clone(&directory_carrier_smoke_gate);
                        let last_started_at =
                            Arc::clone(&directory_carrier_smoke_last_started_at);
                        async move {
                            let Ok(_permit) = gate.try_lock_owned() else {
                                return (
                                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                                    axum::Json(DirectoryMirrorCarrierSmokeReport::busy()),
                                );
                            };
                            let now = unix_now_secs();
                            let last = last_started_at.load(Ordering::Acquire);
                            let elapsed = now.saturating_sub(last);
                            if last != 0
                                && elapsed < DIRECTORY_MIRROR_CARRIER_SMOKE_COOLDOWN_SECS
                            {
                                return (
                                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                                    axum::Json(DirectoryMirrorCarrierSmokeReport::cooldown(
                                        DIRECTORY_MIRROR_CARRIER_SMOKE_COOLDOWN_SECS
                                            .saturating_sub(elapsed),
                                    )),
                                );
                            }
                            last_started_at.store(now, Ordering::Release);
                            let result = tokio::time::timeout(
                                Duration::from_secs(
                                    DIRECTORY_MIRROR_CARRIER_SMOKE_DEADLINE_SECS,
                                ),
                                run_directory_mirror_carrier_smoke(
                                    store,
                                    &peer_store,
                                    &identity,
                                    client.as_deref(),
                                ),
                            )
                            .await;
                            let (status, report) = match result {
                                Ok(report) if report.success => {
                                    (axum::http::StatusCode::OK, report)
                                }
                                Ok(report) => {
                                    (axum::http::StatusCode::SERVICE_UNAVAILABLE, report)
                                }
                                Err(_) => (
                                    axum::http::StatusCode::GATEWAY_TIMEOUT,
                                    DirectoryMirrorCarrierSmokeReport::unavailable(
                                        "smoke_deadline_exceeded",
                                    ),
                                ),
                            };
                            (status, axum::Json(report))
                        }
                    }),
                )
                // [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] This operator
                // route never touches the live replica store and is omitted
                // from the public discovery listener. It replays one pinned
                // producer genesis page through an explicit signed carrier
                // into a fresh SQLite in-memory store, then drops that store.
                .route(
                    "/api/discovery/directory/carrier-cold-bootstrap-smoke",
                    axum::routing::post(move || {
                        let producers = Arc::clone(&directory_cold_bootstrap_producers);
                        let peer_store = Arc::clone(&directory_cold_bootstrap_peer_store);
                        let identity = Arc::clone(&directory_cold_bootstrap_identity);
                        let client = directory_cold_bootstrap_http_client.clone();
                        let gate = Arc::clone(&directory_cold_bootstrap_gate);
                        let last_started_at =
                            Arc::clone(&directory_cold_bootstrap_last_started_at);
                        async move {
                            let configured_count = producers.len();
                            let Ok(_permit) = gate.try_lock_owned() else {
                                return (
                                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                                    axum::Json(
                                        DirectoryCarrierColdBootstrapSmokeReport::busy(
                                            configured_count,
                                        ),
                                    ),
                                );
                            };
                            let now = unix_now_secs();
                            let last = last_started_at.load(Ordering::Acquire);
                            let elapsed = now.saturating_sub(last);
                            if last != 0
                                && elapsed < DIRECTORY_MIRROR_CARRIER_SMOKE_COOLDOWN_SECS
                            {
                                return (
                                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                                    axum::Json(
                                        DirectoryCarrierColdBootstrapSmokeReport::cooldown(
                                            configured_count,
                                            DIRECTORY_MIRROR_CARRIER_SMOKE_COOLDOWN_SECS
                                                .saturating_sub(elapsed),
                                        ),
                                    ),
                                );
                            }
                            last_started_at.store(now, Ordering::Release);
                            let result = tokio::time::timeout(
                                Duration::from_secs(
                                    DIRECTORY_MIRROR_CARRIER_SMOKE_DEADLINE_SECS,
                                ),
                                run_directory_carrier_cold_bootstrap_smoke(
                                    producers.as_ref(),
                                    &peer_store,
                                    &identity,
                                    client.as_deref(),
                                ),
                            )
                            .await;
                            let (status, report) = match result {
                                Ok(report) if report.success => {
                                    (axum::http::StatusCode::OK, report)
                                }
                                Ok(report) => {
                                    (axum::http::StatusCode::SERVICE_UNAVAILABLE, report)
                                }
                                Err(_) => (
                                    axum::http::StatusCode::GATEWAY_TIMEOUT,
                                    DirectoryCarrierColdBootstrapSmokeReport::unavailable(
                                        configured_count,
                                        "smoke_deadline_exceeded",
                                    ),
                                ),
                            };
                            (status, axum::Json(report))
                        }
                    }),
                )
                // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] Both
                // operator and public gossip surfaces use the same audited
                // replica trust anchor; absent replicas fail proof gossip closed.
                .merge(build_discovery_router_with_local_entry_and_attestation_inbox(
                    Arc::clone(&peer_store),
                    discovery_api_policy,
                    local_capability_status,
                    directory_replica_store.clone(),
                    node_identity.public_key_bytes(),
                    endpoint_attestation_inbox,
                ))
                .merge(build_directory_replica_status_router_with_witness_carrier(
                    directory_replica_store.clone(),
                    Arc::clone(&directory_replica_sync_runtime),
                    directory_chain_sync_peer_ids.clone(),
                    directory_observation_witness_min_verified,
                    directory_observation_witness_maturity_delay_secs,
                    directory_full_node_mirror_enabled,
                    directory_full_node_mirror_max_producers,
                    witness_carrier_route_enabled,
                    DirectoryReplicaStatusScope::LocalOperator,
                ));
            let app = if let Some(store) = directory_chain_store {
                app.merge(build_directory_chain_peer_router_with_replica_and_runtime(
                    store,
                    directory_replica_store,
                    Arc::clone(&peer_store),
                    Arc::clone(&node_identity),
                    directory_chain_sync_peer_ids,
                    directory_full_node_mirror_enabled,
                    directory_replica_sync_runtime,
                ))
            } else {
                app
            };
            let app = if let Some(storage) = commitment_storage {
                app.merge(build_memchain_peer_router_with_runtime(
                    storage,
                    peer_store,
                    node_identity,
                    commitment_lease_authorized_coordinator,
                    commitment_sync_tip_notifier,
                ))
            } else {
                app
            };

            info!("[API] Node API on http://{}", listen_addr);
            info!(
                "[API] Client API also available on http://{} (VPN clients only)",
                vpn_listen_addr
            );
            // [ANONYMOUS-MAILBOX-SOURCE-WIRING 2026-09-03 by Codex] The
            // source request route is an MPI-unified-auth client/VPN-only
            // composition surface. Node-peer, public discovery, ordinary
            // ChatRelay and verified-submit routers retain their exact
            // pre-source route set.
            let source_coordinator_enabled = vpn_anonymous_mailbox_source.is_some();
            let (app, vpn_app, dispatcher_admitted) = if let Some(mpi_state) = mpi_state {
                let node_mpi = build_mpi_router(Arc::clone(&mpi_state));
                let (vpn_mpi, dispatcher_admitted) =
                    if let Some(source) = vpn_anonymous_mailbox_source {
                        let vpn_source_router = build_chat_anonymous_mailbox_source_router(
                            source,
                            Arc::clone(&peer_http_client),
                            &anonymous_mailbox_source_config,
                        );
                        (
                            build_mpi_router_with_source(mpi_state, vpn_source_router),
                            true,
                        )
                    } else {
                        (build_mpi_router(mpi_state), false)
                    };
                (
                    app.clone().merge(node_mpi),
                    app.merge(vpn_mpi),
                    dispatcher_admitted,
                )
            } else {
                (app.clone(), app, false)
            };
            anonymous_mailbox_readiness.publish_local_composition(
                anonymous_mailbox_configured,
                custody_store_opened,
                ticket_terminal_wired,
                source_coordinator_enabled,
                dispatcher_admitted,
                anonymous_mailbox_cleanup_runtime_supervised,
            );
            listener_tasks.spawn(Self::serve_required_api_listener(
                "vpn_client_api",
                vpn_listen_addr,
                vpn_listener,
                vpn_app,
                shutdown_rx_vpn,
            ));
            listener_tasks.spawn(Self::serve_required_api_listener(
                "node_api",
                listen_addr,
                node_listener,
                app,
                shutdown_rx,
            ));

            Self::supervise_required_api_listeners(
                listener_tasks,
                runtime_shutdown,
                shutdown_rx_supervisor,
                critical_failure_tx,
            )
            .await;
        }))
    }

    pub(super) async fn bind_required_api_listener(
        role: &'static str,
        listen_addr: SocketAddr,
    ) -> Result<tokio::net::TcpListener> {
        tokio::net::TcpListener::bind(listen_addr)
            .await
            .map_err(|error| {
                error!(
                    listener_role = role,
                    address = %listen_addr,
                    %error,
                    "[STARTUP] Required API listener bind failed"
                );
                ServerError::startup_failed(format!(
                    "required {role} listener {listen_addr} failed to bind: {error}"
                ))
            })
    }

    pub(super) fn vpn_client_api_listen_addr(
        gateway_ip: Ipv4Addr,
        node_api_listen_addr: std::net::SocketAddr,
    ) -> std::net::SocketAddr {
        // [FAIL-CLOSED-SHUTDOWN-SIGNALS 2026-08-12 by Codex] The validated
        // gateway configuration is already a typed address. Re-parsing a
        // hard-coded default both introduced a panic path and silently ignored
        // legitimate non-default privacy-network ranges.
        std::net::SocketAddr::from((gateway_ip, node_api_listen_addr.port()))
    }

    /// Runs one pre-bound API listener until graceful shutdown or failure.
    pub(super) async fn serve_required_api_listener(
        role: &'static str,
        listen_addr: SocketAddr,
        listener: tokio::net::TcpListener,
        app: axum::Router,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> RequiredApiListenerExit {
        // [RUNTIME-SUPERVISION 2026-07-29 by Codex] Return the terminal result
        // to the listener group instead of logging and discarding it.
        let server = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = shutdown_rx.recv().await;
        });
        let result = server.await;
        match result.as_ref() {
            Ok(()) => info!(
                listener_role = role,
                address = %listen_addr,
                "[API] Required listener stopped"
            ),
            Err(error) => error!(
                listener_role = role,
                address = %listen_addr,
                %error,
                "[API] Required listener failed"
            ),
        }
        RequiredApiListenerExit {
            role,
            address: listen_addr,
            result,
        }
    }

    pub(super) async fn serve_public_discovery_api(
        listen_addr: SocketAddr,
        listener: tokio::net::TcpListener,
        app: axum::Router,
        shutdown_rx: broadcast::Receiver<()>,
    ) -> RequiredApiListenerExit {
        info!(
            "[DISCOVERY] Public node API on http://{} (routes: /api/discovery/*, /api/discovery/peer/directory/*, /api/chat/peer/*, /api/memchain/peer/block-announce, /api/memchain/peer/block-range, /api/memchain/peer/checkpoint, /api/memchain/peer/coordinator-lease, /api/memchain/peer/custody-audit-anchor-witness, /api/discovery/peer/verified-delivery-anchor-witness)",
            listen_addr
        );
        Self::serve_required_api_listener(
            "public_node_api",
            listen_addr,
            listener,
            app,
            shutdown_rx,
        )
        .await
    }

    // ============================================
    // Management Reporter
    // ============================================

    /// Builds the complete management plane and returns ownership to `run`.
    ///
    /// [MANAGEMENT-RUNTIME-OWNERSHIP 2026-07-30 by Codex] Network reporting
    /// failures remain fail-open inside each worker, but disappearance of a
    /// worker is a process-liveness failure and must be supervised centrally.
    // ============================================
    // Management Reporter
    // ============================================

    /// Builds the complete management plane and returns ownership to `run`.
    ///
    /// [MANAGEMENT-RUNTIME-OWNERSHIP 2026-07-30 by Codex] Network reporting
    /// failures remain fail-open inside each worker, but disappearance of a
    /// worker is a process-liveness failure and must be supervised centrally.
    pub(super) async fn init_management_reporter(
        &self,
        sessions: &Arc<SessionManager>,
        ip_pool: Arc<IpPoolService>,
        udp: Arc<UdpTransport>,
        traffic_tracker: Arc<TrafficTracker>,
        deny_list: Arc<DenyList>,
        node_policy: Arc<NodePolicyRuntime>,
        voucher_verifier: Arc<VoucherVerifier>,
        encrypted_message_counter: Arc<AtomicU64>,
        packet_handler: Arc<PacketHandler>,
        peer_store: Arc<PeerStore>,
        memchain_storage: Option<Arc<MemoryStorage>>,
        chat_relay: Option<Arc<ChatRelayService>>,
        blind_vault: Option<Arc<BlindVaultService>>,
        chat_relay_enabled: bool,
        anonymous_mailbox_readiness: AnonymousMailboxReadinessProjection,
    ) -> Result<ManagementRuntime> {
        info!("Initializing management reporting...");

        // [MANAGEMENT-CLIENT-STARTUP 2026-08-12 by Codex] Build the shared
        // connector before spawning workers. A connector/TLS initialization
        // failure must unwind the server startup transaction, not panic or
        // leave only a subset of management tasks alive.
        let mgmt_client = Arc::new(
            ManagementClient::new(self.config.management.clone(), self.identity.clone()).map_err(
                |error| {
                    ServerError::startup_failed(format!(
                        "management HTTP client initialization failed: {error}"
                    ))
                },
            )?,
        );
        info!("Node ID: {}", mgmt_client.node_id());

        let public_ip = self.resolve_public_ip().await;

        let (session_reporter, event_tx) = SessionReporter::new(Arc::clone(&mgmt_client));
        let session_event_sender = SessionEventSender::new(event_tx);

        let (cmd_tx, cmd_rx) = mpsc::channel(COMMAND_CHANNEL_BUFFER);
        let cmd_handler = CommandHandler::new(cmd_rx, Arc::clone(&mgmt_client))
            .with_session_control(Arc::clone(sessions), session_event_sender.clone())
            .with_deny_list(Arc::clone(&deny_list))
            .with_node_policy(Arc::clone(&node_policy))
            // [CHAT-RELAY-BACKUP-COMMAND 2026-08-16 by Codex] The signed CMS
            // command pipeline may trigger a verified local backup, but no
            // public API gains access to the resulting custody artifact.
            .with_chat_relay(chat_relay.clone());
        let cmd_shutdown = self.shutdown_tx.subscribe();
        let command_handler_task = tokio::spawn(async move {
            cmd_handler.run(cmd_shutdown).await;
        });

        let memchain_status_fn: Option<crate::management::reporter::MemChainStatusFn> = if self
            .config
            .memchain
            .is_enabled()
        {
            let allow_remote = self.config.memchain.allow_remote_storage;
            let max_owners = self.config.memchain.max_remote_owners;
            let pinned_witnesses_configured =
                self.config.memchain.commitment_witness_node_ids.len();
            let witness_scope = if pinned_witnesses_configured == 0 {
                "permissionless_evidence"
            } else {
                "operator_pinned"
            };
            let startup_evidence_required =
                self.config.memchain.commitment_witness_startup_required;
            let startup_minimum_verified = self.config.memchain.commitment_witness_min_verified;
            let commitment_storage = memchain_storage.clone();
            Some(Box::new(move || {
                let record_commitment_integrity = commitment_storage.as_ref().map(|storage| {
                    let status = storage.record_commitment_chain_integrity_status();
                    crate::management::client::RecordCommitmentIntegrityHeartbeatStatus {
                        contract_version: status.contract_version,
                        state: status.state.to_string(),
                        baseline_verified_at: status.baseline_verified_at,
                        last_verified_at: status.last_verified_at,
                        verification_duration_ms: status.verification_duration_ms,
                        verified_block_count: status.verified_block_count,
                        verified_commitment_count: status.verified_commitment_count,
                        verified_tip_height: status.verified_tip_height,
                        durability_mode: status.durability_mode,
                        coordinator_fence_state: status.coordinator_fence_state,
                        coordinator_fence_acquired_at: status.coordinator_fence_acquired_at,
                        coordinator_fence_acquisition_failures_total: status
                            .coordinator_fence_acquisition_failures_total,
                        coordinator_fence_scope: status.coordinator_fence_scope,
                        coordinator_lease_state: status.coordinator_lease_state,
                        coordinator_lease_granted_witnesses: status
                            .coordinator_lease_granted_witnesses,
                        coordinator_lease_required_witnesses: status
                            .coordinator_lease_required_witnesses,
                        coordinator_lease_expires_at: status.coordinator_lease_expires_at,
                        coordinator_lease_seconds_remaining: status
                            .coordinator_lease_seconds_remaining,
                        coordinator_lease_production_permitted: status
                            .coordinator_lease_production_permitted,
                        coordinator_lease_last_attempted_at: status
                            .coordinator_lease_last_attempted_at,
                        coordinator_lease_last_renewed_at: status.coordinator_lease_last_renewed_at,
                        coordinator_lease_last_failure_at: status.coordinator_lease_last_failure_at,
                        coordinator_lease_renewal_failures_total: status
                            .coordinator_lease_renewal_failures_total,
                        coordinator_lease_consecutive_failures: status
                            .coordinator_lease_consecutive_failures,
                        coordinator_lease_recoveries_total: status
                            .coordinator_lease_recoveries_total,
                        coordinator_lease_scope: status.coordinator_lease_scope,
                        rollback_guard_state: status.rollback_guard_state,
                        rollback_guard_height: status.rollback_guard_height,
                        rollback_guard_last_verified_at: status.rollback_guard_last_verified_at,
                        rollback_guard_last_persisted_at: status.rollback_guard_last_persisted_at,
                        rollback_guard_write_failures_total: status
                            .rollback_guard_write_failures_total,
                    }
                });
                let record_commitment_sync = commitment_storage.as_ref().map(|storage| {
                    let status = storage.record_commitment_sync_status();
                    crate::management::client::RecordCommitmentSyncHeartbeatStatus {
                        contract_version: status.contract_version,
                        role: status.role,
                        state: status.state,
                        follower_readiness_state: status.follower_readiness_state,
                        follower_fully_ready: status.follower_fully_ready,
                        // [FOLLOWER-READINESS-FRESHNESS 2026-07-30 by Codex]
                        // Preserve the storage layer's already-derived,
                        // identity-blind readiness deadline.
                        follower_convergence_confirmed_at: status.follower_convergence_confirmed_at,
                        follower_readiness_stale_after: status.follower_readiness_stale_after,
                        enabled: status.enabled,
                        last_trigger: status.last_trigger,
                        last_announcement_at: status.last_announcement_at,
                        last_announced_height: status.last_announced_height,
                        last_announcement_result: status.last_announcement_result,
                        announcements_accepted_total: status.announcements_accepted_total,
                        announcements_coalesced_total: status.announcements_coalesced_total,
                        announcements_stale_total: status.announcements_stale_total,
                        announcements_unavailable_total: status.announcements_unavailable_total,
                        last_outbound_announcement_at: status.last_outbound_announcement_at,
                        last_outbound_announced_height: status.last_outbound_announced_height,
                        last_outbound_announcement_result: status.last_outbound_announcement_result,
                        outbound_announcement_rounds_total: status
                            .outbound_announcement_rounds_total,
                        outbound_announcement_rounds_skipped_total: status
                            .outbound_announcement_rounds_skipped_total,
                        outbound_announcement_rounds_superseded_total: status
                            .outbound_announcement_rounds_superseded_total,
                        outbound_announcements_attempted_total: status
                            .outbound_announcements_attempted_total,
                        outbound_announcements_accepted_total: status
                            .outbound_announcements_accepted_total,
                        outbound_announcements_stale_total: status
                            .outbound_announcements_stale_total,
                        outbound_announcements_failed_total: status
                            .outbound_announcements_failed_total,
                        outbound_announcement_retries_attempted_total: status
                            .outbound_announcement_retries_attempted_total,
                        outbound_announcement_retries_succeeded_total: status
                            .outbound_announcement_retries_succeeded_total,
                        outbound_announcement_retries_exhausted_total: status
                            .outbound_announcement_retries_exhausted_total,
                        // [FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex]
                        // Forward only the storage layer's source-blind
                        // aggregate contract into the signed heartbeat.
                        last_block_page_pull_at: status.last_block_page_pull_at,
                        last_block_page_pull_result: status.last_block_page_pull_result,
                        last_block_carrier_recovered_at: status.last_block_carrier_recovered_at,
                        block_page_pulls_total: status.block_page_pulls_total,
                        block_page_coordinator_success_total: status
                            .block_page_coordinator_success_total,
                        block_carrier_attempts_total: status.block_carrier_attempts_total,
                        block_carrier_recoveries_total: status.block_carrier_recoveries_total,
                        block_page_availability_exhausted_total: status
                            .block_page_availability_exhausted_total,
                        block_page_security_stops_total: status.block_page_security_stops_total,
                        last_block_page_security_stop_at: status.last_block_page_security_stop_at,
                        block_carrier_cooling_slots: status.block_carrier_cooling_slots,
                        block_carrier_cooldown_skips_total: status
                            .block_carrier_cooldown_skips_total,
                        block_carrier_half_open_attempts_total: status
                            .block_carrier_half_open_attempts_total,
                        certificate_policy_state: status.certificate_policy_state,
                        certificate_policy_ready: status.certificate_policy_ready,
                        certificate_policy_last_evaluated_at: status
                            .certificate_policy_last_evaluated_at,
                        certificate_policy_evaluated_tip_height: status
                            .certificate_policy_evaluated_tip_height,
                        certificate_witnesses_configured: status.certificate_witnesses_configured,
                        certificate_minimum_signers: status.certificate_minimum_signers,
                        last_certificate_sync_at: status.last_certificate_sync_at,
                        last_certificate_sync_result: status.last_certificate_sync_result,
                        last_certificate_carrier_recovered_at: status
                            .last_certificate_carrier_recovered_at,
                        certificate_sync_rounds_total: status.certificate_sync_rounds_total,
                        certificate_coordinator_success_total: status
                            .certificate_coordinator_success_total,
                        certificate_carrier_attempts_total: status
                            .certificate_carrier_attempts_total,
                        certificate_carrier_recoveries_total: status
                            .certificate_carrier_recoveries_total,
                        certificate_verified_unpersisted_total: status
                            .certificate_verified_unpersisted_total,
                        certificate_availability_exhausted_total: status
                            .certificate_availability_exhausted_total,
                        certificate_security_stops_total: status.certificate_security_stops_total,
                        last_certificate_security_stop_at: status.last_certificate_security_stop_at,
                        certificate_carrier_cooling_slots: status.certificate_carrier_cooling_slots,
                        certificate_carrier_cooldown_skips_total: status
                            .certificate_carrier_cooldown_skips_total,
                        certificate_carrier_half_open_attempts_total: status
                            .certificate_carrier_half_open_attempts_total,
                        // [CERTIFICATE-BACKFILL-TELEMETRY 2026-07-29 by Codex]
                        // Coordinator recovery remains a separate,
                        // source-blind domain from follower certificate sync.
                        last_coordinator_certificate_backfill_at: status
                            .last_coordinator_certificate_backfill_at,
                        last_coordinator_certificate_backfill_result: status
                            .last_coordinator_certificate_backfill_result,
                        coordinator_certificate_backfill_rounds_total: status
                            .coordinator_certificate_backfill_rounds_total,
                        coordinator_certificate_backfill_persisted_total: status
                            .coordinator_certificate_backfill_persisted_total,
                        coordinator_certificate_backfill_verified_unpersisted_total: status
                            .coordinator_certificate_backfill_verified_unpersisted_total,
                        coordinator_certificate_backfill_availability_exhausted_total: status
                            .coordinator_certificate_backfill_availability_exhausted_total,
                        coordinator_certificate_backfill_security_stops_total: status
                            .coordinator_certificate_backfill_security_stops_total,
                        last_coordinator_certificate_backfill_security_stop_at: status
                            .last_coordinator_certificate_backfill_security_stop_at,
                        coordinator_certificate_backfill_carrier_attempts_total: status
                            .coordinator_certificate_backfill_carrier_attempts_total,
                        coordinator_certificate_backfill_carrier_cooling_slots: status
                            .coordinator_certificate_backfill_carrier_cooling_slots,
                        coordinator_certificate_backfill_carrier_cooldown_skips_total: status
                            .coordinator_certificate_backfill_carrier_cooldown_skips_total,
                        coordinator_certificate_backfill_carrier_half_open_attempts_total: status
                            .coordinator_certificate_backfill_carrier_half_open_attempts_total,
                        last_attempt_at: status.last_attempt_at,
                        last_success_at: status.last_success_at,
                        last_failure_at: status.last_failure_at,
                        last_recovered_at: status.last_recovered_at,
                        next_poll_at: status.next_poll_at,
                        consecutive_failures: status.consecutive_failures,
                        last_error_code: status.last_error_code,
                        remote_tip_height: status.remote_tip_height,
                        pages_received_total: status.pages_received_total,
                        blocks_received_total: status.blocks_received_total,
                        failure_events_total: status.failure_events_total,
                        recovery_events_total: status.recovery_events_total,
                    }
                });
                let record_commitment_checkpoint = commitment_storage.as_ref().map(|storage| {
                    let status = storage.record_commitment_checkpoint_status();
                    crate::management::client::RecordCommitmentCheckpointHeartbeatStatus {
                        contract_version: status.contract_version,
                        witness_scope,
                        pinned_witnesses_configured,
                        startup_evidence_required,
                        startup_minimum_verified,
                        state: status.state,
                        last_checked_at: status.last_checked_at,
                        last_converged_at: status.last_converged_at,
                        last_divergence_at: status.last_divergence_at,
                        last_failure_at: status.last_failure_at,
                        last_served_at: status.last_served_at,
                        local_tip_height: status.local_tip_height,
                        remote_tip_height: status.remote_tip_height,
                        proofs_verified_total: status.proofs_verified_total,
                        proofs_failed_total: status.proofs_failed_total,
                        divergences_total: status.divergences_total,
                        requests_served_total: status.requests_served_total,
                        evidence_state: status.evidence_state,
                        evidence_records: status.evidence_records,
                        applicable_evidence_records: status.applicable_evidence_records,
                        deferred_evidence_records: status.deferred_evidence_records,
                        divergence_evidence_records: status.divergence_evidence_records,
                        equivocation_incidents: status.equivocation_incidents,
                        trusted_divergence_incidents: status.trusted_divergence_incidents,
                        checkpoint_certificates: status.checkpoint_certificates,
                        latest_certified_height: status.latest_certified_height,
                        latest_certificate_signers: status.latest_certificate_signers,
                        latest_certificate_required_signers: status
                            .latest_certificate_required_signers,
                        block_confirmation_state: status.block_confirmation_state,
                        uncertified_block_count: status.uncertified_block_count,
                        block_confirmation_policy: status.block_confirmation_policy,
                        certificate_rollback_guard_state: status.certificate_rollback_guard_state,
                        certificate_rollback_guard_height: status.certificate_rollback_guard_height,
                        certificate_rollback_guard_last_verified_at: status
                            .certificate_rollback_guard_last_verified_at,
                        certificate_rollback_guard_last_persisted_at: status
                            .certificate_rollback_guard_last_persisted_at,
                        certificate_rollback_guard_write_failures_total: status
                            .certificate_rollback_guard_write_failures_total,
                        certificate_rollback_guard_scope: status.certificate_rollback_guard_scope,
                        production_halted: status.production_halted,
                        last_evidence_at: status.last_evidence_at,
                        observation_freshness: status.observation_freshness,
                        observation_age_seconds: status.observation_age_seconds,
                        freshness_window_seconds: status.freshness_window_seconds,
                        last_round_state: status.last_round_state,
                        last_round_at: status.last_round_at,
                        last_round_eligible: status.last_round_eligible,
                        last_round_attempted: status.last_round_attempted,
                        last_round_verified: status.last_round_verified,
                        last_round_failed: status.last_round_failed,
                        last_round_converged: status.last_round_converged,
                        last_round_remote_ahead: status.last_round_remote_ahead,
                        last_round_remote_behind: status.last_round_remote_behind,
                        last_round_diverged: status.last_round_diverged,
                        evidence_persistence_failures_total: status
                            .evidence_persistence_failures_total,
                    }
                });
                Some(crate::management::client::MemChainHeartbeatStatus {
                    enabled: true,
                    allow_remote_storage: allow_remote,
                    max_remote_owners: max_owners,
                    current_remote_owners: 0,
                    record_commitment_integrity,
                    record_commitment_sync,
                    record_commitment_checkpoint,
                })
            }))
        } else {
            None
        };

        // Note: .with_sessions / .with_traffic_tracker / .with_udp are
        // injected here — all three are available at this call site.
        let mut heartbeat = HeartbeatReporter::new(Arc::clone(&mgmt_client), public_ip)
            .with_command_sender(cmd_tx)
            .with_sessions(Arc::clone(sessions))
            .with_traffic_tracker(Arc::clone(&traffic_tracker))
            .with_udp(Arc::clone(&udp))
            .with_deny_list(Arc::clone(&deny_list))
            .with_node_policy(Arc::clone(&node_policy));

        if let Some(f) = memchain_status_fn {
            heartbeat = heartbeat.with_memchain_status(f);
        }

        let vpn_health_config = self.config.clone();
        let vpn_health_ip_pool = Arc::clone(&ip_pool);
        let vpn_health_sessions = Arc::clone(sessions);
        let vpn_health_policy = Arc::clone(&node_policy);
        let vpn_health_verifier = Arc::clone(&voucher_verifier);
        let vpn_health_message_counter = Arc::clone(&encrypted_message_counter);
        let vpn_health_packet_handler = Arc::clone(&packet_handler);
        let vpn_health_peer_store = Arc::clone(&peer_store);
        let vpn_health_chat_relay = chat_relay.clone();
        let vpn_health_anonymous_mailbox_readiness = anonymous_mailbox_readiness.clone();
        heartbeat = heartbeat.with_vpn_health_status(Box::new(move || {
            let config = vpn_health_config.clone();
            let ip_pool = Arc::clone(&vpn_health_ip_pool);
            let sessions = Arc::clone(&vpn_health_sessions);
            let node_policy = Arc::clone(&vpn_health_policy);
            let verifier = Arc::clone(&vpn_health_verifier);
            let message_counter = Arc::clone(&vpn_health_message_counter);
            let packet_handler = Arc::clone(&vpn_health_packet_handler);
            let peer_store = Arc::clone(&vpn_health_peer_store);
            let chat_relay = vpn_health_chat_relay.clone();
            let anonymous_mailbox_readiness = vpn_health_anonymous_mailbox_readiness.clone();
            Box::pin(async move {
                Some(
                    collect_vpn_health_value_with_anonymous_mailbox_readiness(
                        config,
                        ip_pool,
                        sessions,
                        node_policy,
                        verifier,
                        message_counter,
                        packet_handler,
                        peer_store,
                        chat_relay,
                        anonymous_mailbox_readiness,
                    )
                    .await,
                )
            })
        }));

        let operator_status_config = self.config.clone();
        let operator_status_ip_pool = Arc::clone(&ip_pool);
        let operator_status_sessions = Arc::clone(sessions);
        let operator_status_policy = Arc::clone(&node_policy);
        let operator_status_verifier = Arc::clone(&voucher_verifier);
        let operator_status_message_counter = Arc::clone(&encrypted_message_counter);
        let operator_status_packet_handler = Arc::clone(&packet_handler);
        let operator_status_peer_store = Arc::clone(&peer_store);
        let operator_status_chat_relay = chat_relay.clone();
        let operator_status_anonymous_mailbox_readiness = anonymous_mailbox_readiness.clone();
        heartbeat = heartbeat.with_operator_status(Box::new(move || {
            let config = operator_status_config.clone();
            let ip_pool = Arc::clone(&operator_status_ip_pool);
            let sessions = Arc::clone(&operator_status_sessions);
            let node_policy = Arc::clone(&operator_status_policy);
            let verifier = Arc::clone(&operator_status_verifier);
            let message_counter = Arc::clone(&operator_status_message_counter);
            let packet_handler = Arc::clone(&operator_status_packet_handler);
            let peer_store = Arc::clone(&operator_status_peer_store);
            let chat_relay = operator_status_chat_relay.clone();
            let anonymous_mailbox_readiness = operator_status_anonymous_mailbox_readiness.clone();
            Box::pin(async move {
                Some(
                    collect_node_operator_status_value_with_anonymous_mailbox_readiness(
                        config,
                        ip_pool,
                        sessions,
                        node_policy,
                        verifier,
                        message_counter,
                        packet_handler,
                        peer_store,
                        chat_relay,
                        anonymous_mailbox_readiness,
                    )
                    .await,
                )
            })
        }));

        let discovery_status_config = self.config.clone();
        let discovery_status_peer_store = Arc::clone(&peer_store);
        let discovery_chat_relay_runtime_ready = chat_relay.is_some();
        let discovery_status_blind_vault = blind_vault.clone();
        heartbeat = heartbeat.with_discovery_status(Box::new(move || {
            let config = discovery_status_config.clone();
            let peer_store = Arc::clone(&discovery_status_peer_store);
            let blind_vault = discovery_status_blind_vault.clone();
            Box::pin(async move {
                let now = unix_now_secs();
                let status = peer_store.status(now);
                let blind_vault_runtime_ready =
                    Self::observe_blind_vault_admission_readiness(blind_vault, now).await;
                let local_capabilities = Self::discovery_local_capability_status_for_runtime_state(
                    &config,
                    discovery_chat_relay_runtime_ready,
                    blind_vault_runtime_ready,
                );
                let signed_peer_records = peer_store.export_signed_peer_records_for_heartbeat(
                    now,
                    Some(
                        config
                            .discovery
                            .max_snapshot_limit
                            .min(HEARTBEAT_SIGNED_PEER_RECORD_LIMIT),
                    ),
                );
                Some(discovery_heartbeat_status_value(
                    now,
                    &status,
                    &local_capabilities,
                    signed_peer_records,
                ))
            })
        }));

        if let Some(relay) = chat_relay.as_ref() {
            let chat_relay_status: Arc<ChatRelayService> = Arc::clone(relay);
            let custody_witness_runtime = Arc::clone(&self.custody_witness_runtime);
            let chat_relay_anonymous_mailbox_readiness = anonymous_mailbox_readiness.clone();
            heartbeat = heartbeat.with_chat_relay_status(Box::new(move || {
                let relay = Arc::clone(&chat_relay_status);
                let custody_witness_runtime = Arc::clone(&custody_witness_runtime);
                let anonymous_mailbox_readiness =
                    chat_relay_anonymous_mailbox_readiness.clone();
                Box::pin(async move {
                    let now = unix_now_secs();
                    let storage_usage = relay.storage_usage().ok();
                    let config = relay.config();
                    let mut peer_relay = relay.peer_status();
                    anonymous_mailbox_readiness.apply_to(&mut peer_relay);
                    Some(serde_json::json!({
                        "generated_at": now,
                        "peer_relay": peer_relay,
                        // [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex]
                        // This process-lifetime snapshot exposes only fixed
                        // reasons, timestamps and node-wide aggregate counts.
                        "custody_witness": custody_witness_runtime.snapshot(),
                        "maintenance": relay.maintenance_status(),
                        "storage_usage": storage_usage,
                        "storage_capacity": {
                            "max_pending_messages_total": config.max_pending_messages_total,
                            "max_pending_message_bytes_total": config.max_pending_message_bytes_total,
                            "max_pending_blobs_total": config.max_pending_blobs_total,
                            "max_pending_blob_bytes_total": config.max_pending_blob_bytes_total
                        },
                        "source": "rust_chat_relay_service",
                        "privacy_boundary": "aggregate encrypted chat relay counters, custody witness runtime health, fixed reason buckets, timestamps, and node-wide storage capacity only; no message ids, wallet ids, sender/receiver keys, witness identities, witness endpoints, signatures, anchors, blob ids, session ids, client IPs, destinations, DNS contents, packet payloads, chat plaintext, ciphertext, private keys, voucher secrets, or per-user traffic"
                    }))
                })
            }));
        } else if !chat_relay_enabled {
            let custody_witness_runtime = Arc::clone(&self.custody_witness_runtime);
            let chat_relay_anonymous_mailbox_readiness = anonymous_mailbox_readiness.clone();
            heartbeat = heartbeat.with_chat_relay_status(Box::new(move || {
                let custody_witness_runtime = Arc::clone(&custody_witness_runtime);
                let anonymous_mailbox_readiness =
                    chat_relay_anonymous_mailbox_readiness.clone();
                Box::pin(async move {
                    let now = unix_now_secs();
                    let mut peer_relay = ChatRelayPeerStatus::new(false);
                    anonymous_mailbox_readiness.apply_to(&mut peer_relay);
                    Some(serde_json::json!({
                        "generated_at": now,
                        // [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex]
                        // Reuse the typed schema so disabled heartbeats cannot
                        // silently omit newly added aggregate health fields.
                        "peer_relay": peer_relay,
                        "custody_witness": custody_witness_runtime.snapshot(),
                        "maintenance": null,
                        "storage_usage": null,
                        "storage_capacity": null,
                        "source": "rust_chat_relay_disabled_config",
                        "privacy_boundary": "aggregate encrypted chat relay counters, custody witness runtime health, fixed reason buckets, timestamps, and node-wide storage capacity only; no message ids, wallet ids, sender/receiver keys, witness identities, witness endpoints, signatures, anchors, blob ids, session ids, client IPs, destinations, DNS contents, packet payloads, chat plaintext, ciphertext, private keys, voucher secrets, or per-user traffic"
                    }))
                })
            }));
        }

        let sess = Arc::clone(sessions);
        let hb_shutdown = self.shutdown_tx.subscribe();
        let heartbeat_task = tokio::spawn(async move {
            heartbeat
                .run(move || sess.count() as u32, hb_shutdown)
                .await;
        });

        let sr_shutdown = self.shutdown_tx.subscribe();
        let session_reporter_task = tokio::spawn(async move {
            session_reporter.run(sr_shutdown).await;
        });

        info!("[MANAGEMENT] Reporting started");
        Ok(ManagementRuntime {
            session_events: session_event_sender,
            tasks: [
                ("management-command-handler", command_handler_task),
                ("management-heartbeat", heartbeat_task),
                ("management-session-reporter", session_reporter_task),
            ],
        })
    }
}
