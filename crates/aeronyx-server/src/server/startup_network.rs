// [ARCH-SPLIT 2026-10-02]
// Public address, peer store, and discovery self-check. Bootstrap file import lives next to this call.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl Server {
    // ============================================
    // Public IP Resolution
    // ============================================

    pub(super) async fn resolve_public_ip(&self) -> String {
        if let Some(ip) = self.config.network.public_ip() {
            return ip.to_string();
        }

        let services = [
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://ipinfo.io/ip",
            "http://169.254.169.254/latest/meta-data/public-ipv4",
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
        ];

        if let Ok(client) = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            for url in &services {
                let mut req = client.get(*url);
                if url.contains("metadata.google.internal") {
                    req = req.header("Metadata-Flavor", "Google");
                }
                if let Ok(resp) = req.send().await {
                    if resp.status().is_success() {
                        match read_bounded_http_response(resp, PUBLIC_IP_RESPONSE_MAX_BYTES).await {
                            Ok(body) => {
                                let Ok(ip_str) = std::str::from_utf8(&body) else {
                                    debug!(source = %url, "[NET] Public IP response was not UTF-8");
                                    continue;
                                };
                                let ip_str = ip_str.trim();
                                if ip_str.len() <= 45 {
                                    if let Ok(addr) = ip_str.parse::<std::net::IpAddr>() {
                                        let is_private = match addr {
                                            std::net::IpAddr::V4(v4) => {
                                                v4.is_loopback()
                                                    || v4.is_private()
                                                    || v4.is_unspecified()
                                            }
                                            std::net::IpAddr::V6(v6) => {
                                                v6.is_loopback() || v6.is_unspecified()
                                            }
                                        };
                                        if !is_private {
                                            info!(ip = %addr, source = %url, "[NET] Public IP detected");
                                            return addr.to_string();
                                        }
                                        warn!(ip = %addr, source = %url, "[NET] Ignoring private/loopback IP");
                                    }
                                }
                            }
                            Err(error) => {
                                debug!(
                                    source = %url,
                                    reason = error.as_str(),
                                    "[NET] Public IP response rejected"
                                );
                            }
                        }
                    }
                }
            }
        }

        let fallback = self.config.listen_addr().ip().to_string();
        warn!(ip = %fallback, "[NET] Fallback to listen address");
        fallback
    }

    // ============================================
    // Core Services
    // ============================================

    pub(super) async fn init_peer_store(
        &self,
        chat_relay_runtime_ready: bool,
        control_http_client: &reqwest::Client,
    ) -> Result<Arc<PeerStore>> {
        self.init_peer_store_with_storage_runtime(
            chat_relay_runtime_ready,
            self.config.blind_vault.replica_advertisement_configured(),
            false,
            control_http_client,
        )
        .await
    }

    /// Installs the untrusted-gossip admission boundary before any startup
    /// cache read, network await, or outbound gossip task can import a frame.
    pub(super) fn new_discovery_peer_store() -> Arc<PeerStore> {
        // [PERMISSIONLESS-DISCOVERY-STARTUP-GATE 2026-09-24 by Codex] Router
        // construction repeats this idempotent switch later, but may occur
        // after an immediate outbound gossip snapshot response.
        let peer_store = Arc::new(PeerStore::new());
        peer_store.enable_untrusted_discovery_candidate_mode();
        peer_store
    }

    /// Initializes discovery with explicitly observed storage readiness.
    ///
    /// The compatibility wrapper above remains for focused tests and embedded
    /// callers that do not own a running Blind Vault service.
    pub(super) async fn init_peer_store_with_storage_runtime(
        &self,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
        control_http_client: &reqwest::Client,
    ) -> Result<Arc<PeerStore>> {
        let peer_store = Self::new_discovery_peer_store();
        peer_store.set_max_peers(Some(self.config.discovery.max_peers));
        peer_store.configure_verified_delivery_witness_requesters(
            &self
                .config
                .discovery
                .verified_delivery_witness_requester_node_id_bytes(),
        );
        // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Install custody pins
        // independently. A delivery-witness relationship must never imply
        // authority to advance durable custody evidence.
        peer_store.configure_custody_audit_witness_requesters(
            &self
                .config
                .discovery
                .custody_audit_witness_requester_node_id_bytes(),
        );
        // [ROUTE-DOMAIN-ATTESTED-SELECTION 2026-08-03 by Codex] Install the
        // verifier-local policy before importing any peer descriptor. Invalid
        // policy is a startup failure; silently disabling strict mode would be
        // a privacy downgrade. No trust-root identity is logged here.
        let route_domains = self
            .config
            .discovery
            .pinned_route_domain_assignments()
            .into_iter()
            .map(|assignment| (assignment.node_id, assignment.route_domain))
            .collect::<Vec<_>>();
        peer_store
            .configure_route_domain_attestor_policy(
                &route_domains,
                &self.config.discovery.route_domain_attestor_node_id_bytes(),
                self.config.discovery.route_domain_attestation_min_verified,
                self.config
                    .discovery
                    .require_route_domain_attestations_for_multi_hop,
            )
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "route-domain attestor policy initialization failed: {error}"
                ))
            })?;
        peer_store.configure_bootstrap_status(
            self.config.discovery.enabled,
            self.config.discovery.peer_cache_path.is_some(),
            self.config.discovery.gossip_enabled,
            self.config.discovery.seed_endpoints.len(),
        );
        let now = unix_now_secs();
        let (self_check_status, self_check_detail) =
            Self::discovery_startup_self_check(&self.config);
        peer_store.record_startup_self_check(now, self_check_status, self_check_detail.clone());
        if self_check_status == "warning" {
            warn!(
                detail = %self_check_detail,
                "[DISCOVERY] Startup self-check warning"
            );
        } else {
            info!(
                status = %self_check_status,
                detail = %self_check_detail,
                "[DISCOVERY] Startup self-check complete"
            );
        }
        if !self.config.discovery.enabled {
            info!("[DISCOVERY] Bootstrap disabled");
            peer_store.record_bootstrap_source(now, "config", "skipped", "discovery_enabled=false");
            return Ok(peer_store);
        }

        if let Some(path) = &self.config.discovery.bootstrap_snapshot_path {
            match Self::read_bounded_file(Path::new(path), DISCOVERY_SNAPSHOT_MAX_BYTES).await {
                Ok(bytes) => {
                    Self::import_bootstrap_snapshot_bytes(
                        &peer_store,
                        "file",
                        path,
                        &bytes,
                        now,
                        None,
                    );
                }
                Err(e) => {
                    peer_store.record_bootstrap_source(
                        now,
                        "file",
                        "failed",
                        Self::bounded_file_error_reason(&e),
                    );
                    warn!(
                        source = %path,
                        error = %e,
                        "[DISCOVERY] Failed to read bootstrap snapshot"
                    );
                }
            }
        }

        if let Some(url) = &self.config.discovery.bootstrap_snapshot_url {
            match reqwest::Client::builder()
                .timeout(Duration::from_secs(
                    self.config.discovery.fetch_timeout_secs,
                ))
                .build()
            {
                Ok(client) => match client.get(url).send().await {
                    Ok(response) => {
                        let status = response.status();
                        if status.is_success() {
                            match read_bounded_http_response(response, DISCOVERY_SNAPSHOT_MAX_BYTES)
                                .await
                            {
                                Ok(bytes) => {
                                    Self::import_bootstrap_snapshot_bytes(
                                        &peer_store,
                                        "url",
                                        url,
                                        &bytes,
                                        now,
                                        None,
                                    );
                                }
                                Err(e) => {
                                    peer_store.record_bootstrap_source(
                                        now,
                                        "url",
                                        "failed",
                                        e.as_str(),
                                    );
                                    warn!(
                                        source = %url,
                                        reason = e.as_str(),
                                        "[DISCOVERY] Bootstrap response rejected"
                                    );
                                }
                            }
                        } else {
                            peer_store.record_bootstrap_source(
                                now,
                                "url",
                                "failed",
                                format!("http_status={status}"),
                            );
                            warn!(
                                source = %url,
                                status = %status,
                                "[DISCOVERY] Bootstrap URL returned non-success status"
                            );
                        }
                    }
                    Err(e) => {
                        peer_store.record_bootstrap_source(now, "url", "failed", "fetch_failed");
                        warn!(
                            source = %url,
                            error = %e,
                            "[DISCOVERY] Failed to fetch bootstrap snapshot"
                        );
                    }
                },
                Err(e) => {
                    peer_store.record_bootstrap_source(
                        now,
                        "url",
                        "failed",
                        "http_client_build_failed",
                    );
                    warn!(
                        error = %e,
                        "[DISCOVERY] Failed to build bootstrap HTTP client"
                    );
                }
            }
        }

        if let Some(path) = &self.config.discovery.peer_cache_path {
            self.load_peer_cache(&peer_store, path, now).await;
        }

        if self.config.discovery.advertise_self {
            crate::services::onion_keys::tick_rotation(now);
            match self.build_self_discovery_descriptor_with_runtime_state(
                now,
                chat_relay_runtime_ready,
                blind_vault_runtime_ready,
                anonymous_mailbox_runtime_ready,
            ) {
                Ok(descriptor) => match peer_store
                    .upsert_verified_from_source(descriptor, now, "self")
                {
                    Ok(true) => {
                        peer_store.record_self_descriptor_status(now, "success", "registered");
                        info!("[DISCOVERY] Self descriptor registered");
                    }
                    Ok(false) => {
                        peer_store.record_self_descriptor_status(now, "success", "already_current");
                        info!("[DISCOVERY] Self descriptor already current");
                    }
                    Err(e) => {
                        peer_store.record_self_descriptor_status(
                            now,
                            "failed",
                            "peer_store_rejected",
                        );
                        warn!(
                            error = %e,
                            "[DISCOVERY] Self descriptor rejected by local PeerStore"
                        );
                    }
                },
                Err(e) => {
                    peer_store.record_self_descriptor_status(now, "failed", "build_failed");
                    warn!(
                        error = %e,
                        "[DISCOVERY] Failed to build self descriptor"
                    );
                }
            }
        }

        let snapshot = peer_store.snapshot(now);
        info!(
            total_peers = snapshot.total_peers,
            valid_peers = snapshot.valid_peers,
            public_peers = snapshot.public_peers,
            public_exit_peers = snapshot.public_exit_peers,
            "[DISCOVERY] PeerStore bootstrap complete"
        );

        let custody_witness_ids = self.config.discovery.custody_audit_witness_node_id_bytes();
        if !custody_witness_ids.is_empty() {
            // [CUSTODY-WITNESS-PLANNER 2026-08-16 by Codex] Planning is a
            // local, read-only startup check. No anchor, identity list, digest,
            // endpoint, or request leaves the process.
            match plan_custody_audit_witnesses(
                &peer_store,
                &self.identity.public_key_bytes(),
                &custody_witness_ids,
                self.config.discovery.custody_audit_witness_min_verified,
                now,
            ) {
                Ok(plan) => info!(
                    configured = plan.configured,
                    eligible = plan.eligible,
                    unavailable = plan.unavailable,
                    minimum_verified = plan.minimum_verified,
                    quorum_ready = plan.quorum_ready,
                    self_excluded = plan.self_excluded,
                    duplicates_ignored = plan.duplicates_ignored,
                    "[MEMCHAIN] Custody witness dry-run plan evaluated"
                ),
                Err(reason) => warn!(reason, "[MEMCHAIN] Custody witness dry-run policy rejected"),
            }
        }

        if let Some(path) = &self.config.discovery.peer_cache_path {
            let cache_save_at = unix_now_secs();
            match Self::persist_peer_store_cache_with_delivery_witnesses(
                &self.identity,
                &peer_store,
                &self.config.discovery,
                control_http_client,
                path,
                cache_save_at,
                true,
            )
            .await
            {
                Ok(PeerStoreCachePersistOutcome::Persisted) => {
                    debug!(
                        source = %path,
                        "[DISCOVERY] Initial peer cache snapshot persisted"
                    );
                }
                Ok(PeerStoreCachePersistOutcome::Deferred) => {
                    warn!(
                        source = %path,
                        "[DISCOVERY] Initial peer cache snapshot deferred pending delivery-witness protection"
                    );
                }
                Err(e) => warn!(
                    source = %path,
                    error = %e,
                    "[DISCOVERY] Failed to persist initial peer cache snapshot"
                ),
            }
        }
        Ok(peer_store)
    }

    pub(super) fn discovery_startup_self_check(config: &ServerConfig) -> (&'static str, String) {
        let discovery = &config.discovery;
        if !discovery.enabled {
            return ("skipped", "discovery_enabled=false".to_string());
        }

        let mut missing = Vec::new();
        if !discovery.advertise_self {
            missing.push("self_advertisement");
        }
        if discovery.peer_cache_path.is_none() {
            missing.push("peer_cache_path");
        }
        if !discovery.gossip_enabled {
            missing.push("gossip_enabled");
        }
        if discovery.gossip_enabled && discovery.seed_endpoints.is_empty() {
            missing.push("seed_endpoints");
        }
        let public_endpoint_configured =
            discovery.public_endpoint.is_some() || config.network.public_endpoint.is_some();
        if !public_endpoint_configured {
            missing.push("public_endpoint");
        }
        if discovery.public_api_listen_addr.is_none() {
            missing.push("public_api_listener");
        }
        if discovery.descriptor_ttl_secs < discovery.gossip_interval_secs.saturating_mul(2) {
            missing.push("descriptor_ttl_for_gossip");
        }

        if missing.is_empty() {
            (
                "ready",
                "cache,gossip,self_advertisement,public_endpoint,public_api_listener configured"
                    .to_string(),
            )
        } else {
            ("warning", format!("missing={}", missing.join(",")))
        }
    }
}
