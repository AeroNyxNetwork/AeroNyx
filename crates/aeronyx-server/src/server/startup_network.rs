// [ARCH-SPLIT 2026-10-02]
// Public address, peer store, and discovery self-check. Bootstrap file import lives next to this call.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

// [REVERSE-ONION-IDENTITY-SEED 2026-10-05 by Codex] Operator configuration
// pins stable identities; descriptor/KEM bytes are historical seed evidence.
// Return whether P's seed descriptor is still current enough to import. An
// expired seed never replaces a newer PeerStore descriptor or blocks gossip
// from refreshing the same pinned identity.
pub(super) fn reverse_onion_recipient_seed_is_current(
    local_relay: [u8; 32],
    relay: &SignedNodeDescriptor,
    recipient: &SignedNodeDescriptor,
    authorization: &SignedPrivateOnionRecipientAuthorizationV1,
    now: u64,
) -> Result<bool> {
    let issued_at = authorization.issued_at();
    if now == 0
        || local_relay == [0; 32]
        || relay.node_id() != local_relay
        || !relay.descriptor.public_endpoint.as_deref()
            .is_some_and(crate::api::reverse_onion_endpoint_supported)
        || recipient.node_id() == local_relay
        || recipient.node_id() != authorization.recipient_node_id()
        || authorization.relay_node_id() != local_relay
        || issued_at > now
        || relay.verify_at(issued_at).is_err()
        || recipient.verify_at(issued_at).is_err()
        || authorization
            .verify_at(
                relay,
                recipient,
                OnionRoutePurpose::BlindVaultPull.as_str(),
                issued_at,
            )
            .is_err()
    {
        return Err(ServerError::startup_failed(
            "reverse onion recipient identity pin rejected",
        ));
    }
    // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] Do not seed a signed
    // relay identity whose endpoint cannot authenticate durable poll receipts.
    Ok(recipient.verify_at(now).is_ok())
}

impl Server {
    // ============================================
    // Public IP Resolution
    // ============================================

    pub(super) async fn resolve_public_ip(&self) -> String {
        // [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] Configuration
        // rejects this management path for private recipients; keep a local
        // guard too, so future callers cannot trigger public-IP probes.
        if self.config.reverse_onion.recipient.enabled {
            return String::new();
        }
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

    // [REVERSE-ONION-AUTHORITY-RENEWAL 2026-10-05 by Codex] Current R/P
    // descriptor commitments belong to each enqueue authorization, not a
    // process-lifetime pin. Return None so normal startup/gossip builds the
    // current R descriptor and its ephemeral KEM key is never persisted.
    pub(super) fn pinned_private_experiment_self_descriptor_for(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
        _chat_relay_runtime_ready: bool,
        _blind_vault_runtime_ready: bool,
        _anonymous_mailbox_runtime_ready: bool,
    ) -> Result<Option<SignedNodeDescriptor>> {
        let queue = &config.reverse_onion.queue;
        if !queue.enabled || queue.recovery_only || !queue.signed_private_admission_configured() {
            return Ok(None);
        }
        let (relay_descriptor, recipient_descriptor, authorization) = queue
            .signed_private_authority_material()
            .map_err(|_| ServerError::startup_failed("reverse onion authority unavailable"))?;
        if now == 0
            || relay_descriptor.node_id() != identity.public_key_bytes()
            || recipient_descriptor.node_id() == identity.public_key_bytes()
            || authorization.issued_at() > now
            || authorization
                .verify_at(
                    &relay_descriptor,
                    &recipient_descriptor,
                    OnionRoutePurpose::BlindVaultPull.as_str(),
                    authorization.issued_at(),
                )
                .is_err()
        {
            return Err(ServerError::startup_failed(
                "reverse onion self descriptor authority rejected",
            ));
        }
        Ok(None)
    }

    fn pinned_private_experiment_self_descriptor(
        &self,
        _peer_store: &PeerStore,
        now: u64,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
    ) -> Result<Option<SignedNodeDescriptor>> {
        Self::pinned_private_experiment_self_descriptor_for(
            &self.config,
            &self.identity,
            now,
            chat_relay_runtime_ready,
            blind_vault_runtime_ready,
            anonymous_mailbox_runtime_ready,
        )
    }

    pub(super) async fn init_peer_store(
        &self,
        chat_relay_runtime_ready: bool,
        control_http_client: &reqwest::Client,
    ) -> Result<Arc<PeerStore>> {
        self.init_peer_store_with_storage_runtime(
            chat_relay_runtime_ready,
            self.config.blind_vault.replica_advertisement_configured(),
            false,
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

    // [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex] Install egress
    // identity pins before cache/gossip import, also in recovery-only mode.
    // A pin permits descriptor refresh/appraisal, never a new Claim or POST.
    fn pin_configured_reverse_onion_egress(
        peers: &PeerStore, config: &ServerConfig, identity: &IdentityKeyPair,
    ) -> Result<()> {
        let recipient = &config.reverse_onion.recipient;
        if recipient.enabled {
            let mut relay_id = [0u8; 32];
            if recipient.relay_node_id.len() != 64
                || hex::decode_to_slice(&recipient.relay_node_id, &mut relay_id).is_err()
                || relay_id == identity.public_key_bytes()
            {
                return Err(ServerError::startup_failed("reverse onion recipient relay identity pin rejected"));
            }
            peers.pin_private_onion_route_identities(relay_id, identity.public_key_bytes())
                .map_err(|_| ServerError::startup_failed("reverse onion recipient identity pair pin unavailable"))?;
        }
        let source = &config.reverse_onion.source;
        if source.enabled {
            let (relay, recipient, _) = source.identity_pins()
                .map_err(|_| ServerError::startup_failed("reverse onion source identity pins rejected"))?;
            if identity.public_key_bytes() == relay || identity.public_key_bytes() == recipient {
                return Err(ServerError::startup_failed("reverse onion source identity pins rejected"));
            }
            peers.pin_private_onion_route_identities(relay, recipient)
                .map_err(|_| ServerError::startup_failed("reverse onion source identity pair pin unavailable"))?;
        }
        Ok(())
    }

    /// Initializes discovery with explicitly observed storage readiness.
    ///
    /// The compatibility wrapper above remains for focused tests and embedded
    /// callers that do not own a running Blind Vault service.
    pub(super) async fn init_peer_store_with_storage_runtime(
        &self,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
        // [PRIVATE-ONION-PULL-READINESS 2026-10-05 by Codex] Keep read-only
        // terminal availability separate from public mutation admission.
        private_pull_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
        control_http_client: &reqwest::Client,
    ) -> Result<Arc<PeerStore>> {
        let peer_store = Self::new_discovery_peer_store();
        peer_store.set_max_peers(Some(self.config.discovery.max_peers));
        // [PHALA-PEER-ATTESTED-ROUTING 2026-10-06 by Codex] Install the
        // route gate before bootstrap/cache import can supply candidates.
        peer_store.configure_phala_attested_peer_routes(
            self.config.discovery.phala_attested_peers_required,
            self.config.discovery.phala_peer_attestation_max_age_secs,
        );
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
        // [PHALA-REVERSE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex] Pin the
        // configured recipient identity before importing cache/gossip. Signed
        // descriptors and P's grant may arrive later; the live queue fails
        // closed until the current authority snapshot exists.
        let queue = &self.config.reverse_onion.queue;
        if queue.enabled && !queue.recovery_only {
            // [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Discovery
            // installs the same full policy as queue startup, atomically and
            // before cache import; it never leaves a half-pinned source set.
            self.config.reverse_onion.validate()?;
            let pins = queue.live_identity_pins(self.identity.public_key_bytes())
                .map_err(|_| ServerError::startup_failed("reverse onion identity policy rejected"))?;
            let recipient_id = pins.recipient();
            peer_store
                .pin_private_onion_queue_identities(&pins)
                .map_err(|_| ServerError::startup_failed(
                    "reverse onion identity policy capacity rejected",
                ))?;
            if queue.signed_private_admission_configured() {
                let (relay, recipient, authorization) = queue
                    .signed_private_authority_material()
                    .map_err(|_| ServerError::startup_failed(
                        "reverse onion authority seed rejected",
                    ))?;
                if relay.node_id() != self.identity.public_key_bytes()
                    || recipient.node_id() != recipient_id
                {
                    return Err(ServerError::startup_failed(
                        "reverse onion authority seed identity mismatch",
                    ));
                }
                let seed_is_current = reverse_onion_recipient_seed_is_current(
                    self.identity.public_key_bytes(), &relay, &recipient, &authorization, now,
                )?;
                if seed_is_current {
                    // [REVERSE-ONION-STALE-SEED 2026-10-05 by Codex] Preserve
                    // newer cache state; gossip refreshes the live authority.
                    if peer_store
                        .seed_private_onion_route_descriptor(
                            &self.identity.public_key_bytes(), &recipient_id, recipient, now,
                            "reverse_onion_identity_seed",
                        )
                        .is_err()
                    {
                        // A newer cached descriptor or same-sequence conflict
                        // must win over this optional historical seed. Gossip is
                        // already scheduled later in startup and remains the
                        // refresh path; live admission stays fail-closed meanwhile.
                        debug!("[DISCOVERY] Reverse onion identity seed deferred to gossip");
                    }
                }
            }
        }
        Self::pin_configured_reverse_onion_egress(&peer_store, &self.config, &self.identity)?;
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

        // [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] Keep a second
        // runtime gate here for embedders that construct config without the
        // normal ServerConfig validation path. Local snapshot/cache imports
        // remain available; remote bootstrap is relay-only for this role.
        if self.config.reverse_onion.recipient.enabled
            && self.config.discovery.bootstrap_snapshot_url.is_some()
        {
            peer_store.record_bootstrap_source(
                now,
                "url",
                "skipped",
                "private_recipient_relay_only",
            );
        } else if let Some(url) = &self.config.discovery.bootstrap_snapshot_url {
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
            // [PHALA-SELF-DESCRIPTOR-SEQUENCE 2026-10-08 by Codex] Cache
            // and bootstrap awaits must not reuse the pre-import issue time.
            let now = if self.config.reverse_onion.queue.enabled
                || self.config.reverse_onion.recipient.enabled
                || self.config.reverse_onion.source.enabled
            {
                let observed = unix_now_secs();
                if observed < now {
                    return Err(ServerError::startup_failed("private self descriptor clock rejected"));
                }
                observed
            } else { now };
            // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Startup must
            // not sign a replacement descriptor after a rejected epoch.
            crate::services::onion_keys::try_tick_rotation(now)
                .map_err(|_| ServerError::startup_failed("onion key epoch rejected"))?;
            // [PHALA-SELF-DESCRIPTOR-SEQUENCE 2026-10-08 by Codex] A warm
            // restart takes only the authenticated counter from the peer cache.
            let self_sequence = Self::private_authority_self_descriptor_sequence(
                &self.config, &self.identity, None, &peer_store, now,
            )?;
            let descriptor = self.pinned_private_experiment_self_descriptor(
                &peer_store,
                now,
                chat_relay_runtime_ready,
                blind_vault_runtime_ready,
                anonymous_mailbox_runtime_ready,
            )?;
            let descriptor = match descriptor {
                Some(descriptor) => Ok(descriptor),
                None => Self::build_self_discovery_descriptor_for_runtime_state_with_private_pull_and_sequence(
                    &self.config,
                    &self.identity,
                    now,
                    chat_relay_runtime_ready,
                    blind_vault_runtime_ready,
                    anonymous_mailbox_runtime_ready,
                    private_pull_runtime_ready,
                    self_sequence,
                ),
            };
            match descriptor {
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
        // [PHALA-RECIPIENT-DESCRIPTOR-RECOVERY 2026-10-06 by Codex]
        // Recovery-only also needs descriptor refresh to resend exact durable
        // frames; this bootstrap never grants authority to create new Claims.
        let private_recipient_bootstrap = config.reverse_onion.recipient.enabled
            && !config.reverse_onion.recipient.relay_node_id.is_empty()
            && !config.reverse_onion.recipient.relay_endpoint.is_empty();
        // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Do not label a
        // directly constructed private config pinned-relay-only while it still
        // carries general seeds. Parent validation rejects the same mismatch.
        if config.reverse_onion.recipient.enabled && !discovery.seed_endpoints.is_empty() {
            missing.push("private_recipient_seed_endpoints");
        }
        if discovery.gossip_enabled
            && discovery.seed_endpoints.is_empty()
            && !private_recipient_bootstrap
        {
            missing.push("seed_endpoints");
        }
        let public_endpoint_configured =
            discovery.public_endpoint.is_some() || config.network.public_endpoint.is_some();
        // [PHALA-PRIVATE-RECIPIENT-PROFILE 2026-10-06 by Codex] A private
        // recipient bootstraps through its pinned relay and intentionally has
        // no signed public endpoint. Do not report that supported role as an
        // incomplete public-peer deployment.
        let private_recipient_mode = private_recipient_bootstrap
            && !public_endpoint_configured
            && discovery.public_api_listen_addr.is_none();
        if !public_endpoint_configured && !private_recipient_mode {
            missing.push("public_endpoint");
        }
        if discovery.public_api_listen_addr.is_none() && !private_recipient_mode {
            missing.push("public_api_listener");
        }
        if discovery.descriptor_ttl_secs < discovery.gossip_interval_secs.saturating_mul(2) {
            missing.push("descriptor_ttl_for_gossip");
        }

        if missing.is_empty() {
            let endpoint_mode = if private_recipient_mode {
                "private_recipient_endpoint_free"
            } else {
                "public_endpoint"
            };
            let listener_mode = if private_recipient_mode {
                "outbound_only"
            } else {
                "public_api_listener"
            };
            let egress_mode = if private_recipient_mode {
                ",pinned_relay_only"
            } else {
                ""
            };
            (
                "ready",
                format!(
                    "cache,gossip,self_advertisement,{endpoint_mode},{listener_mode}{egress_mode},bootstrap configured"
                ),
            )
        } else {
            ("warning", format!("missing={}", missing.join(",")))
        }
    }
}

#[cfg(test)]
mod phala_recipient_bootstrap_tests {
    use super::*;

    // [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex] Authored only:
    // bootstrap pins are installed in both modes without enabling fresh work.
    #[test]
    fn private_egress_pins_exist_before_import_in_live_and_recovery_modes() {
        let identity = IdentityKeyPair::from_bytes(&[171; 32]).unwrap();
        for recovery_only in [false, true] {
            let mut config = ServerConfig::default();
            config.reverse_onion.recipient.enabled = true;
            config.reverse_onion.recipient.recovery_only = recovery_only;
            config.reverse_onion.recipient.relay_node_id = "11".repeat(32);
            let peers = PeerStore::new();
            Server::pin_configured_reverse_onion_egress(&peers, &config, &identity).unwrap();
            assert!(peers.has_private_onion_route_identity_pin(&[0x11; 32], &identity.public_key_bytes()));
            assert_eq!(config.reverse_onion.recipient.permits_new_claims(), !recovery_only);
            config.reverse_onion.recipient.relay_node_id = hex::encode(identity.public_key_bytes());
            assert!(Server::pin_configured_reverse_onion_egress(&PeerStore::new(), &config, &identity).is_err());

            config.reverse_onion.recipient.enabled = false;
            config.reverse_onion.source.enabled = true;
            config.reverse_onion.source.recovery_only = recovery_only;
            // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex]
            // Pins are valid Ed25519 public points, not repeated raw bytes.
            let source_relay = IdentityKeyPair::from_bytes(&[172; 32]).unwrap().public_key_bytes();
            let source_recipient = IdentityKeyPair::from_bytes(&[173; 32]).unwrap().public_key_bytes();
            config.reverse_onion.source.relay_node_id = hex::encode(source_relay);
            config.reverse_onion.source.recipient_node_id = hex::encode(source_recipient);
            // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex]
            // Reserved .example authorities must stay rejected in production.
            config.reverse_onion.source.relay_endpoint = "https://relay.aeronyx.network".into();
            let peers = PeerStore::new();
            Server::pin_configured_reverse_onion_egress(&peers, &config, &identity).unwrap();
            assert!(peers.has_private_onion_route_identity_pin(&source_relay, &source_recipient));
            // [PHALA-APPRAISAL-EGRESS-PIN 2026-10-07 by Codex] Explicit
            // ordinary scope still cannot turn bare identity pins into peers.
            assert!(peers.next_phala_peer_appraisal_target(1, 64, 0,
                &crate::services::peer_store::PhalaPeerAppraisalEgress::discovery()).is_none(),
                "identity pins do not create descriptors or appraisal evidence");
            assert_eq!(config.reverse_onion.source.recovery_only, recovery_only);
        }
    }

    fn configured_discovery() -> ServerConfig {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;
        config.discovery.peer_cache_path = Some("/var/lib/aeronyx/peers-cache.json".into());
        config.discovery.public_endpoint = Some("https://node.example".into());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config
    }

    // [PHALA-RECIPIENT-GOSSIP-BOOTSTRAP 2026-10-06 by Codex]
    #[test]
    fn pinned_recipient_replaces_general_seeds_for_live_and_recovery_modes() {
        let mut config = configured_discovery();
        assert_eq!(Server::discovery_startup_self_check(&config).0, "warning");

        config.reverse_onion.recipient.enabled = true;
        config.reverse_onion.recipient.relay_node_id = "11".repeat(32);
        config.reverse_onion.recipient.relay_endpoint = "https://relay.example".into();
        assert_eq!(Server::discovery_startup_self_check(&config).0, "ready");

        config.reverse_onion.recipient.recovery_only = true;
        assert_eq!(Server::discovery_startup_self_check(&config).0, "ready");
        config.reverse_onion.recipient.recovery_only = false;
        // [PHALA-PRIVATE-RECIPIENT-PROFILE 2026-10-06 by Codex] The separate
        // private identity is ready without publishing its own descriptor URL.
        config.discovery.public_endpoint = None;
        config.discovery.public_api_listen_addr = None;
        let (status, detail) = Server::discovery_startup_self_check(&config);
        assert_eq!(status, "ready");
        assert!(detail.contains("private_recipient_endpoint_free"));
        assert!(detail.contains("outbound_only"));
        assert!(detail.contains("pinned_relay_only"));
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        assert_eq!(Server::discovery_startup_self_check(&config).0, "warning");
        config.reverse_onion.recipient.relay_endpoint.clear();
        assert_eq!(Server::discovery_startup_self_check(&config).0, "warning");
    }

    // [PHALA-ROLE-SEED-ISOLATION 2026-10-07 by Codex] Authored only: a
    // stale mounted seed list must not produce misleading private readiness.
    #[test]
    fn private_recipient_self_check_requires_cleared_general_seeds() {
        let mut config = configured_discovery();
        config.reverse_onion.recipient.enabled = true;
        config.reverse_onion.recipient.relay_node_id = "11".repeat(32);
        config.reverse_onion.recipient.relay_endpoint = "https://relay.aeronyx.network".into();
        config.discovery.public_endpoint = None;
        config.discovery.public_api_listen_addr = None;
        config.discovery.public_discovery = false;
        for recovery_only in [false, true] {
            config.reverse_onion.recipient.recovery_only = recovery_only;
            config.discovery.seed_endpoints = vec!["https://seed.aeronyx.network".into()];
            let (status, detail) = Server::discovery_startup_self_check(&config);
            assert_eq!(status, "warning");
            assert!(detail.contains("private_recipient_seed_endpoints"));
            assert!(!detail.contains("pinned_relay_only"));
            assert_eq!(config.discovery.seed_endpoints.len(), 1);

            config.discovery.seed_endpoints.clear();
            let (status, detail) = Server::discovery_startup_self_check(&config);
            assert_eq!(status, "ready");
            assert!(detail.contains("pinned_relay_only"));
            assert_eq!(config.reverse_onion.recipient.relay_endpoint, "https://relay.aeronyx.network");
            assert_eq!(config.reverse_onion.recipient.recovery_only, recovery_only);
        }
    }
}
