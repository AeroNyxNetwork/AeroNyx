// [ARCH-SPLIT 2026-10-02]
// Self descriptor and local capability advertisement after runtime state exists.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

// [BLIND-VAULT-REPLICA-ADMISSION 2026-10-04 by Codex] Admission readiness
// only proves that durable jobs can be accepted locally. Until a supervised
// outbound dispatcher is wired, do not advertise end-to-end replication.
const BLIND_VAULT_REPLICA_DISPATCHER_READY: bool = false;

impl Server {
    pub(super) fn build_self_discovery_descriptor(&self, now: u64) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for(&self.config, &self.identity, now)
    }

    pub(super) fn build_self_discovery_descriptor_with_runtime(
        &self,
        now: u64,
        chat_relay_runtime_ready: bool,
    ) -> Result<SignedNodeDescriptor> {
        self.build_self_discovery_descriptor_with_runtime_state(
            now,
            chat_relay_runtime_ready,
            self.config.blind_vault.replica_advertisement_configured(),
            false,
        )
    }

    pub(super) fn build_self_discovery_descriptor_with_runtime_state(
        &self,
        now: u64,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
    ) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for_runtime_state(
            &self.config,
            &self.identity,
            now,
            chat_relay_runtime_ready,
            blind_vault_runtime_ready,
            anonymous_mailbox_runtime_ready,
        )
    }

    pub(super) fn build_self_discovery_descriptor_for(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
    ) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for_runtime(
            config,
            identity,
            now,
            config.memchain.is_chat_relay_enabled(),
            false,
        )
    }

    pub(super) fn build_self_discovery_descriptor_for_runtime(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
        chat_relay_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
    ) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for_runtime_state(
            config,
            identity,
            now,
            chat_relay_runtime_ready,
            config.blind_vault.replica_advertisement_configured(),
            anonymous_mailbox_runtime_ready,
        )
    }

    pub(super) fn build_self_discovery_descriptor_for_runtime_state(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
    ) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for_runtime_state_with_private_pull(
            config,
            identity,
            now,
            chat_relay_runtime_ready,
            blind_vault_runtime_ready,
            anonymous_mailbox_runtime_ready,
            false,
        )
    }

    pub(super) fn build_self_discovery_descriptor_for_runtime_state_with_private_pull(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
        private_pull_runtime_ready: bool,
    ) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for_runtime_state_with_private_pull_and_sequence(
            config, identity, now, chat_relay_runtime_ready, blind_vault_runtime_ready,
            anonymous_mailbox_runtime_ready, private_pull_runtime_ready, now,
        )
    }

    // [PHALA-SELF-DESCRIPTOR-SEQUENCE 2026-10-08 by Codex] The local
    // sequence is chosen before any nested policy or descriptor is signed.
    pub(super) fn build_self_discovery_descriptor_for_runtime_state_with_private_pull_and_sequence(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
        anonymous_mailbox_runtime_ready: bool,
        private_pull_runtime_ready: bool,
        sequence: u64,
    ) -> Result<SignedNodeDescriptor> {
        let ttl = config.discovery.descriptor_ttl_secs;
        // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Embedded callers
        // also pass the KEM lifetime gate; never sign a saturated expiry.
        if !crate::services::onion_keys::descriptor_ttl_is_supported(ttl) {
            return Err(ServerError::config_invalid(
                "discovery.descriptor_ttl_secs",
                "unsupported onion key overlap lifetime",
            ));
        }
        if now == 0 {
            return Err(ServerError::startup_failed("self descriptor clock rejected"));
        }
        let expires_at = now.checked_add(ttl)
            .ok_or_else(|| ServerError::startup_failed("self descriptor expiry rejected"))?;
        // [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] The exact feature
        // token is covered by the existing descriptor signature and remains a
        // valid opaque version string to pre-feature decoders.
        let mut protocol_features = vec![
            NodeProtocolFeature::BlindRelayFailureReceiptV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::PurposeBoundDeliveryReceiptV2,
            NodeProtocolFeature::DirectPeerRelayAuthV2,
            NodeProtocolFeature::DirectPeerRelayReceiptV2,
            NodeProtocolFeature::DirectPeerRelayTargetBindingV3,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            NodeProtocolFeature::OnionBlindVaultLargePullV1,
            NodeProtocolFeature::OnionBlindLeaseAdmissionV1,
            NodeProtocolFeature::OnionBlindVaultPutReceiptV1,
            NodeProtocolFeature::OnionBlindVaultLeaseRetireV1,
            NodeProtocolFeature::OnionBlindVaultLeaseRenewalV1,
            NodeProtocolFeature::OnionBlindVaultLeaseStatusV1,
            NodeProtocolFeature::OnionBlindVaultLeaseInventoryV1,
            NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
        ];
        // [PRIVATE-ONION-PULL-ROLE 2026-10-05 by Codex] Keep the operation
        // terminal feature distinct from the path-wide large-reply feature and
        // from public replica mutation admission.
        // [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex] An
        // operator recovery hold withdraws fresh-poll support consistently.
        if config.blind_vault.enabled
            && ((config.blind_vault.public_api_enabled && blind_vault_runtime_ready)
                || (config.reverse_onion.recipient.permits_new_claims()
                    && config.discovery.enabled
                    && config.discovery.gossip_enabled
                    && config.memchain.is_chat_relay_enabled()
                    && chat_relay_runtime_ready
                    && private_pull_runtime_ready))
        {
            protocol_features.push(NodeProtocolFeature::PrivateOnionBlindVaultPullTerminalV1);
        }
        // [PRIVATE-ONION-ADMISSION-ROLE 2026-10-05 by Codex] Lease admission
        // shares only the already-running private terminal + Blind Vault
        // service. It does not enable public replica admission or a listener.
        if config.blind_vault.enabled
            && !config.blind_vault.public_api_enabled
            && blind_vault_runtime_ready
            && config.reverse_onion.recipient.permits_new_claims()
            && config.discovery.enabled
            && config.discovery.gossip_enabled
            && config.memchain.is_chat_relay_enabled()
            && chat_relay_runtime_ready
            && private_pull_runtime_ready
        {
            protocol_features
                .push(NodeProtocolFeature::PrivateOnionBlindVaultAdmissionTerminalV1);
        }
        // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex] Advertise the
        // append-only grant carrier only when this node has live queue
        // admission. Recovery-only nodes must not invite new authority.
        if config.reverse_onion.requires_live_authority_gossip() {
            protocol_features.push(NodeProtocolFeature::PrivateOnionAuthorizationGossipV1);
        }
        // [PHALA-NODE-ATTESTATION-API 2026-10-06 by Codex] Advertise only
        // when the local dstack socket is configured and config validation
        // has established that public discovery is enabled. This is endpoint
        // support, not evidence that a quote has been independently verified.
        if config.discovery.phala_attestation_socket_path.is_some()
            && config.discovery.enabled
            && config.discovery.advertise_self
            && config.discovery.public_discovery
            && config.discovery.public_api_listen_addr.is_some()
            && config
                .effective_public_endpoint()
                .is_some_and(crate::api::reverse_onion_endpoint_supported)
        {
            protocol_features.push(NodeProtocolFeature::PhalaNodeAttestationV1);
        }
        // [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] This
        // signed hint is limited to one explicitly pinned queue recipient.
        if protocol_features.contains(&NodeProtocolFeature::PhalaNodeAttestationV1)
            && config.reverse_onion.queue.permits_new_claims()
            && config.reverse_onion.queue.recipient_node_ids.len() == 1
        {
            protocol_features.push(NodeProtocolFeature::PhalaPrivateRecipientAttestationV1);
        }
        if anonymous_mailbox_runtime_ready {
            // [BLIND-RELAY-ANONYMOUS-MAILBOX 2026-09-03 by Codex] Sign this
            // token only after the default-off store opened successfully and
            // was injected into both peer routers.
            protocol_features.push(NodeProtocolFeature::AnonymousMailboxV1);
        }
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            sequence,
            now,
            expires_at,
            env!("CARGO_PKG_VERSION"),
        )
        .with_protocol_features(protocol_features);
        descriptor.public_endpoint = config
            .effective_public_endpoint()
            .map(str::trim)
            .map(str::to_string)
            .filter(|endpoint| !endpoint.is_empty());

        descriptor.capabilities = Self::discovery_capabilities_for_runtime_state(
            config,
            chat_relay_runtime_ready,
            blind_vault_runtime_ready,
        );
        descriptor.capacity = NodeCapacity {
            max_sessions: u32::try_from(config.max_sessions()).unwrap_or(u32::MAX),
            max_bps: None,
            max_pps: None,
        };
        descriptor.policy = NodePolicy {
            allows_public_exit: false,
            public_discovery: config.discovery.public_discovery,
            region: config
                .discovery
                .region
                .as_ref()
                .map(|region| region.trim().to_string())
                .filter(|region| !region.is_empty()),
        };

        // Onion routing: publish the node's CURRENT rotating onion KEM public
        // key (forward secrecy — see services::onion_keys). NOT identity-derived:
        // the X25519 public is not recoverable from the Ed25519 node_id, and a
        // rotating key bounds exposure of past routing metadata if a key leaks.
        // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Capture one public
        // key observation; a concurrent later epoch cannot sign old local time.
        let (kem_public, _) = crate::services::onion_keys::advertised_public_keys(now);
        if kem_public == [0; 32] {
            return Err(ServerError::startup_failed("self descriptor onion key rejected"));
        }
        descriptor = descriptor.with_x25519_kem(kem_public);

        if anonymous_mailbox_runtime_ready {
            // [ANONYMOUS-MAILBOX-WORK-POLICY 2026-09-07 by Codex] The same
            // validated configuration that opened the target store is signed
            // only after that runtime exists. Failure is coarse and prevents a
            // ready feature token without its enforceable work policy.
            descriptor = descriptor
                .with_anonymous_mailbox_work_policy(
                    config
                        .memchain
                        .chat_relay
                        .anonymous_mailbox
                        .ticket_issue_work_bits,
                    identity,
                )
                .map_err(|_| {
                    ServerError::startup_failed(
                        "anonymous mailbox work-policy advertisement failed",
                    )
                })?;
        }

        SignedNodeDescriptor::sign(descriptor, identity).map_err(ServerError::from)
    }

    // [PHALA-AUTHORITY-DESCRIPTOR-EPOCH 2026-10-08 by Codex] Exact R/P
    // commitments must survive ordinary gossip heartbeats long enough for P
    // to issue and R to forward a grant. Only the running gossip owner supplies
    // `previous`; never restore this generation from a persisted peer cache.
    pub(super) fn select_private_authority_descriptor_epoch(
        config: &ServerConfig,
        candidate: SignedNodeDescriptor,
        previous: Option<&SignedNodeDescriptor>,
        now: u64,
    ) -> SignedNodeDescriptor {
        if !config.reverse_onion.requires_live_authority_gossip() {
            return candidate;
        }
        let Some(previous) = previous else { return candidate; };
        if now < previous.descriptor.issued_at
            || previous.verify_at(now).is_err()
            || candidate.verify_at(now).is_err()
        {
            return candidate;
        }
        let lifetime = previous.descriptor.expires_at
            .saturating_sub(previous.descriptor.issued_at);
        let candidate_lifetime = candidate.descriptor.expires_at
            .saturating_sub(candidate.descriptor.issued_at);
        if lifetime != candidate_lifetime
            || now.saturating_sub(previous.descriptor.issued_at) >= (lifetime / 2).max(1)
        {
            return candidate;
        }
        // Compare every signed field, ignoring only this heartbeat's temporal
        // renewal. KEM, endpoint, feature/readiness, capacity and policy changes
        // immediately supersede the epoch, including capability withdrawals.
        let mut comparable = candidate.descriptor.clone();
        // [PHALA-POLICY-HEARTBEAT-EPOCH 2026-10-08 by Codex] This nested
        // policy signs sequence/TTL too. Normalize only its authenticated token
        // in this local comparison; never publish the reconstructed metadata.
        use aeronyx_core::protocol::discovery::AnonymousMailboxWorkPolicyError;
        match (
            candidate.anonymous_mailbox_work_policy_at(now),
            previous.anonymous_mailbox_work_policy_at(now),
        ) {
            (Ok(current_policy), Ok(previous_policy)) => {
                if current_policy.work_bits() != previous_policy.work_bits()
                    || current_policy.target_node_id() != previous_policy.target_node_id()
                {
                    return candidate;
                }
                let current_token = current_policy.semver_build_token();
                let previous_token = previous_policy.semver_build_token();
                let Some((release, metadata)) = comparable.software_version.split_once('+')
                    else { return candidate; };
                let metadata = metadata.split('.').map(|token|
                    if token == current_token.as_str() { previous_token.as_str() } else { token }
                ).collect::<Vec<_>>().join(".");
                comparable.software_version = format!("{release}+{metadata}");
            }
            (Err(AnonymousMailboxWorkPolicyError::MissingPolicy),
                Err(AnonymousMailboxWorkPolicyError::MissingPolicy))
                if !candidate.descriptor.advertises_protocol_feature(NodeProtocolFeature::AnonymousMailboxV1)
                    && !previous.descriptor.advertises_protocol_feature(NodeProtocolFeature::AnonymousMailboxV1) => {}
            _ => return candidate,
        }
        comparable.sequence = previous.descriptor.sequence;
        comparable.issued_at = previous.descriptor.issued_at;
        comparable.expires_at = previous.descriptor.expires_at;
        if comparable == previous.descriptor {
            previous.clone()
        } else {
            candidate
        }
    }

    // [PHALA-SELF-DESCRIPTOR-SEQUENCE 2026-10-08 by Codex] Read only
    // authenticated local counters. Allocate before building sequence-bound
    // signed policies; never rewrite the sequence of an already signed surface.
    pub(super) fn private_authority_self_descriptor_sequence(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        previous: Option<&SignedNodeDescriptor>,
        peer_store: &PeerStore,
        now: u64,
    ) -> Result<u64> {
        if !config.reverse_onion.queue.enabled
            && !config.reverse_onion.recipient.enabled
            && !config.reverse_onion.source.enabled
        {
            return Ok(now);
        }
        let local = identity.public_key_bytes();
        let rejected = || ServerError::startup_failed("private self descriptor sequence rejected");
        if now == 0 { return Err(rejected()); }
        let cached = peer_store.get_signature_verified_cached(&local);
        let mut sequence = now;
        // Expiry removes authority, not the retained anti-reuse counter.
        // Rollback below an authentic issue time is not forward renewal.
        for descriptor in cached.as_ref().into_iter().chain(previous) {
            if descriptor.node_id() != local || descriptor.verify_signature().is_err()
                || now < descriptor.descriptor.issued_at
            {
                return Err(rejected());
            }
            sequence = sequence.max(descriptor.sequence().checked_add(1).ok_or_else(rejected)?);
        }
        Ok(sequence)
    }

    pub(super) fn discovery_capabilities_for_runtime(
        config: &ServerConfig,
        chat_relay_runtime_ready: bool,
    ) -> Vec<NodeCapability> {
        Self::discovery_capabilities_for_runtime_state(
            config,
            chat_relay_runtime_ready,
            config.blind_vault.replica_advertisement_configured(),
        )
    }

    pub(super) fn discovery_capabilities_for_runtime_state(
        config: &ServerConfig,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
    ) -> Vec<NodeCapability> {
        let mut capabilities = vec![NodeCapability::PrivacyRelay];
        let advertises_peer_api = Self::discovery_peer_api_ready_for(config);

        if config.memchain.is_chat_relay_enabled()
            && chat_relay_runtime_ready
            && advertises_peer_api
        {
            capabilities.push(NodeCapability::ChatRelay);
        }
        // [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex] OnionMiddle is a
        // routeability promise: the durable ChatRelay handler must be live,
        // not merely an operator flag plus a reachable HTTP listener.
        if config.discovery.advertise_onion_middle
            && config.memchain.is_chat_relay_enabled()
            && chat_relay_runtime_ready
            && advertises_peer_api
        {
            capabilities.push(NodeCapability::OnionMiddle);
        }
        // [MIRROR-CAPABILITY 2026-07-24 by Codex] This capability is an
        // operator-controlled staged rollout because pre-upgrade binaries
        // cannot decode the appended enum variant. Re-check every runtime
        // prerequisite here even though parsed configuration validates them,
        // since tests and embedders may construct ServerConfig directly.
        if config.discovery.advertise_directory_mirror_carrier
            && config.discovery.directory_full_node_mirror_enabled
            && config.discovery.directory_chain_path.is_some()
            && config.discovery.public_discovery
            && advertises_peer_api
        {
            capabilities.push(NodeCapability::DirectoryMirrorCarrier);
        }
        // [BLIND-VAULT-REPLICA-CAPABILITY 2026-08-10 by Codex] This appended
        // wire variant remains operator-gated during mixed-version rollout.
        // Re-check the full runtime transport surface so a node never signs a
        // capability that its public peer listener cannot actually serve.
        if config.blind_vault.replica_advertisement_configured()
            && config.memchain.is_chat_relay_enabled()
            && chat_relay_runtime_ready
            && blind_vault_runtime_ready
            && advertises_peer_api
            && BLIND_VAULT_REPLICA_DISPATCHER_READY
        {
            capabilities.push(NodeCapability::BlindVaultReplica);
        }
        if config.memchain.is_enabled() {
            capabilities.push(NodeCapability::EncryptedStorage);
        }
        if config.memchain.is_supernode_enabled() {
            capabilities.push(NodeCapability::AgentRelay);
        }

        capabilities
    }

    pub(super) fn discovery_peer_api_ready_for(config: &ServerConfig) -> bool {
        config.discovery.public_api_listen_addr.is_some()
            && config
                .discovery
                .public_endpoint
                .as_deref()
                .map(str::trim)
                .map(|endpoint| !endpoint.is_empty())
                .unwrap_or(false)
    }

    pub(super) fn discovery_local_capability_status_for(
        config: &ServerConfig,
    ) -> DiscoveryLocalCapabilityStatus {
        Self::discovery_local_capability_status_for_runtime(
            config,
            config.memchain.is_chat_relay_enabled(),
        )
    }

    pub(super) fn discovery_local_capability_status_for_runtime(
        config: &ServerConfig,
        chat_relay_runtime_ready: bool,
    ) -> DiscoveryLocalCapabilityStatus {
        Self::discovery_local_capability_status_for_runtime_state(
            config,
            chat_relay_runtime_ready,
            config.blind_vault.replica_advertisement_configured(),
        )
    }

    pub(super) fn discovery_local_capability_status_for_runtime_state(
        config: &ServerConfig,
        chat_relay_runtime_ready: bool,
        blind_vault_runtime_ready: bool,
    ) -> DiscoveryLocalCapabilityStatus {
        let capabilities = Self::discovery_capabilities_for_runtime_state(
            config,
            chat_relay_runtime_ready,
            blind_vault_runtime_ready,
        );
        DiscoveryLocalCapabilityStatus::new_with_blind_vault(
            config.memchain.is_chat_relay_enabled(),
            Self::discovery_peer_api_ready_for(config),
            chat_relay_runtime_ready,
            capabilities.contains(&NodeCapability::ChatRelay),
            DiscoveryBlindVaultCapabilityObservation {
                configured: config.blind_vault.replica_advertisement_configured(),
                runtime_ready: blind_vault_runtime_ready,
                advertised: capabilities.contains(&NodeCapability::BlindVaultReplica),
            },
        )
    }
}
