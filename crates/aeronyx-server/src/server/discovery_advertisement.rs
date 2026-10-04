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
        let ttl = config.discovery.descriptor_ttl_secs;
        let expires_at = now.saturating_add(ttl);
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
        if anonymous_mailbox_runtime_ready {
            // [BLIND-RELAY-ANONYMOUS-MAILBOX 2026-09-03 by Codex] Sign this
            // token only after the default-off store opened successfully and
            // was injected into both peer routers.
            protocol_features.push(NodeProtocolFeature::AnonymousMailboxV1);
        }
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            now,
            now,
            expires_at,
            env!("CARGO_PKG_VERSION"),
        )
        .with_protocol_features(protocol_features);
        descriptor.public_endpoint = config
            .discovery
            .public_endpoint
            .as_ref()
            .or(config.network.public_endpoint.as_ref())
            .map(|endpoint| endpoint.trim().to_string())
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
        descriptor = descriptor.with_x25519_kem(crate::services::onion_keys::current_public_key());

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
        if config.discovery.advertise_onion_middle && advertises_peer_api {
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
