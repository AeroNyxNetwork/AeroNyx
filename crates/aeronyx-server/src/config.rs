// ============================================
// File: crates/aeronyx-server/src/config.rs
// ============================================
//! # Server Configuration — Entry Layer
//!
//! ## Creation Reason
//! Central configuration entry point for all AeroNyx server subsystems.
//! Loaded from a TOML file at startup.
//!
//! ## Modification Reason
//! v1.1.0-ChatRelay — 🌟 Refactored into multi-file layout:
//!   - config_infra.rs    — NetworkConfig, VpnConfig, TunConfig,
//!                          ServerKeyConfig, LimitsConfig, LoggingConfig
//!   - config_saas.rs     — SaasConfig
//!   - config_chat_relay.rs — ChatRelayConfig
//!   - config_memchain.rs — MemChainConfig, MemChainMode, VectorQuantizationMode
//!   - config_supernode.rs — SuperNodeConfig (pre-existing)
//!   This file is now a thin composition + re-export layer only.
//!   All logic lives in the sub-modules above.
//!
//! ## Main Functionality
//! - `ServerConfig` — top-level struct; owns all sub-configs
//! - `ServerConfig::load(path)` — async TOML load + validate
//! - `ServerConfig::from_str(s)` — sync TOML parse + validate (tests)
//! - `ServerConfig::validate()` — delegates to every sub-config
//! - Re-exports all public types from sub-modules for downstream crates
//!   that `use crate::config::*`
//!
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `config.rs`; bodies unchanged.
//! This root keeps the composition layer: `ServerConfig` (load/parse,
//! cross-section validation, accessors, `Default`), the crate-internal
//! `PinnedRouteDomainAssignment`, and the re-exports. `ServerConfig` stays
//! here so `load()` keeps logging under the `aeronyx_server::config` tracing
//! target. `DiscoveryConfig` is re-exported from its child module, so every
//! existing `crate::config::*` and `aeronyx_server::config::*` path is
//! unchanged.
//! - `config/discovery.rs`: `DiscoveryConfig` fields and serde defaults,
//!   default-value constructors, decoded pin accessors, and `Default`
//! - `config/discovery/validation.rs`: discovery bounds, the shared node-id
//!   pin-set validator, `DiscoveryConfig::validate`, the runtime identity
//!   check, and seed endpoint syntax
//! - Tests sit beside the code they exercise: `config/discovery/tests.rs` and
//!   `config/discovery/validation/tests.rs`. Full-stack `ServerConfig` tests
//!   stay inline below; the shipped-example test's `include_str!` path is
//!   relative to this file.
//! - The earlier `config_*.rs` extractions are still declared at crate root
//!   (`lib.rs`) and re-exported here unchanged.
//!
//! ## Dependencies
//! - config_infra.rs    — infrastructure configs (no MemChain awareness)
//! - config_memchain.rs — MemChain + all nested subsystem configs
//! - config_supernode.rs — SuperNodeConfig (re-exported via config_memchain)
//! - config_saas.rs     — SaasConfig (re-exported via config_memchain)
//! - config_chat_relay.rs — ChatRelayConfig (re-exported via config_memchain)
//! - config_blind_vault.rs — BlindVaultConfig (independent node-blind store)
//! - management.rs      — ManagementConfig (owned by that subsystem)
//! - server.rs          — consumes ServerConfig for full initialization
//!
//! ## Main Logical Flow
//! 1. `load(path)` reads TOML file → `toml::from_str` → `ServerConfig`
//! 2. `validate()` calls each sub-config's validate in order:
//!    network → vpn → tun → limits → management → memchain
//! 3. Validated config returned to server.rs for subsystem initialization
//!
//! ⚠️ Important Note for Next Developer:
//! - Do NOT add business logic to this file. It is an orchestration layer only.
//! - All serde defaults on sub-structs ensure any missing TOML section is
//!   backward-compatible (defaults to disabled / safe values).
//! - Adding a new subsystem config: create config_<name>.rs, add a field
//!   to MemChainConfig (or ServerConfig if infra-level), add validate()
//!   delegation, and add a `pub use` here.
//! - Integration tests that span multiple sub-configs belong in this file's
//!   #[cfg(test)] block; unit tests belong in each sub-module's own tests.
//!
//! ## Last Modified
//! v0.27.0-ArchSplit - Split into focused child modules under `config/`;
//! bodies unchanged.
//! v0.26.0-CustodyWitnessAutoRenewal - Added an explicit default-off runtime
//! renewal policy that requires both strict startup and runtime custody gates.
//! v0.25.0-CustodyWitnessRuntimeGuard - Added a default-off, local-only
//! runtime re-audit gate that requires the strict startup gate.
//! v0.24.0-CustodyWitnessStartupGate - Added a default-off, local-only
//! current-anchor receipt threshold and bounded freshness policy for startup
//! v0.23.0-CustodyWitnessReceiptVault - Clarified that explicit durable rounds
//! use producer pins while configuration alone never schedules transmission
//! v0.22.0-CustodyWitnessPlanner - Added independent producer witness pins and
//! a validated local quorum target without enabling outbound transmission
//! v0.21.0-CustodyWitnessAdmission - Added an independent fail-closed producer
//! pin set for custody-audit witness writes
//! v0.20.0-RouteDomainAttestors - Added opt-in pinned attestor quorum and
//! fail-closed certificate requirement for multi-hop route-domain assignments
//! v0.19.0-PinnedRouteDomains - Added optional fail-closed, operator-audited
//! opaque route-domain assignments for multi-hop anti-affinity
//! v0.18.0-DirectoryProofMaturity - Added an automatically safe, operator-
//! configurable minimum age for outbound Directory gossip proofs
//! v0.17.0-DiscoveryGossipIsolation - Added bounded outbound gossip concurrency
//! with a fail-closed operator limit
//! v0.16.0-DirectoryMirrorCarrierCapability - Added a fail-closed staged
//! advertisement gate for signed Directory Mirror carrier capability
//! v0.15.0-FullNodeMirror - Added opt-in bounded non-authoritative Directory mirrors
//! v0.14.0-DirectoryWitnessThreshold — Added a bounded independent checkpoint corroboration threshold
//! v0.13.0-DirectorySyncPins — Added fail-closed Directory Sync peer admission pins
//! v0.12.0-DirectoryChainStore — Added optional fail-closed local directory ledger path
//! v0.11.0-VerifiedDeliveryWitnessAdmission — Added explicit witness requester pins
//! v0.10.0-VerifiedDeliveryWitness — Added optional pinned cache-anchor witnesses
//! v1.4.0-DiscoveryRelayCapabilities — Added explicit onion-middle advertisement gate
//! v1.3.0-TransportCapability — Added VPN transport capability accessors
//! v2.1.0            — Added MemChain config fields
//! v1.2.0-DNSOwnership — Added DNS proxy ownership accessor for server startup
//! v2.1.0+MVF+Auth   — Added api_secret
//! v2.3.0+RemoteStorage — Added allow_remote_storage, max_remote_owners
//! v2.4.0-GraphCognition — Added NER/graph/entropy/miner/vector fields
//! v2.5.0-SuperNode  — Added SuperNode config
//! v1.0.0-MultiTenant — Added SaaS mode
//! v1.1.0-ChatRelay  — 🌟 Split into multi-file layout; this file now thin
//! v0.7.0-DiscoverySafetyStatus — Added nodeboard status and safety policy config
//! v0.6.0-DiscoveryOutboundGossip — Added optional outbound peer gossip config
//! v0.5.0-DiscoveryPeerCache — Added optional local PeerStore cache config
//! v0.4.0-DiscoverySelfDescriptor — Added signed self descriptor config
//! v0.3.0-DiscoveryBootstrap — Added optional discovery bootstrap config
//! v0.9.0-DiscoveryGossipBackpressure — Added outbound gossip jitter/backpressure controls
//! v0.8.0-DiscoveryPublicApi — Added optional public-only discovery listener

use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Result, ServerError};
use crate::management::ManagementConfig;

mod discovery;

pub use discovery::DiscoveryConfig;

/// One validated, canonical local route-domain assignment.
///
/// [ROUTE-DOMAIN-POLICY-HISTORY 2026-08-03 by Codex] The opaque 128-bit
/// token groups nodes that the operator has independently reviewed as sharing
/// one routing failure domain. It is not an AS proof, operator identity,
/// network vote, or Sybil-resistance credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PinnedRouteDomainAssignment {
    pub(crate) node_id: [u8; 32],
    pub(crate) route_domain: [u8; 16],
}

// ── Sub-module re-exports (keep callers' use-paths stable) ────────────────
pub use crate::config_blind_vault::BlindVaultConfig;
pub use crate::config_chat_relay::ChatRelayConfig;
pub use crate::config_infra::{
    LimitsConfig, LoggingConfig, NetworkConfig, ServerKeyConfig, ServerKeySource, TunConfig,
    VpnConfig, VpnTransportConfig,
};
pub use crate::config_memchain::{MemChainConfig, MemChainMode, VectorQuantizationMode};
pub use crate::config_saas::SaasConfig;
pub use crate::config_supernode::SuperNodeConfig;

// ============================================
// ServerConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub vpn: VpnConfig,
    #[serde(default)]
    pub tun: TunConfig,
    #[serde(default)]
    pub server_key: ServerKeyConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub management: ManagementConfig,
    #[serde(default)]
    pub memchain: MemChainConfig,
    #[serde(default)]
    pub discovery: DiscoveryConfig,
    /// [BLIND-VAULT-SERVICE 2026-07-23 by Codex] Independent anonymous
    /// encrypted-object storage; disabled unless explicitly configured.
    #[serde(default)]
    pub blind_vault: BlindVaultConfig,
}

impl ServerConfig {
    /// Load and validate configuration from a TOML file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading configuration from: {}", path.display());
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        config.validate()?;
        info!("Configuration loaded successfully");
        Ok(config)
    }

    /// Parse and validate configuration from a TOML string (used in tests).
    pub fn from_str(content: &str) -> Result<Self> {
        let config: Self = toml::from_str(content)
            .map_err(|e| ServerError::config_load("<string>", e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate all sub-configs in dependency order.
    pub fn validate(&self) -> Result<()> {
        self.network.validate()?;
        self.vpn.validate()?;
        self.tun.validate()?;
        self.limits.validate()?;
        self.management
            .validate()
            .map_err(|e| ServerError::config_invalid("management", e))?;
        self.memchain.validate()?;
        self.discovery.validate()?;
        self.blind_vault.validate()?;
        if let Some(directory_path) = self.discovery.directory_chain_path.as_deref() {
            let directory_path = directory_path.trim();
            if [
                self.memchain.db_path.as_str(),
                self.memchain.chat_relay.db_path.as_str(),
            ]
            .into_iter()
            .any(|other| other.trim() == directory_path)
            {
                return Err(ServerError::config_invalid(
                    "discovery.directory_chain_path",
                    "must not reuse a MemChain or ChatRelay database path",
                ));
            }
        }
        if self.blind_vault.enabled {
            let blind_vault_path = self.blind_vault.db_path.trim();
            if [
                self.memchain.db_path.as_str(),
                self.memchain.chat_relay.db_path.as_str(),
            ]
            .into_iter()
            .any(|other| other.trim() == blind_vault_path)
                || self
                    .discovery
                    .directory_chain_path
                    .as_deref()
                    .is_some_and(|path| path.trim() == blind_vault_path)
            {
                return Err(ServerError::config_invalid(
                    "blind_vault.db_path",
                    "must not reuse a MemChain, ChatRelay, or Directory Chain database path",
                ));
            }
        }
        // [WITNESS-CONFIG-DIAGNOSTICS 2026-08-16 by Codex] Keep cross-module
        // errors bound to real TOML fields so installers can report and repair
        // the exact unsafe policy instead of receiving a wildcard pseudo-key.
        if !self.memchain.is_enabled()
            && !self.discovery.verified_delivery_witness_node_ids.is_empty()
        {
            return Err(ServerError::config_invalid(
                "discovery.verified_delivery_witness_node_ids",
                "authenticated witness transport requires a local MemChain storage mode",
            ));
        }
        if !self.memchain.is_enabled()
            && !self
                .discovery
                .verified_delivery_witness_requester_node_ids
                .is_empty()
        {
            return Err(ServerError::config_invalid(
                "discovery.verified_delivery_witness_requester_node_ids",
                "the authenticated delivery witness endpoint requires a local MemChain storage mode",
            ));
        }
        if !self.memchain.is_enabled()
            && !self
                .discovery
                .custody_audit_witness_requester_node_ids
                .is_empty()
        {
            return Err(ServerError::config_invalid(
                "discovery.custody_audit_witness_requester_node_ids",
                "the authenticated custody witness endpoint requires a local MemChain storage mode",
            ));
        }
        if !self.discovery.custody_audit_witness_node_ids.is_empty()
            && !self.memchain.is_chat_relay_enabled()
        {
            return Err(ServerError::config_invalid(
                "discovery.custody_audit_witness_node_ids",
                "requires memchain.chat_relay.enabled = true to produce custody anchors",
            ));
        }
        Ok(())
    }

    // ── Convenience accessors ──────────────────────────────────────────

    #[must_use]
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    #[must_use]
    pub fn listen_addr(&self) -> SocketAddr {
        self.network.listen_addr
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.tun.device_name
    }

    #[must_use]
    pub fn ip_range(&self) -> &str {
        &self.vpn.virtual_ip_range
    }

    #[must_use]
    pub fn gateway_ip(&self) -> Ipv4Addr {
        self.vpn.gateway_ip
    }

    /// Whether this node runs the VPN data plane.
    #[must_use]
    pub fn vpn_enabled(&self) -> bool {
        self.vpn.enabled
    }

    /// The gateway DNS proxy only exists as part of the VPN data plane.
    #[must_use]
    pub fn dns_proxy_enabled(&self) -> bool {
        self.vpn.enabled && self.vpn.dns_proxy_enabled
    }

    #[must_use]
    pub fn vpn_transports(&self) -> &VpnTransportConfig {
        &self.vpn.transports
    }

    #[must_use]
    pub fn mtu(&self) -> u16 {
        self.tun.mtu
    }

    #[must_use]
    pub fn max_sessions(&self) -> usize {
        self.limits.max_connections
    }

    #[must_use]
    pub fn session_timeout_secs(&self) -> u64 {
        self.limits.session_timeout
    }

    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> {
        self.vpn.parse_ip_range()
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            vpn: VpnConfig::default(),
            tun: TunConfig::default(),
            server_key: ServerKeyConfig::default(),
            limits: LimitsConfig::default(),
            logging: LoggingConfig::default(),
            management: ManagementConfig::default(),
            memchain: MemChainConfig::default(),
            discovery: DiscoveryConfig::default(),
            blind_vault: BlindVaultConfig::default(),
        }
    }
}

// ============================================
// Integration Tests
// (unit tests live in each sub-module's own #[cfg(test)])
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Full-stack default validation ─────────────────────────────────────

    #[test]
    fn test_default_config_valid() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
        // Spot-check a few fields to ensure sub-modules wired correctly
        assert_eq!(config.memchain.mvf_alpha, 0.5);
        assert!(!config.memchain.mvf_enabled);
        assert!(config.memchain.api_secret.is_none());
        assert!(!config.memchain.allow_remote_storage);
        assert!(!config.memchain.ner_enabled);
        assert!(!config.memchain.graph_enabled);
        assert!(!config.memchain.supernode.enabled);
        assert!(!config.memchain.is_saas());
        assert!(!config.memchain.is_chat_relay_enabled());
        assert!(config.dns_proxy_enabled());
        assert!(!config.discovery.enabled);
        assert!(config.discovery.advertise_self);
        assert_eq!(
            config.discovery.fetch_timeout_secs,
            DiscoveryConfig::default_fetch_timeout_secs()
        );
        assert!(config.discovery.peer_cache_path.is_none());
        assert!(config
            .discovery
            .verified_delivery_witness_node_ids
            .is_empty());
        assert!(config
            .discovery
            .verified_delivery_witness_requester_node_ids
            .is_empty());
        assert!(config
            .discovery
            .custody_audit_witness_requester_node_ids
            .is_empty());
        assert!(config.discovery.custody_audit_witness_node_ids.is_empty());
        assert_eq!(
            config.discovery.custody_audit_witness_min_verified,
            DiscoveryConfig::default_custody_audit_witness_min_verified()
        );
        assert!(!config.discovery.custody_audit_witness_startup_required);
        assert!(!config.discovery.custody_audit_witness_runtime_required);
        assert!(!config.discovery.custody_audit_witness_auto_renewal_enabled);
        assert_eq!(
            config.discovery.custody_audit_witness_max_age_secs,
            DiscoveryConfig::default_custody_audit_witness_max_age_secs()
        );
        assert_eq!(
            config.discovery.verified_delivery_witness_min_verified,
            DiscoveryConfig::default_verified_delivery_witness_min_verified()
        );
        assert!(
            !config
                .discovery
                .verified_delivery_witness_required_for_restore
        );
        assert_eq!(
            config.discovery.peer_cache_write_interval_secs,
            DiscoveryConfig::default_peer_cache_write_interval_secs()
        );
        assert_eq!(
            config.discovery.directory_chain_sync_interval_secs,
            DiscoveryConfig::default_directory_chain_sync_interval_secs()
        );
        assert!(config
            .discovery
            .directory_gossip_proof_min_age_secs
            .is_none());
        assert_eq!(
            config
                .discovery
                .effective_directory_gossip_proof_min_age_secs(),
            DiscoveryConfig::default_directory_chain_sync_interval_secs() * 2
        );
        assert!(!config.discovery.directory_full_node_mirror_enabled);
        assert!(!config.discovery.advertise_directory_mirror_carrier);
        assert_eq!(
            config.discovery.directory_full_node_mirror_max_producers,
            DiscoveryConfig::default_directory_full_node_mirror_max_producers()
        );
        assert_eq!(
            config.discovery.directory_observation_witness_min_verified,
            DiscoveryConfig::default_directory_observation_witness_min_verified()
        );
        assert!(!config.discovery.gossip_enabled);
        assert_eq!(
            config.discovery.gossip_interval_secs,
            DiscoveryConfig::default_gossip_interval_secs()
        );
        assert_eq!(
            config.discovery.gossip_peer_limit,
            DiscoveryConfig::default_gossip_peer_limit()
        );
        assert_eq!(
            config.discovery.gossip_concurrency_limit,
            DiscoveryConfig::default_gossip_concurrency_limit()
        );
        assert_eq!(
            config.discovery.gossip_jitter_percent,
            DiscoveryConfig::default_gossip_jitter_percent()
        );
        assert_eq!(
            config.discovery.gossip_backpressure_failure_threshold,
            DiscoveryConfig::default_gossip_backpressure_failure_threshold()
        );
        assert_eq!(
            config.discovery.gossip_failure_backoff_max_secs,
            DiscoveryConfig::default_gossip_failure_backoff_max_secs()
        );
        assert_eq!(
            config.discovery.max_peers,
            DiscoveryConfig::default_max_peers()
        );
        assert_eq!(
            config.discovery.max_snapshot_limit,
            DiscoveryConfig::default_max_snapshot_limit()
        );
        assert_eq!(
            config.discovery.gossip_rate_limit_per_minute,
            DiscoveryConfig::default_gossip_rate_limit_per_minute()
        );
        assert!(config.discovery.allowed_peer_ids.is_empty());
        assert!(config.discovery.denied_peer_ids.is_empty());
        assert!(config.discovery.pinned_route_domains.is_empty());
        assert!(!config.discovery.require_pinned_route_domains_for_multi_hop);
        assert!(config.discovery.route_domain_attestor_node_ids.is_empty());
        assert_eq!(config.discovery.route_domain_attestation_min_verified, 1);
        assert!(
            !config
                .discovery
                .require_route_domain_attestations_for_multi_hop
        );
        assert_eq!(
            config.discovery.descriptor_ttl_secs,
            DiscoveryConfig::default_descriptor_ttl_secs()
        );
        assert!(!config.discovery.permissionless_endpoint_proof_enabled);
        assert_eq!(
            config.discovery.permissionless_endpoint_proof_max_entries,
            DiscoveryConfig::default_permissionless_endpoint_proof_max_entries()
        );
        assert_eq!(
            config.discovery.permissionless_endpoint_proof_ttl_secs,
            DiscoveryConfig::default_permissionless_endpoint_proof_ttl_secs()
        );
        assert!(config.discovery.public_discovery);
    }

    #[test]
    fn test_dns_proxy_enabled_backward_compat_default() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(config.dns_proxy_enabled());
    }

    #[test]
    fn test_vpn_transports_backward_compat_default_udp_only() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(config.vpn_transports().udp_enabled);
        assert!(!config.vpn_transports().tcp_tls_enabled);
        assert!(!config.vpn_transports().websocket_enabled);
        assert_eq!(config.vpn_transports().preferred_transport, "udp");
    }

    #[test]
    fn test_vpn_transports_toml_parse_future_fallback_metadata() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"

[vpn.transports]
udp_enabled = true
tcp_tls_enabled = true
tcp_tls_public_endpoint = "vpn.example.com:443"
websocket_enabled = true
websocket_public_url = "wss://vpn.example.com/aeronyx/vpn"
preferred_transport = "udp"
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        let transports = config.vpn_transports();
        assert!(transports.udp_enabled);
        assert!(transports.tcp_tls_enabled);
        assert!(transports.websocket_enabled);
        assert_eq!(
            transports.tcp_tls_public_endpoint.as_deref(),
            Some("vpn.example.com:443")
        );
        assert_eq!(
            transports.websocket_public_url.as_deref(),
            Some("wss://vpn.example.com/aeronyx/vpn")
        );
    }

    #[test]
    fn test_dns_proxy_can_be_disabled_for_external_gateway_dns() {
        let toml_str = r#"
[vpn]
virtual_ip_range = "100.64.0.0/22"
gateway_ip = "100.64.0.1"
dns_proxy_enabled = false
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(!config.dns_proxy_enabled());
    }

    #[test]
    fn the_shipped_example_config_loads_and_its_seeds_are_current() {
        // [NODE-SEED-REFRESH 2026-10-09 by Claude] install.sh copies this file
        // into every new node's server.toml, so a dead seed here lands in
        // every new node. It must load with the real parser and validator, and
        // the retired addresses must not come back.
        let example = include_str!("../../../deploy/node/server.example.toml");
        let config = ServerConfig::from_str(example).expect("example config loads and validates");
        let seeds = &config.discovery.seed_endpoints;
        assert!(
            seeds.len() >= 3,
            "keep several seeds so one outage is survivable"
        );
        let retired = [
            "149.33.18.44",
            "35.253.79.169",
            "34.136.167.59",
            "34.58.204.226",
        ];
        for seed in seeds {
            for address in retired {
                assert!(
                    !seed.contains(address),
                    "retired seed {address} is back in the example"
                );
            }
        }
        let distinct: std::collections::HashSet<_> = seeds.iter().collect();
        assert_eq!(distinct.len(), seeds.len(), "duplicate seed");
    }

    // ── v1.1.0-ChatRelay: full TOML integration ───────────────────────────

    #[test]
    fn test_chat_relay_full_toml_integration() {
        let toml_str = r#"
[memchain]
mode = "local"

[memchain.chat_relay]
enabled = true
offline_ttl_secs = 86400
max_pending_per_wallet = 200
db_path = "data/chat_test.db"
max_message_size = 32768
max_blob_size = 5242880
max_blobs_per_receiver = 20
cleanup_interval_secs = 30
dedup_lru_capacity = 5000
expired_notification_ttl_secs = 172800
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let cr = &config.memchain.chat_relay;
        assert!(cr.enabled);
        assert_eq!(cr.offline_ttl_secs, 86_400);
        assert_eq!(cr.max_pending_per_wallet, 200);
        assert_eq!(cr.db_path, "data/chat_test.db");
        assert_eq!(cr.max_message_size, 32_768);
        assert_eq!(cr.max_blob_size, 5_242_880);
        assert_eq!(cr.max_blobs_per_receiver, 20);
        assert_eq!(cr.cleanup_interval_secs, 30);
        assert_eq!(cr.dedup_lru_capacity, 5_000);
        assert_eq!(cr.expired_notification_ttl_secs, 172_800);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_chat_relay_backward_compat_no_section() {
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.memchain.chat_relay.enabled);
        assert!(config.validate().is_ok());
    }

    // ── v2.5.0: SuperNode + v1.1.0 ChatRelay combined ────────────────────

    #[test]
    fn test_supernode_and_chat_relay_combined() {
        let toml_str = r#"
[memchain]
mode = "local"
ner_enabled = true

[memchain.supernode]
enabled = true

[[memchain.supernode.providers]]
name = "ollama"
type = "openai_compatible"
api_base = "http://localhost:11434/v1"
model = "llama3"

[memchain.chat_relay]
enabled = true
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.memchain.is_supernode_enabled());
        assert!(config.memchain.is_chat_relay_enabled());
        assert!(config.validate().is_ok());
    }

    // ── v1.0.0-MT + v1.1.0-ChatRelay combined ────────────────────────────

    #[test]
    fn test_saas_and_chat_relay_combined() {
        let toml_str = r#"
[memchain]
mode = "saas"
jwt_secret = "a-very-long-secret-key-that-is-at-least-32-chars"

[memchain.saas]
data_root = "/var/memchain"
pool_max_connections = 50

[memchain.chat_relay]
enabled = true
db_path = "/var/memchain/chat_pending.db"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.memchain.is_saas());
        assert!(config.memchain.is_chat_relay_enabled());
        assert!(config.validate().is_ok());
    }

    // ── v2.4.0: Full cognitive graph TOML ────────────────────────────────

    #[test]
    fn test_v240_toml_full_config() {
        let toml_str = r#"
[memchain]
mode = "local"
ner_enabled = true
ner_model_path = "models/gliner"
ner_confidence_threshold = 0.45
graph_enabled = true
graph_max_depth = 2
graph_max_nodes_per_hop = 30
graph_min_edge_weight = 0.25
entropy_filter_enabled = true
entropy_filter_threshold = 0.4
entropy_window_size = 8
entropy_window_overlap = 1
miner_entity_extraction = true
miner_community_detection = true
miner_session_summary = true
miner_artifact_extraction = true
vector_quantization = "scalar_uint8"
vector_early_termination = true
vector_saturation_threshold = 3
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert!(mc.ner_enabled);
        assert!(mc.graph_enabled);
        assert!(mc.entropy_filter_enabled);
        assert!(mc.miner_entity_extraction);
        assert_eq!(mc.vector_quantization, VectorQuantizationMode::ScalarUint8);
        assert!(config.validate().is_ok());
        assert!(mc.is_cognitive_graph_enabled());
        assert!(mc.has_cognitive_miner_steps());
        assert!(!mc.is_supernode_enabled());
        assert!(!mc.is_saas());
        assert!(!mc.is_chat_relay_enabled());
    }

    // ── Backward compatibility: old TOML with none of the new sections ────

    #[test]
    fn test_full_backward_compat() {
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
mvf_alpha = 0.5
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert!(!mc.ner_enabled);
        assert!(!mc.graph_enabled);
        assert!(!mc.entropy_filter_enabled);
        assert!(!mc.supernode.enabled);
        assert!(!mc.is_saas());
        assert!(!mc.is_chat_relay_enabled());
        assert!(config.validate().is_ok());
    }

    // ── v2.5.0: EmbeddingGemma config ────────────────────────────────────

    #[test]
    fn test_embed_gemma_config() {
        let toml_str = r#"
[memchain]
mode = "local"
embed_model_path = "models/embeddinggemma"
embed_max_tokens = 256
embed_output_dim = 384
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert_eq!(mc.embed_model_path, "models/embeddinggemma");
        assert_eq!(mc.embed_output_dim, 384);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_embed_engine_can_be_disabled_for_protocol_nodes() {
        let toml_str = r#"
[memchain]
mode = "local"
embed_enabled = false
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        let mc = &config.memchain;
        assert!(!mc.embed_enabled);
        assert!(config.validate().is_ok());
    }
}
