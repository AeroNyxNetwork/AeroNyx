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
//! ## Dependencies
//! - config_infra.rs    — infrastructure configs (no MemChain awareness)
//! - config_memchain.rs — MemChain + all nested subsystem configs
//! - config_supernode.rs — SuperNodeConfig (re-exported via config_memchain)
//! - config_saas.rs     — SaasConfig (re-exported via config_memchain)
//! - config_chat_relay.rs — ChatRelayConfig (re-exported via config_memchain)
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
//! v0.8.0-DiscoveryPublicApi — Added optional public-only discovery listener

use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Result, ServerError};
use crate::management::ManagementConfig;

// ── Sub-module re-exports (keep callers' use-paths stable) ────────────────
pub use crate::config_chat_relay::ChatRelayConfig;
pub use crate::config_infra::{
    LimitsConfig, LoggingConfig, NetworkConfig, ServerKeyConfig, TunConfig, VpnConfig,
    VpnTransportConfig,
};
pub use crate::config_memchain::{MemChainConfig, MemChainMode, VectorQuantizationMode};
pub use crate::config_saas::SaasConfig;
pub use crate::config_supernode::SuperNodeConfig;

// ============================================
// DiscoveryConfig
// ============================================

/// Configuration for decentralized node discovery bootstrap and self advertisement.
///
/// The bootstrap layer is disabled by default for backward compatibility.
/// When enabled, the node can hydrate its verified in-memory peer store from
/// a local JSON snapshot and/or an HTTPS JSON snapshot URL, then optionally
/// sign and publish its own descriptor into the local peer store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enables bootstrap snapshot loading at server startup.
    #[serde(default)]
    pub enabled: bool,
    /// Enables generating this node's signed descriptor at startup.
    #[serde(default = "DiscoveryConfig::default_advertise_self")]
    pub advertise_self: bool,
    /// Optional local JSON bootstrap snapshot path.
    #[serde(default)]
    pub bootstrap_snapshot_path: Option<String>,
    /// Optional HTTP(S) JSON bootstrap snapshot URL.
    #[serde(default)]
    pub bootstrap_snapshot_url: Option<String>,
    /// Optional public discovery endpoints contacted on every gossip round.
    ///
    /// These seed endpoints are not trusted authorities; they only provide
    /// signed discovery gossip/snapshot transport so nodes can recover when a
    /// cached peer descriptor has an outdated public endpoint.
    #[serde(default)]
    pub seed_endpoints: Vec<String>,
    /// Timeout in seconds for fetching a remote bootstrap snapshot.
    #[serde(default = "DiscoveryConfig::default_fetch_timeout_secs")]
    pub fetch_timeout_secs: u64,
    /// Optional local verified peer cache path.
    ///
    /// The cache uses the same JSON schema as bootstrap snapshots and is
    /// re-verified on every load, so stale or tampered descriptors are skipped.
    #[serde(default)]
    pub peer_cache_path: Option<String>,
    /// Periodic cache write interval in seconds.
    #[serde(default = "DiscoveryConfig::default_peer_cache_write_interval_secs")]
    pub peer_cache_write_interval_secs: u64,
    /// Enables periodic outbound discovery gossip to known public peers.
    ///
    /// Kept disabled by default so simply enabling bootstrap does not create
    /// unexpected outbound network traffic.
    #[serde(default)]
    pub gossip_enabled: bool,
    /// Periodic outbound gossip interval in seconds.
    #[serde(default = "DiscoveryConfig::default_gossip_interval_secs")]
    pub gossip_interval_secs: u64,
    /// Maximum number of public peers contacted per gossip round.
    #[serde(default = "DiscoveryConfig::default_gossip_peer_limit")]
    pub gossip_peer_limit: u16,
    /// Maximum descriptors retained in the local verified peer store.
    #[serde(default = "DiscoveryConfig::default_max_peers")]
    pub max_peers: usize,
    /// Maximum descriptors returned by a single snapshot response.
    #[serde(default = "DiscoveryConfig::default_max_snapshot_limit")]
    pub max_snapshot_limit: usize,
    /// Global inbound gossip request budget per minute.
    #[serde(default = "DiscoveryConfig::default_gossip_rate_limit_per_minute")]
    pub gossip_rate_limit_per_minute: u32,
    /// Optional allow-list of peer node ids as lowercase/uppercase hex.
    #[serde(default)]
    pub allowed_peer_ids: Vec<String>,
    /// Optional deny-list of peer node ids as lowercase/uppercase hex.
    #[serde(default)]
    pub denied_peer_ids: Vec<String>,
    /// Optional discovery control-plane endpoint advertised to other nodes.
    ///
    /// When absent, `network.public_endpoint` is reused. If both are absent,
    /// the node still signs a descriptor but leaves endpoint discovery empty.
    #[serde(default)]
    pub public_endpoint: Option<String>,
    /// Optional public-only API listener for discovery and peer chat relay.
    ///
    /// This listener is separate from `memchain.api_listen_addr` and exposes
    /// only `/api/discovery/*` plus `/api/chat/peer/relay`. It stays disabled
    /// by default so existing deployments never expose the full local API.
    #[serde(default)]
    pub public_api_listen_addr: Option<SocketAddr>,
    /// Optional region label for nodeboard and future peer selection.
    #[serde(default)]
    pub region: Option<String>,
    /// Descriptor validity window in seconds.
    #[serde(default = "DiscoveryConfig::default_descriptor_ttl_secs")]
    pub descriptor_ttl_secs: u64,
    /// Whether this node may appear in public bootstrap snapshots.
    #[serde(default = "DiscoveryConfig::default_public_discovery")]
    pub public_discovery: bool,
}

impl DiscoveryConfig {
    /// Default self advertisement behavior when discovery is enabled.
    #[must_use]
    pub const fn default_advertise_self() -> bool {
        true
    }

    /// Default remote bootstrap fetch timeout.
    #[must_use]
    pub const fn default_fetch_timeout_secs() -> u64 {
        10
    }

    /// Default local peer cache write interval.
    #[must_use]
    pub const fn default_peer_cache_write_interval_secs() -> u64 {
        300
    }

    /// Default outbound gossip interval.
    #[must_use]
    pub const fn default_gossip_interval_secs() -> u64 {
        60
    }

    /// Default outbound gossip peer limit per round.
    #[must_use]
    pub const fn default_gossip_peer_limit() -> u16 {
        32
    }

    /// Default maximum verified peers retained locally.
    #[must_use]
    pub const fn default_max_peers() -> usize {
        2048
    }

    /// Default maximum descriptors in one snapshot response.
    #[must_use]
    pub const fn default_max_snapshot_limit() -> usize {
        256
    }

    /// Default global inbound gossip request budget per minute.
    #[must_use]
    pub const fn default_gossip_rate_limit_per_minute() -> u32 {
        120
    }

    /// Default signed descriptor time-to-live.
    #[must_use]
    pub const fn default_descriptor_ttl_secs() -> u64 {
        3600
    }

    /// Default public discovery visibility.
    #[must_use]
    pub const fn default_public_discovery() -> bool {
        true
    }

    /// Validates discovery bootstrap configuration.
    pub fn validate(&self) -> Result<()> {
        if self.fetch_timeout_secs == 0 {
            return Err(ServerError::config_invalid(
                "discovery.fetch_timeout_secs",
                "must be greater than zero",
            ));
        }

        if let Some(path) = &self.bootstrap_snapshot_path {
            if path.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.bootstrap_snapshot_path",
                    "must not be empty when provided",
                ));
            }
        }

        if let Some(url) = &self.bootstrap_snapshot_url {
            let trimmed = url.trim();
            if trimmed.is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.bootstrap_snapshot_url",
                    "must not be empty when provided",
                ));
            }
            if !(trimmed.starts_with("https://") || trimmed.starts_with("http://")) {
                return Err(ServerError::config_invalid(
                    "discovery.bootstrap_snapshot_url",
                    "must start with http:// or https://",
                ));
            }
        }

        for endpoint in &self.seed_endpoints {
            Self::validate_seed_endpoint(endpoint)?;
        }

        if let Some(path) = &self.peer_cache_path {
            if path.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.peer_cache_path",
                    "must not be empty when provided",
                ));
            }
        }

        if self.peer_cache_write_interval_secs < 30 {
            return Err(ServerError::config_invalid(
                "discovery.peer_cache_write_interval_secs",
                "must be at least 30 seconds",
            ));
        }

        if self.gossip_interval_secs < 30 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_interval_secs",
                "must be at least 30 seconds",
            ));
        }

        if self.gossip_peer_limit == 0 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_peer_limit",
                "must be greater than zero",
            ));
        }

        if self.max_peers == 0 {
            return Err(ServerError::config_invalid(
                "discovery.max_peers",
                "must be greater than zero",
            ));
        }

        if self.max_snapshot_limit == 0 {
            return Err(ServerError::config_invalid(
                "discovery.max_snapshot_limit",
                "must be greater than zero",
            ));
        }

        if self.gossip_rate_limit_per_minute == 0 {
            return Err(ServerError::config_invalid(
                "discovery.gossip_rate_limit_per_minute",
                "must be greater than zero",
            ));
        }

        for peer_id in self
            .allowed_peer_ids
            .iter()
            .chain(self.denied_peer_ids.iter())
        {
            let trimmed = peer_id.trim();
            if trimmed.len() != 64 || !trimmed.chars().all(|ch| ch.is_ascii_hexdigit()) {
                return Err(ServerError::config_invalid(
                    "discovery.allowed_peer_ids/denied_peer_ids",
                    "peer ids must be 32-byte hex strings",
                ));
            }
        }

        if let Some(endpoint) = &self.public_endpoint {
            if endpoint.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.public_endpoint",
                    "must not be empty when provided",
                ));
            }
        }

        if let Some(addr) = self.public_api_listen_addr {
            if addr.port() == 0 {
                return Err(ServerError::config_invalid(
                    "discovery.public_api_listen_addr",
                    "port must be greater than zero",
                ));
            }
        }

        if let Some(region) = &self.region {
            if region.trim().is_empty() {
                return Err(ServerError::config_invalid(
                    "discovery.region",
                    "must not be empty when provided",
                ));
            }
        }

        if self.descriptor_ttl_secs < 60 {
            return Err(ServerError::config_invalid(
                "discovery.descriptor_ttl_secs",
                "must be at least 60 seconds",
            ));
        }

        Ok(())
    }

    fn validate_seed_endpoint(endpoint: &str) -> Result<()> {
        let trimmed = endpoint.trim();
        if trimmed.is_empty() {
            return Err(ServerError::config_invalid(
                "discovery.seed_endpoints",
                "entries must not be empty",
            ));
        }
        if trimmed.contains(char::is_whitespace) {
            return Err(ServerError::config_invalid(
                "discovery.seed_endpoints",
                "entries must not contain whitespace",
            ));
        }
        if !(trimmed.starts_with("http://")
            || trimmed.starts_with("https://")
            || trimmed.contains(':'))
        {
            return Err(ServerError::config_invalid(
                "discovery.seed_endpoints",
                "entries must be http(s) URLs or host:port endpoints",
            ));
        }
        Ok(())
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            advertise_self: Self::default_advertise_self(),
            bootstrap_snapshot_path: None,
            bootstrap_snapshot_url: None,
            seed_endpoints: Vec::new(),
            fetch_timeout_secs: Self::default_fetch_timeout_secs(),
            peer_cache_path: None,
            peer_cache_write_interval_secs: Self::default_peer_cache_write_interval_secs(),
            gossip_enabled: false,
            gossip_interval_secs: Self::default_gossip_interval_secs(),
            gossip_peer_limit: Self::default_gossip_peer_limit(),
            max_peers: Self::default_max_peers(),
            max_snapshot_limit: Self::default_max_snapshot_limit(),
            gossip_rate_limit_per_minute: Self::default_gossip_rate_limit_per_minute(),
            allowed_peer_ids: Vec::new(),
            denied_peer_ids: Vec::new(),
            public_endpoint: None,
            public_api_listen_addr: None,
            region: None,
            descriptor_ttl_secs: Self::default_descriptor_ttl_secs(),
            public_discovery: Self::default_public_discovery(),
        }
    }
}

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

    #[must_use]
    pub fn dns_proxy_enabled(&self) -> bool {
        self.vpn.dns_proxy_enabled
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
        assert_eq!(
            config.discovery.peer_cache_write_interval_secs,
            DiscoveryConfig::default_peer_cache_write_interval_secs()
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
        assert_eq!(
            config.discovery.descriptor_ttl_secs,
            DiscoveryConfig::default_descriptor_ttl_secs()
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
    fn test_discovery_backward_compat_default_disabled() {
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(!config.discovery.enabled);
        assert!(config.discovery.bootstrap_snapshot_path.is_none());
        assert!(config.discovery.bootstrap_snapshot_url.is_none());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_discovery_bootstrap_toml_parse() {
        let toml_str = r#"
[discovery]
enabled = true
bootstrap_snapshot_path = "/etc/aeronyx/bootstrap-peers.json"
bootstrap_snapshot_url = "https://nodes.aeronyx.network/bootstrap.json"
seed_endpoints = ["http://34.136.167.59:8422", "8.213.146.244:8422"]
fetch_timeout_secs = 15
peer_cache_path = "/var/lib/aeronyx/peers-cache.json"
peer_cache_write_interval_secs = 120
gossip_enabled = true
gossip_interval_secs = 45
gossip_peer_limit = 8
max_peers = 512
max_snapshot_limit = 64
gossip_rate_limit_per_minute = 30
allowed_peer_ids = ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
denied_peer_ids = ["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
public_endpoint = "node.example.com:443"
public_api_listen_addr = "0.0.0.0:8422"
region = "us-central"
descriptor_ttl_secs = 7200
public_discovery = false
"#;
        let config = ServerConfig::from_str(toml_str).unwrap();
        assert!(config.discovery.enabled);
        assert!(config.discovery.advertise_self);
        assert_eq!(
            config.discovery.bootstrap_snapshot_path.as_deref(),
            Some("/etc/aeronyx/bootstrap-peers.json")
        );
        assert_eq!(
            config.discovery.bootstrap_snapshot_url.as_deref(),
            Some("https://nodes.aeronyx.network/bootstrap.json")
        );
        assert_eq!(
            config.discovery.seed_endpoints,
            vec![
                "http://34.136.167.59:8422".to_string(),
                "8.213.146.244:8422".to_string()
            ]
        );
        assert_eq!(config.discovery.fetch_timeout_secs, 15);
        assert_eq!(
            config.discovery.peer_cache_path.as_deref(),
            Some("/var/lib/aeronyx/peers-cache.json")
        );
        assert_eq!(config.discovery.peer_cache_write_interval_secs, 120);
        assert!(config.discovery.gossip_enabled);
        assert_eq!(config.discovery.gossip_interval_secs, 45);
        assert_eq!(config.discovery.gossip_peer_limit, 8);
        assert_eq!(config.discovery.max_peers, 512);
        assert_eq!(config.discovery.max_snapshot_limit, 64);
        assert_eq!(config.discovery.gossip_rate_limit_per_minute, 30);
        assert_eq!(
            config.discovery.allowed_peer_ids,
            vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
        );
        assert_eq!(
            config.discovery.denied_peer_ids,
            vec!["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
        );
        assert_eq!(
            config.discovery.public_endpoint.as_deref(),
            Some("node.example.com:443")
        );
        assert_eq!(
            config.discovery.public_api_listen_addr,
            Some("0.0.0.0:8422".parse().unwrap())
        );
        assert_eq!(config.discovery.region.as_deref(), Some("us-central"));
        assert_eq!(config.discovery.descriptor_ttl_secs, 7200);
        assert!(!config.discovery.public_discovery);
    }

    #[test]
    fn test_discovery_rejects_invalid_url_scheme() {
        let toml_str = r#"
[discovery]
enabled = true
bootstrap_snapshot_url = "file:///tmp/bootstrap.json"
"#;
        assert!(ServerConfig::from_str(toml_str).is_err());
    }

    #[test]
    fn test_discovery_rejects_invalid_seed_endpoint() {
        let empty_seed = r#"
[discovery]
enabled = true
seed_endpoints = [""]
"#;
        assert!(ServerConfig::from_str(empty_seed).is_err());

        let missing_scheme_or_port = r#"
[discovery]
enabled = true
seed_endpoints = ["node.example.com"]
"#;
        assert!(ServerConfig::from_str(missing_scheme_or_port).is_err());
    }

    #[test]
    fn test_discovery_rejects_zero_timeout() {
        let toml_str = r#"
[discovery]
enabled = true
fetch_timeout_secs = 0
"#;
        assert!(ServerConfig::from_str(toml_str).is_err());
    }

    #[test]
    fn test_discovery_rejects_short_peer_cache_interval() {
        let toml_str = r#"
[discovery]
enabled = true
peer_cache_path = "/var/lib/aeronyx/peers-cache.json"
peer_cache_write_interval_secs = 5
"#;
        assert!(ServerConfig::from_str(toml_str).is_err());
    }

    #[test]
    fn test_discovery_rejects_zero_public_api_port() {
        let toml_str = r#"
[discovery]
enabled = true
public_api_listen_addr = "0.0.0.0:0"
"#;
        assert!(ServerConfig::from_str(toml_str).is_err());
    }

    #[test]
    fn test_discovery_rejects_invalid_gossip_policy() {
        let short_interval = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_interval_secs = 5
"#;
        assert!(ServerConfig::from_str(short_interval).is_err());

        let zero_limit = r#"
[discovery]
enabled = true
gossip_enabled = true
gossip_peer_limit = 0
"#;
        assert!(ServerConfig::from_str(zero_limit).is_err());
    }

    #[test]
    fn test_discovery_rejects_invalid_safety_policy() {
        let zero_max_peers = r#"
[discovery]
enabled = true
max_peers = 0
"#;
        assert!(ServerConfig::from_str(zero_max_peers).is_err());

        let zero_snapshot_limit = r#"
[discovery]
enabled = true
max_snapshot_limit = 0
"#;
        assert!(ServerConfig::from_str(zero_snapshot_limit).is_err());

        let zero_rate_limit = r#"
[discovery]
enabled = true
gossip_rate_limit_per_minute = 0
"#;
        assert!(ServerConfig::from_str(zero_rate_limit).is_err());

        let bad_peer_id = r#"
[discovery]
enabled = true
allowed_peer_ids = ["not-a-node-id"]
"#;
        assert!(ServerConfig::from_str(bad_peer_id).is_err());
    }

    #[test]
    fn test_discovery_rejects_short_descriptor_ttl() {
        let toml_str = r#"
[discovery]
enabled = true
descriptor_ttl_secs = 10
"#;
        assert!(ServerConfig::from_str(toml_str).is_err());
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
}
