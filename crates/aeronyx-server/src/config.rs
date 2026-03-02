// ============================================
// File: crates/aeronyx-server/src/config.rs
// ============================================
//! # Server Configuration
//!
//! ## Creation Reason
//! Provides configuration management for the AeroNyx server,
//! supporting TOML files and environment variables.
//!
//! ## Modification Reason
//! - Added ManagementConfig for CMS integration.
//! - 🌟 Added MemChainConfig for MemChain mode control and API listen address.
//! - 🌟 v0.4.0: Added `trusted_agents` whitelist to MemChainConfig for
//!   Phase 3 P2P trust boundary enforcement.
//!
//! ## Main Functionality
//! - `ServerConfig`: Main configuration structure
//! - TOML file loading and parsing
//! - Configuration validation
//! - Default values for MVP
//!
//! ## Configuration Sections
//! - `network`: UDP listen address, public endpoint
//! - `vpn`: Virtual IP range, gateway
//! - `tun`: TUN device settings
//! - `server_key`: Key file path
//! - `limits`: Connection and resource limits
//! - `logging`: Log level
//! - `management`: CMS integration settings (REQUIRED)
//! - `memchain`: 🌟 MemChain mode and API settings (NEW)
//!
//! ## Example Configuration
//! ```toml
//! [network]
//! listen_addr = "0.0.0.0:51820"
//! public_endpoint = "1.2.3.4:51820"
//!
//! [vpn]
//! virtual_ip_range = "100.64.0.0/24"
//! gateway_ip = "100.64.0.1"
//!
//! [tun]
//! device_name = "aeronyx0"
//! mtu = 1420
//!
//! [limits]
//! max_connections = 1000
//! session_timeout = 300
//!
//! [management]
//! cms_url = "https://api.aeronyx.network/api/privacy_network"
//! heartbeat_interval_secs = 30
//!
//! [memchain]
//! mode = "local"                       # "local", "p2p", or "off"
//! api_listen_addr = "127.0.0.1:8421"   # Agent API bind address
//! aof_path = ".memchain"               # Append-only ledger file
//! trusted_agents = [                   # Ed25519 pubkey hex whitelist
//!   "fa29c129f789d4f79ed2075c5c2706cdbcf8ae11196b13048174598e1dca4d54",
//! ]
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - All config changes require server restart
//! - Validate config before server startup
//! - Node registration is MANDATORY for official network
//! - MemChain API only binds to loopback by default (security)
//! - `trusted_agents` empty = trust all authenticated sessions (MVP fallback)
//! - `trusted_agents` hex strings are validated at startup (must be 64 hex chars)
//!
//! ## Last Modified
//! v0.1.0 - Initial configuration implementation
//! v0.2.0 - Added management configuration for CMS integration
//! v0.3.0 - 🌟 Added MemChainConfig for mode control and API settings
//! v0.4.0 - 🌟 Added trusted_agents whitelist for Phase 3 P2P trust

use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Result, ServerError};
use crate::management::ManagementConfig;

// ============================================
// ServerConfig
// ============================================

/// Main server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Network configuration.
    #[serde(default)]
    pub network: NetworkConfig,

    /// VPN configuration (virtual IP range).
    #[serde(default)]
    pub vpn: VpnConfig,

    /// TUN device configuration.
    #[serde(default)]
    pub tun: TunConfig,

    /// Server key configuration.
    #[serde(default)]
    pub server_key: ServerKeyConfig,

    /// Resource limits.
    #[serde(default)]
    pub limits: LimitsConfig,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Management/CMS configuration (REQUIRED for official network).
    #[serde(default)]
    pub management: ManagementConfig,

    /// 🌟 MemChain configuration (mode, API, storage, trust).
    #[serde(default)]
    pub memchain: MemChainConfig,
}

impl ServerConfig {
    /// Loads configuration from a TOML file.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.display().to_string();

        info!("Loading configuration from: {}", path_str);

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ServerError::config_load(&path_str, e.to_string()))?;

        let config: Self = toml::from_str(&content)
            .map_err(|e| ServerError::config_load(&path_str, e.to_string()))?;

        config.validate()?;

        info!("Configuration loaded successfully");
        Ok(config)
    }

    /// Loads configuration from a string (useful for testing).
    pub fn from_str(content: &str) -> Result<Self> {
        let config: Self = toml::from_str(content)
            .map_err(|e| ServerError::config_load("<string>", e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        self.network.validate()?;
        self.vpn.validate()?;
        self.tun.validate()?;
        self.limits.validate()?;

        self.management.validate().map_err(|e| {
            ServerError::config_invalid("management", e)
        })?;

        self.memchain.validate()?;

        Ok(())
    }

    /// Serializes configuration to TOML string.
    #[must_use]
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    // ========================================
    // Helper methods for compatibility
    // ========================================

    /// Returns listen address (from network config).
    pub fn listen_addr(&self) -> SocketAddr {
        self.network.listen_addr
    }

    /// Returns TUN device name.
    pub fn device_name(&self) -> &str {
        &self.tun.device_name
    }

    /// Returns virtual IP range.
    pub fn ip_range(&self) -> &str {
        &self.vpn.virtual_ip_range
    }

    /// Returns gateway IP.
    pub fn gateway_ip(&self) -> Ipv4Addr {
        self.vpn.gateway_ip
    }

    /// Returns MTU.
    pub fn mtu(&self) -> u16 {
        self.tun.mtu
    }

    /// Returns max sessions.
    pub fn max_sessions(&self) -> usize {
        self.limits.max_connections
    }

    /// Returns session timeout in seconds.
    pub fn session_timeout_secs(&self) -> u64 {
        self.limits.session_timeout
    }

    /// Parses the IP range and returns (network, prefix_len).
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
        }
    }
}

// ============================================
// 🌟 MemChainConfig
// ============================================

/// MemChain operating mode.
///
/// Controls whether the MemChain storage engine and API are active.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemChainMode {
    /// MemChain is disabled — no MemPool, no AOF, no API.
    Off,
    /// Local-only mode — MemPool + AOF + local HTTP API, no P2P broadcast.
    /// This is the default for MVP.
    Local,
    /// Full P2P mode — local storage + broadcast to connected peers.
    P2p,
}

impl Default for MemChainMode {
    fn default() -> Self {
        Self::Local
    }
}

/// MemChain configuration section.
///
/// # Example TOML
/// ```toml
/// [memchain]
/// mode = "p2p"
/// api_listen_addr = "127.0.0.1:8421"
/// aof_path = ".memchain"
/// trusted_agents = [
///   "fa29c129f789d4f79ed2075c5c2706cdbcf8ae11196b13048174598e1dca4d54",
/// ]
/// ```
///
/// # Trust Model
/// - If `trusted_agents` is **empty** (default), all facts from
///   authenticated VPN sessions are accepted (MVP behaviour).
/// - If `trusted_agents` is **non-empty**, only facts whose
///   `origin` public key appears in the list (or matches the
///   server's own identity) are accepted. Everything else is
///   dropped with a warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemChainConfig {
    /// Operating mode: "off", "local", or "p2p".
    #[serde(default)]
    pub mode: MemChainMode,

    /// HTTP API listen address for Agent integration.
    /// Defaults to loopback only for security.
    #[serde(default = "default_memchain_api_addr")]
    pub api_listen_addr: SocketAddr,

    /// Path to the append-only ledger file.
    #[serde(default = "default_memchain_aof_path")]
    pub aof_path: String,

    /// 🌟 Trusted agent Ed25519 public keys (hex-encoded).
    ///
    /// Empty = trust all authenticated sessions (MVP fallback).
    /// Non-empty = strict whitelist; only listed keys + server's own key
    /// are allowed to contribute Facts.
    #[serde(default)]
    pub trusted_agents: Vec<String>,
}

fn default_memchain_api_addr() -> SocketAddr {
    "127.0.0.1:8421".parse().unwrap()
}

fn default_memchain_aof_path() -> String {
    ".memchain".to_string()
}

impl MemChainConfig {
    /// Validates the MemChain configuration.
    pub fn validate(&self) -> Result<()> {
        if self.mode != MemChainMode::Off {
            // API port must not be zero
            if self.api_listen_addr.port() == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.api_listen_addr",
                    "port cannot be 0",
                ));
            }

            // AOF path must not be empty
            if self.aof_path.is_empty() {
                return Err(ServerError::config_invalid(
                    "memchain.aof_path",
                    "cannot be empty when memchain is enabled",
                ));
            }

            // Warn (but allow) non-loopback API addresses
            if !self.api_listen_addr.ip().is_loopback() {
                tracing::warn!(
                    "[MEMCHAIN] ⚠️ API is binding to non-loopback address {}. \
                     This exposes the MemChain API to the network!",
                    self.api_listen_addr
                );
            }

            // 🌟 Validate trusted_agents hex format
            for (i, hex_key) in self.trusted_agents.iter().enumerate() {
                if hex_key.len() != 64 {
                    return Err(ServerError::config_invalid(
                        &format!("memchain.trusted_agents[{}]", i),
                        format!(
                            "expected 64 hex chars (32 bytes), got {} chars: '{}'",
                            hex_key.len(),
                            hex_key
                        ),
                    ));
                }
                if hex::decode(hex_key).is_err() {
                    return Err(ServerError::config_invalid(
                        &format!("memchain.trusted_agents[{}]", i),
                        format!("invalid hex string: '{}'", hex_key),
                    ));
                }
            }

            if !self.trusted_agents.is_empty() {
                tracing::info!(
                    "[MEMCHAIN] 🔒 Trust whitelist active: {} trusted agent(s)",
                    self.trusted_agents.len()
                );
            } else {
                tracing::info!(
                    "[MEMCHAIN] ⚠️ No trusted_agents configured — \
                     accepting facts from all authenticated sessions"
                );
            }
        }

        Ok(())
    }

    /// Returns true if MemChain is enabled (local or p2p).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.mode != MemChainMode::Off
    }

    /// Returns true if P2P broadcast is enabled.
    #[must_use]
    pub fn is_p2p(&self) -> bool {
        self.mode == MemChainMode::P2p
    }

    /// Checks whether a given origin public key is trusted.
    ///
    /// # Arguments
    /// * `origin_hex` - Hex-encoded Ed25519 public key (64 chars)
    /// * `server_pubkey_hex` - This server's own public key (always trusted)
    ///
    /// # Returns
    /// - `true` if trusted (whitelist empty, or key in list, or is server's own key)
    /// - `false` if the key is explicitly not in a non-empty whitelist
    #[must_use]
    pub fn is_origin_trusted(&self, origin_hex: &str, server_pubkey_hex: &str) -> bool {
        // Server's own key is always trusted
        if origin_hex == server_pubkey_hex {
            return true;
        }

        // Empty whitelist = trust all (MVP fallback)
        if self.trusted_agents.is_empty() {
            return true;
        }

        // Check whitelist
        self.trusted_agents.iter().any(|trusted| trusted == origin_hex)
    }
}

impl Default for MemChainConfig {
    fn default() -> Self {
        Self {
            mode: MemChainMode::default(),
            api_listen_addr: default_memchain_api_addr(),
            aof_path: default_memchain_aof_path(),
            trusted_agents: Vec::new(),
        }
    }
}

// ============================================
// NetworkConfig
// ============================================

/// Network configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// UDP listen address.
    #[serde(default = "default_listen_addr")]
    pub listen_addr: SocketAddr,

    /// Public endpoint (IP:port) for clients to connect.
    #[serde(default)]
    pub public_endpoint: Option<String>,
}

fn default_listen_addr() -> SocketAddr {
    "0.0.0.0:51820".parse().unwrap()
}

impl NetworkConfig {
    fn validate(&self) -> Result<()> {
        if self.listen_addr.port() == 0 {
            return Err(ServerError::config_invalid(
                "network.listen_addr",
                "port cannot be 0",
            ));
        }
        Ok(())
    }

    /// Returns public IP if configured.
    pub fn public_ip(&self) -> Option<Ipv4Addr> {
        self.public_endpoint.as_ref().and_then(|ep| {
            ep.split(':').next().and_then(|ip| ip.parse().ok())
        })
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            public_endpoint: None,
        }
    }
}

// ============================================
// VpnConfig
// ============================================

/// VPN configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpnConfig {
    /// Virtual IP range (CIDR notation).
    #[serde(default = "default_ip_range")]
    pub virtual_ip_range: String,

    /// Gateway IP (server's virtual IP).
    #[serde(default = "default_gateway_ip")]
    pub gateway_ip: Ipv4Addr,
}

fn default_ip_range() -> String {
    "100.64.0.0/24".to_string()
}

fn default_gateway_ip() -> Ipv4Addr {
    Ipv4Addr::new(100, 64, 0, 1)
}

impl VpnConfig {
    fn validate(&self) -> Result<()> {
        if !self.virtual_ip_range.contains('/') {
            return Err(ServerError::config_invalid(
                "vpn.virtual_ip_range",
                "must be in CIDR notation (e.g., 100.64.0.0/24)",
            ));
        }
        Ok(())
    }

    /// Parses the IP range and returns (network, prefix_len).
    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> {
        let parts: Vec<&str> = self.virtual_ip_range.split('/').collect();
        if parts.len() != 2 {
            return Err(ServerError::config_invalid(
                "vpn.virtual_ip_range",
                "invalid CIDR format",
            ));
        }

        let network: Ipv4Addr = parts[0].parse().map_err(|_| {
            ServerError::config_invalid("vpn.virtual_ip_range", "invalid network address")
        })?;

        let prefix: u8 = parts[1].parse().map_err(|_| {
            ServerError::config_invalid("vpn.virtual_ip_range", "invalid prefix length")
        })?;

        if prefix > 32 {
            return Err(ServerError::config_invalid(
                "vpn.virtual_ip_range",
                "prefix length cannot exceed 32",
            ));
        }

        Ok((network, prefix))
    }
}

impl Default for VpnConfig {
    fn default() -> Self {
        Self {
            virtual_ip_range: default_ip_range(),
            gateway_ip: default_gateway_ip(),
        }
    }
}

// ============================================
// TunConfig
// ============================================

/// TUN device configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunConfig {
    /// TUN device name.
    #[serde(default = "default_device_name")]
    pub device_name: String,

    /// MTU size.
    #[serde(default = "default_mtu")]
    pub mtu: u16,
}

fn default_device_name() -> String {
    "aeronyx0".to_string()
}

fn default_mtu() -> u16 {
    1420
}

impl TunConfig {
    fn validate(&self) -> Result<()> {
        if self.device_name.is_empty() {
            return Err(ServerError::config_invalid(
                "tun.device_name",
                "cannot be empty",
            ));
        }

        if self.device_name.len() > 15 {
            return Err(ServerError::config_invalid(
                "tun.device_name",
                "cannot exceed 15 characters",
            ));
        }

        if self.mtu < 576 {
            return Err(ServerError::config_invalid(
                "tun.mtu",
                "must be at least 576",
            ));
        }

        if self.mtu > 9000 {
            return Err(ServerError::config_invalid(
                "tun.mtu",
                "cannot exceed 9000",
            ));
        }

        Ok(())
    }
}

impl Default for TunConfig {
    fn default() -> Self {
        Self {
            device_name: default_device_name(),
            mtu: default_mtu(),
        }
    }
}

// ============================================
// ServerKeyConfig
// ============================================

/// Server key configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerKeyConfig {
    /// Path to key file.
    #[serde(default = "default_key_file")]
    pub key_file: String,
}

fn default_key_file() -> String {
    "/etc/aeronyx/server_key.json".to_string()
}

impl Default for ServerKeyConfig {
    fn default() -> Self {
        Self {
            key_file: default_key_file(),
        }
    }
}

// ============================================
// LimitsConfig
// ============================================

/// Resource limits configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    /// Maximum concurrent connections/sessions.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Session timeout in seconds.
    #[serde(default = "default_session_timeout")]
    pub session_timeout: u64,
}

fn default_max_connections() -> usize {
    1000
}

fn default_session_timeout() -> u64 {
    300
}

impl LimitsConfig {
    fn validate(&self) -> Result<()> {
        if self.max_connections == 0 {
            return Err(ServerError::config_invalid(
                "limits.max_connections",
                "must be greater than 0",
            ));
        }

        if self.session_timeout == 0 {
            return Err(ServerError::config_invalid(
                "limits.session_timeout",
                "must be greater than 0",
            ));
        }

        Ok(())
    }
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_connections: default_max_connections(),
            session_timeout: default_session_timeout(),
        }
    }
}

// ============================================
// LoggingConfig
// ============================================

/// Logging configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error).
    #[serde(default = "default_log_level")]
    pub level: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
        }
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_existing_config_format() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:51820"
            public_endpoint = "8.213.146.244:51820"

            [vpn]
            virtual_ip_range = "100.64.0.0/24"
            gateway_ip = "100.64.0.1"

            [tun]
            device_name = "aeronyx0"
            mtu = 1420

            [server_key]
            key_file = "/etc/aeronyx/server_key.json"

            [limits]
            max_connections = 1000
            session_timeout = 300

            [logging]
            level = "info"
        "#;

        let config = ServerConfig::from_str(toml).unwrap();
        assert_eq!(config.network.listen_addr.port(), 51820);
        assert_eq!(config.vpn.virtual_ip_range, "100.64.0.0/24");
        assert_eq!(config.tun.device_name, "aeronyx0");
        assert_eq!(config.limits.max_connections, 1000);
        // memchain should default to "local" when not specified
        assert_eq!(config.memchain.mode, MemChainMode::Local);
        assert!(config.memchain.is_enabled());
        // trusted_agents should default to empty
        assert!(config.memchain.trusted_agents.is_empty());
    }

    #[test]
    fn test_config_with_management() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:51820"

            [vpn]
            virtual_ip_range = "100.64.0.0/24"
            gateway_ip = "100.64.0.1"

            [tun]
            device_name = "aeronyx0"

            [management]
            cms_url = "https://api.example.com/api/privacy_network"
            heartbeat_interval_secs = 30
        "#;

        let config = ServerConfig::from_str(toml).unwrap();
        assert_eq!(config.management.cms_url, "https://api.example.com/api/privacy_network");
        assert_eq!(config.management.heartbeat_interval_secs, 30);
    }

    #[test]
    fn test_config_with_memchain() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:51820"

            [vpn]
            virtual_ip_range = "100.64.0.0/24"
            gateway_ip = "100.64.0.1"

            [tun]
            device_name = "aeronyx0"

            [memchain]
            mode = "local"
            api_listen_addr = "127.0.0.1:9999"
            aof_path = "/data/my.memchain"
        "#;

        let config = ServerConfig::from_str(toml).unwrap();
        assert_eq!(config.memchain.mode, MemChainMode::Local);
        assert_eq!(config.memchain.api_listen_addr.port(), 9999);
        assert_eq!(config.memchain.aof_path, "/data/my.memchain");
        assert!(config.memchain.is_enabled());
        assert!(!config.memchain.is_p2p());
    }

    #[test]
    fn test_config_memchain_off() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:51820"

            [vpn]
            virtual_ip_range = "100.64.0.0/24"
            gateway_ip = "100.64.0.1"

            [tun]
            device_name = "aeronyx0"

            [memchain]
            mode = "off"
        "#;

        let config = ServerConfig::from_str(toml).unwrap();
        assert_eq!(config.memchain.mode, MemChainMode::Off);
        assert!(!config.memchain.is_enabled());
    }

    #[test]
    fn test_config_with_trusted_agents() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:51820"

            [vpn]
            virtual_ip_range = "100.64.0.0/24"
            gateway_ip = "100.64.0.1"

            [tun]
            device_name = "aeronyx0"

            [memchain]
            mode = "p2p"
            trusted_agents = [
                "fa29c129f789d4f79ed2075c5c2706cdbcf8ae11196b13048174598e1dca4d54",
                "ab12cd34ef56789012345678901234567890123456789012345678901234abcd",
            ]
        "#;

        let config = ServerConfig::from_str(toml).unwrap();
        assert_eq!(config.memchain.trusted_agents.len(), 2);
        assert!(config.memchain.is_p2p());
    }

    #[test]
    fn test_trusted_agents_invalid_hex_rejected() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:51820"

            [vpn]
            virtual_ip_range = "100.64.0.0/24"
            gateway_ip = "100.64.0.1"

            [tun]
            device_name = "aeronyx0"

            [memchain]
            mode = "local"
            trusted_agents = ["not_valid_hex"]
        "#;

        let result = ServerConfig::from_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_trusted_agents_wrong_length_rejected() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:51820"

            [vpn]
            virtual_ip_range = "100.64.0.0/24"
            gateway_ip = "100.64.0.1"

            [tun]
            device_name = "aeronyx0"

            [memchain]
            mode = "local"
            trusted_agents = ["abcd1234"]
        "#;

        let result = ServerConfig::from_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_origin_trusted() {
        let config = MemChainConfig {
            trusted_agents: vec![
                "aaaa0000000000000000000000000000000000000000000000000000000000aa".to_string(),
            ],
            ..Default::default()
        };

        let server_key = "bbbb0000000000000000000000000000000000000000000000000000000000bb";

        // Server's own key is always trusted
        assert!(config.is_origin_trusted(server_key, server_key));

        // Whitelisted key is trusted
        assert!(config.is_origin_trusted(
            "aaaa0000000000000000000000000000000000000000000000000000000000aa",
            server_key,
        ));

        // Unknown key is NOT trusted
        assert!(!config.is_origin_trusted(
            "cccc0000000000000000000000000000000000000000000000000000000000cc",
            server_key,
        ));
    }

    #[test]
    fn test_is_origin_trusted_empty_whitelist() {
        let config = MemChainConfig::default();
        let server_key = "bbbb0000000000000000000000000000000000000000000000000000000000bb";

        // Empty whitelist = trust everyone
        assert!(config.is_origin_trusted("any_key_at_all", server_key));
    }

    #[test]
    fn test_parse_ip_range() {
        let config = VpnConfig::default();
        let (network, prefix) = config.parse_ip_range().unwrap();

        assert_eq!(network, Ipv4Addr::new(100, 64, 0, 0));
        assert_eq!(prefix, 24);
    }

    #[test]
    fn test_public_ip_extraction() {
        let config = NetworkConfig {
            listen_addr: default_listen_addr(),
            public_endpoint: Some("8.213.146.244:51820".to_string()),
        };

        assert_eq!(config.public_ip(), Some(Ipv4Addr::new(8, 213, 146, 244)));
    }
}
