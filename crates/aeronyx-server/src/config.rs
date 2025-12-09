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
//! Added ManagementConfig for CMS integration.
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
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - All config changes require server restart
//! - Validate config before server startup
//! - Node registration is MANDATORY for official network
//!
//! ## Last Modified
//! v0.1.0 - Initial configuration implementation
//! v0.2.0 - Added management configuration for CMS integration

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
}

impl ServerConfig {
    /// Loads configuration from a TOML file.
    ///
    /// # Arguments
    /// * `path` - Path to the configuration file
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
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
