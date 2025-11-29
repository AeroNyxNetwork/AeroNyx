// ============================================
// File: crates/aeronyx-server/src/config.rs
// ============================================
//! # Server Configuration
//!
//! ## Creation Reason
//! Provides configuration management for the AeroNyx server,
//! supporting TOML files and environment variables.
//!
//! ## Main Functionality
//! - `ServerConfig`: Main configuration structure
//! - TOML file loading and parsing
//! - Configuration validation
//! - Default values for MVP
//!
//! ## Configuration Sections
//! - `network`: UDP listen address, public IP
//! - `tunnel`: TUN device settings, IP range
//! - `server_key`: Key file paths
//! - `limits`: Connection and resource limits
//! - `logging`: Log level and output
//!
//! ## Example Configuration
//! ```toml
//! [network]
//! listen_addr = "0.0.0.0:51820"
//! public_ip = "1.2.3.4"
//!
//! [tunnel]
//! device_name = "aeronyx0"
//! ip_range = "100.64.0.0/24"
//! gateway_ip = "100.64.0.1"
//! mtu = 1420
//!
//! [limits]
//! max_sessions = 1000
//! session_timeout_secs = 300
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - All config changes require server restart
//! - Validate config before server startup
//! - Sensitive values should use environment variables
//!
//! ## Last Modified
//! v0.1.0 - Initial configuration implementation

use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Result, ServerError};

// ============================================
// ServerConfig
// ============================================

/// Main server configuration.
///
/// # Example
/// ```
/// use aeronyx_server::config::ServerConfig;
///
/// // Load from default values
/// let config = ServerConfig::default();
///
/// // Load from file
/// // let config = ServerConfig::load("config.toml").await?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Network configuration.
    #[serde(default)]
    pub network: NetworkConfig,

    /// TUN device configuration.
    #[serde(default)]
    pub tunnel: TunnelConfig,

    /// Server key configuration.
    #[serde(default)]
    pub server_key: ServerKeyConfig,

    /// Resource limits.
    #[serde(default)]
    pub limits: LimitsConfig,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,
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
    ///
    /// # Arguments
    /// * `content` - TOML configuration string
    ///
    /// # Errors
    /// Returns error if parsing fails.
    pub fn from_str(content: &str) -> Result<Self> {
        let config: Self = toml::from_str(content)
            .map_err(|e| ServerError::config_load("<string>", e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validates the configuration.
    ///
    /// # Errors
    /// Returns error if any configuration value is invalid.
    pub fn validate(&self) -> Result<()> {
        self.network.validate()?;
        self.tunnel.validate()?;
        self.limits.validate()?;
        Ok(())
    }

    /// Serializes configuration to TOML string.
    #[must_use]
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            tunnel: TunnelConfig::default(),
            server_key: ServerKeyConfig::default(),
            limits: LimitsConfig::default(),
            logging: LoggingConfig::default(),
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

    /// Public IP address (for clients behind NAT).
    #[serde(default)]
    pub public_ip: Option<Ipv4Addr>,
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
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            public_ip: None,
        }
    }
}

// ============================================
// TunnelConfig
// ============================================

/// TUN device configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelConfig {
    /// TUN device name.
    #[serde(default = "default_device_name")]
    pub device_name: String,

    /// Virtual IP range (CIDR notation).
    #[serde(default = "default_ip_range")]
    pub ip_range: String,

    /// Gateway IP (server's virtual IP).
    #[serde(default = "default_gateway_ip")]
    pub gateway_ip: Ipv4Addr,

    /// MTU size.
    #[serde(default = "default_mtu")]
    pub mtu: u16,
}

fn default_device_name() -> String {
    "aeronyx0".to_string()
}

fn default_ip_range() -> String {
    "100.64.0.0/24".to_string()
}

fn default_gateway_ip() -> Ipv4Addr {
    Ipv4Addr::new(100, 64, 0, 1)
}

fn default_mtu() -> u16 {
    1420
}

impl TunnelConfig {
    fn validate(&self) -> Result<()> {
        if self.device_name.is_empty() {
            return Err(ServerError::config_invalid(
                "tunnel.device_name",
                "cannot be empty",
            ));
        }

        if self.device_name.len() > 15 {
            return Err(ServerError::config_invalid(
                "tunnel.device_name",
                "cannot exceed 15 characters",
            ));
        }

        if self.mtu < 576 {
            return Err(ServerError::config_invalid(
                "tunnel.mtu",
                "must be at least 576",
            ));
        }

        if self.mtu > 9000 {
            return Err(ServerError::config_invalid(
                "tunnel.mtu",
                "cannot exceed 9000",
            ));
        }

        // Validate IP range format
        if !self.ip_range.contains('/') {
            return Err(ServerError::config_invalid(
                "tunnel.ip_range",
                "must be in CIDR notation (e.g., 100.64.0.0/24)",
            ));
        }

        Ok(())
    }

    /// Parses the IP range and returns (network, prefix_len).
    ///
    /// # Returns
    /// Tuple of (network address, prefix length)
    ///
    /// # Errors
    /// Returns error if format is invalid.
    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> {
        let parts: Vec<&str> = self.ip_range.split('/').collect();
        if parts.len() != 2 {
            return Err(ServerError::config_invalid(
                "tunnel.ip_range",
                "invalid CIDR format",
            ));
        }

        let network: Ipv4Addr = parts[0].parse().map_err(|_| {
            ServerError::config_invalid("tunnel.ip_range", "invalid network address")
        })?;

        let prefix: u8 = parts[1].parse().map_err(|_| {
            ServerError::config_invalid("tunnel.ip_range", "invalid prefix length")
        })?;

        if prefix > 32 {
            return Err(ServerError::config_invalid(
                "tunnel.ip_range",
                "prefix length cannot exceed 32",
            ));
        }

        Ok((network, prefix))
    }
}

impl Default for TunnelConfig {
    fn default() -> Self {
        Self {
            device_name: default_device_name(),
            ip_range: default_ip_range(),
            gateway_ip: default_gateway_ip(),
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

    /// Whether to generate key if missing.
    #[serde(default = "default_auto_generate")]
    pub auto_generate: bool,
}

fn default_key_file() -> String {
    "/etc/aeronyx/server_key.json".to_string()
}

fn default_auto_generate() -> bool {
    true
}

impl Default for ServerKeyConfig {
    fn default() -> Self {
        Self {
            key_file: default_key_file(),
            auto_generate: default_auto_generate(),
        }
    }
}

// ============================================
// LimitsConfig
// ============================================

/// Resource limits configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    /// Maximum concurrent sessions.
    #[serde(default = "default_max_sessions")]
    pub max_sessions: usize,

    /// Session timeout in seconds.
    #[serde(default = "default_session_timeout")]
    pub session_timeout_secs: u64,

    /// Handshake timeout in seconds.
    #[serde(default = "default_handshake_timeout")]
    pub handshake_timeout_secs: u64,
}

fn default_max_sessions() -> usize {
    1000
}

fn default_session_timeout() -> u64 {
    300
}

fn default_handshake_timeout() -> u64 {
    10
}

impl LimitsConfig {
    fn validate(&self) -> Result<()> {
        if self.max_sessions == 0 {
            return Err(ServerError::config_invalid(
                "limits.max_sessions",
                "must be greater than 0",
            ));
        }

        if self.session_timeout_secs == 0 {
            return Err(ServerError::config_invalid(
                "limits.session_timeout_secs",
                "must be greater than 0",
            ));
        }

        Ok(())
    }
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_sessions: default_max_sessions(),
            session_timeout_secs: default_session_timeout(),
            handshake_timeout_secs: default_handshake_timeout(),
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

    /// Output format (text, json).
    #[serde(default = "default_log_format")]
    pub format: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "text".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
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
    fn test_config_from_toml() {
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:12345"
            
            [tunnel]
            device_name = "test0"
            ip_range = "10.0.0.0/24"
            gateway_ip = "10.0.0.1"
            mtu = 1400
            
            [limits]
            max_sessions = 500
            session_timeout_secs = 600
        "#;

        let config = ServerConfig::from_str(toml).unwrap();
        assert_eq!(config.network.listen_addr.port(), 12345);
        assert_eq!(config.tunnel.device_name, "test0");
        assert_eq!(config.limits.max_sessions, 500);
    }

    #[test]
    fn test_invalid_config() {
        // Invalid port
        let toml = r#"
            [network]
            listen_addr = "0.0.0.0:0"
        "#;
        assert!(ServerConfig::from_str(toml).is_err());

        // Invalid MTU
        let toml = r#"
            [tunnel]
            mtu = 100
        "#;
        assert!(ServerConfig::from_str(toml).is_err());

        // Invalid max_sessions
        let toml = r#"
            [limits]
            max_sessions = 0
        "#;
        assert!(ServerConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_parse_ip_range() {
        let config = TunnelConfig::default();
        let (network, prefix) = config.parse_ip_range().unwrap();
        
        assert_eq!(network, Ipv4Addr::new(100, 64, 0, 0));
        assert_eq!(prefix, 24);
    }

    #[test]
    fn test_config_to_toml() {
        let config = ServerConfig::default();
        let toml = config.to_toml();
        
        assert!(toml.contains("[network]"));
        assert!(toml.contains("[tunnel]"));
        assert!(toml.contains("[limits]"));
    }
}
