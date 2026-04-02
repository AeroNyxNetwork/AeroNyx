// ============================================
// File: crates/aeronyx-server/src/config_infra.rs
// ============================================
//! # Infrastructure Configuration
//!
//! ## Creation Reason
//! Extracted from config.rs as part of the v1.1.0-ChatRelay refactor split.
//! config.rs grew beyond 1000 lines; pure infrastructure configs
//! (network, VPN, TUN, key, limits, logging) have no dependency on
//! MemChain subsystems and are cleanly separable.
//!
//! ## Modification Reason
//! v1.1.0-ChatRelay — 🌟 Split from config.rs (no logic changes).
//!
//! ## Main Functionality
//! - `NetworkConfig`  — listen address + public endpoint
//! - `VpnConfig`      — virtual IP range + gateway
//! - `TunConfig`      — TUN device name + MTU
//! - `ServerKeyConfig`— key file path
//! - `LimitsConfig`   — max connections + session timeout
//! - `LoggingConfig`  — log level
//!
//! ## Dependencies
//! - Consumed by `ServerConfig` in config.rs (no MemChain awareness needed)
//! - `ManagementConfig` remains in management.rs (owned by that subsystem)
//!
//! ## Main Logical Flow
//! 1. Each struct derives `Serialize` / `Deserialize` with serde defaults
//! 2. `validate()` on each struct is called by `ServerConfig::validate()`
//! 3. No cross-struct dependencies within this file
//!
//! ⚠️ Important Note for Next Developer:
//! - Do NOT add MemChain-aware logic here; this file must stay infra-only.
//! - All serde defaults must remain for backward-compatible TOML loading.
//! - `TunConfig::device_name` max length 15 is a Linux kernel limit (IFNAMSIZ-1).
//! - `VpnConfig::parse_ip_range()` is called by `ServerConfig::parse_ip_range()`
//!   and must remain pub.
//!
//! ## Last Modified
//! v1.1.0-ChatRelay — Extracted from config.rs; zero logic changes.

use std::net::{Ipv4Addr, SocketAddr};

use serde::{Deserialize, Serialize};

use crate::error::{Result, ServerError};

// ============================================
// NetworkConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: SocketAddr,
    #[serde(default)]
    pub public_endpoint: Option<String>,
}

fn default_listen_addr() -> SocketAddr { "0.0.0.0:51820".parse().unwrap() }

impl NetworkConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.listen_addr.port() == 0 {
            return Err(ServerError::config_invalid("network.listen_addr", "port cannot be 0"));
        }
        Ok(())
    }

    /// Parses the host portion of `public_endpoint` as an `Ipv4Addr`.
    pub fn public_ip(&self) -> Option<Ipv4Addr> {
        self.public_endpoint.as_ref().and_then(|ep| {
            ep.split(':').next().and_then(|ip| ip.parse().ok())
        })
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self { listen_addr: default_listen_addr(), public_endpoint: None }
    }
}

// ============================================
// VpnConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpnConfig {
    #[serde(default = "default_ip_range")]
    pub virtual_ip_range: String,
    #[serde(default = "default_gateway_ip")]
    pub gateway_ip: Ipv4Addr,
}

fn default_ip_range() -> String { "100.64.0.0/24".into() }
fn default_gateway_ip() -> Ipv4Addr { Ipv4Addr::new(100, 64, 0, 1) }

impl VpnConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.virtual_ip_range.contains('/') {
            return Err(ServerError::config_invalid("vpn.virtual_ip_range", "must be CIDR"));
        }
        Ok(())
    }

    /// Parses `virtual_ip_range` into `(network_addr, prefix_len)`.
    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> {
        let parts: Vec<&str> = self.virtual_ip_range.split('/').collect();
        if parts.len() != 2 {
            return Err(ServerError::config_invalid("vpn.virtual_ip_range", "invalid CIDR"));
        }
        let network: Ipv4Addr = parts[0].parse()
            .map_err(|_| ServerError::config_invalid("vpn.virtual_ip_range", "invalid address"))?;
        let prefix: u8 = parts[1].parse()
            .map_err(|_| ServerError::config_invalid("vpn.virtual_ip_range", "invalid prefix"))?;
        if prefix > 32 {
            return Err(ServerError::config_invalid("vpn.virtual_ip_range", "prefix > 32"));
        }
        Ok((network, prefix))
    }
}

impl Default for VpnConfig {
    fn default() -> Self {
        Self { virtual_ip_range: default_ip_range(), gateway_ip: default_gateway_ip() }
    }
}

// ============================================
// TunConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunConfig {
    #[serde(default = "default_device_name")]
    pub device_name: String,
    #[serde(default = "default_mtu")]
    pub mtu: u16,
}

fn default_device_name() -> String { "aeronyx0".into() }
fn default_mtu() -> u16 { 1420 }

impl TunConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.device_name.is_empty() {
            return Err(ServerError::config_invalid("tun.device_name", "empty"));
        }
        // Linux IFNAMSIZ = 16; interface name must be <= 15 chars (null-terminated)
        if self.device_name.len() > 15 {
            return Err(ServerError::config_invalid("tun.device_name", "> 15 chars"));
        }
        if self.mtu < 576 {
            return Err(ServerError::config_invalid("tun.mtu", "< 576"));
        }
        if self.mtu > 9000 {
            return Err(ServerError::config_invalid("tun.mtu", "> 9000"));
        }
        Ok(())
    }
}

impl Default for TunConfig {
    fn default() -> Self { Self { device_name: default_device_name(), mtu: default_mtu() } }
}

// ============================================
// ServerKeyConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerKeyConfig {
    #[serde(default = "default_key_file")]
    pub key_file: String,
}

fn default_key_file() -> String { "/etc/aeronyx/server_key.json".into() }

impl Default for ServerKeyConfig {
    fn default() -> Self { Self { key_file: default_key_file() } }
}

// ============================================
// LimitsConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_session_timeout")]
    pub session_timeout: u64,
}

fn default_max_connections() -> usize { 1000 }
fn default_session_timeout() -> u64 { 300 }

impl LimitsConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_connections == 0 {
            return Err(ServerError::config_invalid("limits.max_connections", "= 0"));
        }
        if self.session_timeout == 0 {
            return Err(ServerError::config_invalid("limits.session_timeout", "= 0"));
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
}

fn default_log_level() -> String { "info".into() }

impl Default for LoggingConfig {
    fn default() -> Self { Self { level: default_log_level() } }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── NetworkConfig ──

    #[test]
    fn test_network_default_valid() {
        assert!(NetworkConfig::default().validate().is_ok());
    }

    #[test]
    fn test_network_port_zero_rejected() {
        let nc = NetworkConfig {
            listen_addr: "0.0.0.0:0".parse().unwrap(),
            ..Default::default()
        };
        assert!(nc.validate().is_err());
    }

    #[test]
    fn test_network_public_ip_parsed() {
        let nc = NetworkConfig {
            public_endpoint: Some("1.2.3.4:51820".into()),
            ..Default::default()
        };
        assert_eq!(nc.public_ip(), Some(Ipv4Addr::new(1, 2, 3, 4)));
    }

    #[test]
    fn test_network_public_ip_none_when_unset() {
        assert!(NetworkConfig::default().public_ip().is_none());
    }

    // ── VpnConfig ──

    #[test]
    fn test_vpn_default_valid() {
        assert!(VpnConfig::default().validate().is_ok());
    }

    #[test]
    fn test_vpn_missing_cidr_rejected() {
        let vc = VpnConfig { virtual_ip_range: "10.0.0.0".into(), ..Default::default() };
        assert!(vc.validate().is_err());
    }

    #[test]
    fn test_vpn_parse_ip_range_ok() {
        let vc = VpnConfig::default();
        let (net, prefix) = vc.parse_ip_range().unwrap();
        assert_eq!(net, Ipv4Addr::new(100, 64, 0, 0));
        assert_eq!(prefix, 24);
    }

    #[test]
    fn test_vpn_prefix_over_32_rejected() {
        let vc = VpnConfig { virtual_ip_range: "10.0.0.0/33".into(), ..Default::default() };
        assert!(vc.parse_ip_range().is_err());
    }

    // ── TunConfig ──

    #[test]
    fn test_tun_default_valid() {
        assert!(TunConfig::default().validate().is_ok());
    }

    #[test]
    fn test_tun_empty_name_rejected() {
        let tc = TunConfig { device_name: String::new(), ..Default::default() };
        assert!(tc.validate().is_err());
    }

    #[test]
    fn test_tun_name_too_long_rejected() {
        let tc = TunConfig { device_name: "a".repeat(16), ..Default::default() };
        assert!(tc.validate().is_err());
    }

    #[test]
    fn test_tun_mtu_too_small_rejected() {
        let tc = TunConfig { mtu: 575, ..Default::default() };
        assert!(tc.validate().is_err());
    }

    #[test]
    fn test_tun_mtu_too_large_rejected() {
        let tc = TunConfig { mtu: 9001, ..Default::default() };
        assert!(tc.validate().is_err());
    }

    // ── LimitsConfig ──

    #[test]
    fn test_limits_default_valid() {
        assert!(LimitsConfig::default().validate().is_ok());
    }

    #[test]
    fn test_limits_zero_connections_rejected() {
        let lc = LimitsConfig { max_connections: 0, ..Default::default() };
        assert!(lc.validate().is_err());
    }

    #[test]
    fn test_limits_zero_timeout_rejected() {
        let lc = LimitsConfig { session_timeout: 0, ..Default::default() };
        assert!(lc.validate().is_err());
    }
}
