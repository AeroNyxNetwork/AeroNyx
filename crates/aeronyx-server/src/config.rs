// ============================================
// File: crates/aeronyx-server/src/config.rs
// ============================================
//! # Server Configuration
//!
//! v2.1+MVF — Added db_path, compaction_threshold, mvf_alpha, mvf_enabled,
//! cold_start_threshold, cold_start_until, rawlog_batch_threshold.

use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Result, ServerError};
use crate::management::ManagementConfig;

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
}

impl ServerConfig {
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading configuration from: {}", path.display());
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| ServerError::config_load(&path.display().to_string(), e.to_string()))?;
        config.validate()?;
        info!("Configuration loaded successfully");
        Ok(config)
    }

    pub fn from_str(content: &str) -> Result<Self> {
        let config: Self = toml::from_str(content)
            .map_err(|e| ServerError::config_load("<string>", e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.network.validate()?;
        self.vpn.validate()?;
        self.tun.validate()?;
        self.limits.validate()?;
        self.management.validate().map_err(|e| ServerError::config_invalid("management", e))?;
        self.memchain.validate()?;
        Ok(())
    }

    #[must_use]
    pub fn to_toml(&self) -> String { toml::to_string_pretty(self).unwrap_or_default() }
    pub fn listen_addr(&self) -> SocketAddr { self.network.listen_addr }
    pub fn device_name(&self) -> &str { &self.tun.device_name }
    pub fn ip_range(&self) -> &str { &self.vpn.virtual_ip_range }
    pub fn gateway_ip(&self) -> Ipv4Addr { self.vpn.gateway_ip }
    pub fn mtu(&self) -> u16 { self.tun.mtu }
    pub fn max_sessions(&self) -> usize { self.limits.max_connections }
    pub fn session_timeout_secs(&self) -> u64 { self.limits.session_timeout }
    pub fn parse_ip_range(&self) -> Result<(Ipv4Addr, u8)> { self.vpn.parse_ip_range() }
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
// MemChainMode
// ============================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemChainMode {
    Off,
    Local,
    P2p,
}

impl Default for MemChainMode {
    fn default() -> Self { Self::Local }
}

// ============================================
// MemChainConfig
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemChainConfig {
    #[serde(default)]
    pub mode: MemChainMode,
    #[serde(default = "default_memchain_api_addr")]
    pub api_listen_addr: SocketAddr,
    #[serde(default = "default_memchain_db_path")]
    pub db_path: String,
    #[serde(default = "default_memchain_aof_path")]
    pub aof_path: String,
    #[serde(default)]
    pub trusted_agents: Vec<String>,
    #[serde(default = "default_miner_interval")]
    pub miner_interval_secs: u64,
    #[serde(default = "default_compaction_threshold")]
    pub compaction_threshold: u64,
    #[serde(default = "default_mvf_alpha")]
    pub mvf_alpha: f32,
    #[serde(default)]
    pub mvf_enabled: bool,
    #[serde(default = "default_cold_start_threshold")]
    pub cold_start_threshold: usize,
    #[serde(default = "default_cold_start_until")]
    pub cold_start_until: usize,
    #[serde(default = "default_rawlog_batch_threshold")]
    pub rawlog_batch_threshold: usize,
}

fn default_memchain_api_addr() -> SocketAddr { "127.0.0.1:8421".parse().unwrap() }
fn default_memchain_db_path() -> String { "memchain.db".into() }
fn default_memchain_aof_path() -> String { ".memchain".into() }
fn default_miner_interval() -> u64 { 3600 }
fn default_compaction_threshold() -> u64 { 500 }
fn default_mvf_alpha() -> f32 { 0.5 }
fn default_cold_start_threshold() -> usize { 10 }
fn default_cold_start_until() -> usize { 200 }
fn default_rawlog_batch_threshold() -> usize { 100 }

impl MemChainConfig {
    pub fn validate(&self) -> Result<()> {
        if self.mode != MemChainMode::Off {
            if self.api_listen_addr.port() == 0 {
                return Err(ServerError::config_invalid("memchain.api_listen_addr", "port cannot be 0"));
            }
            if self.db_path.is_empty() {
                return Err(ServerError::config_invalid("memchain.db_path", "cannot be empty"));
            }
            if self.aof_path.is_empty() {
                return Err(ServerError::config_invalid("memchain.aof_path", "cannot be empty"));
            }
            if !self.api_listen_addr.ip().is_loopback() {
                tracing::warn!("[MEMCHAIN] API binding to non-loopback {}", self.api_listen_addr);
            }
            for (i, hex_key) in self.trusted_agents.iter().enumerate() {
                if hex_key.len() != 64 {
                    return Err(ServerError::config_invalid(
                        &format!("memchain.trusted_agents[{}]", i),
                        format!("expected 64 hex chars, got {}", hex_key.len()),
                    ));
                }
                if hex::decode(hex_key).is_err() {
                    return Err(ServerError::config_invalid(
                        &format!("memchain.trusted_agents[{}]", i),
                        format!("invalid hex: '{}'", hex_key),
                    ));
                }
            }
        }
        Ok(())
    }

    #[must_use] pub fn is_enabled(&self) -> bool { self.mode != MemChainMode::Off }
    #[must_use] pub fn is_p2p(&self) -> bool { self.mode == MemChainMode::P2p }

    #[must_use]
    pub fn is_origin_trusted(&self, origin_hex: &str, server_pubkey_hex: &str) -> bool {
        if origin_hex == server_pubkey_hex { return true; }
        if self.trusted_agents.is_empty() { return true; }
        self.trusted_agents.iter().any(|t| t == origin_hex)
    }
}

impl Default for MemChainConfig {
    fn default() -> Self {
        Self {
            mode: MemChainMode::default(),
            api_listen_addr: default_memchain_api_addr(),
            db_path: default_memchain_db_path(),
            aof_path: default_memchain_aof_path(),
            trusted_agents: Vec::new(),
            miner_interval_secs: default_miner_interval(),
            compaction_threshold: default_compaction_threshold(),
            mvf_alpha: default_mvf_alpha(),
            mvf_enabled: false,
            cold_start_threshold: default_cold_start_threshold(),
            cold_start_until: default_cold_start_until(),
            rawlog_batch_threshold: default_rawlog_batch_threshold(),
        }
    }
}

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
    fn validate(&self) -> Result<()> {
        if self.listen_addr.port() == 0 {
            return Err(ServerError::config_invalid("network.listen_addr", "port cannot be 0"));
        }
        Ok(())
    }

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
    fn validate(&self) -> Result<()> {
        if !self.virtual_ip_range.contains('/') {
            return Err(ServerError::config_invalid("vpn.virtual_ip_range", "must be CIDR"));
        }
        Ok(())
    }

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
    fn validate(&self) -> Result<()> {
        if self.device_name.is_empty() {
            return Err(ServerError::config_invalid("tun.device_name", "empty"));
        }
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
    fn validate(&self) -> Result<()> {
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
        Self { max_connections: default_max_connections(), session_timeout: default_session_timeout() }
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

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.memchain.mvf_alpha, 0.5);
        assert!(!config.memchain.mvf_enabled);
        assert_eq!(config.memchain.cold_start_threshold, 10);
    }

    #[test]
    fn test_memchain_defaults() {
        let mc = MemChainConfig::default();
        assert_eq!(mc.db_path, "memchain.db");
        assert_eq!(mc.compaction_threshold, 500);
        assert_eq!(mc.mvf_alpha, 0.5);
        assert!(!mc.mvf_enabled);
        assert_eq!(mc.cold_start_threshold, 10);
        assert_eq!(mc.cold_start_until, 200);
        assert_eq!(mc.rawlog_batch_threshold, 100);
    }

    #[test]
    fn test_is_origin_trusted() {
        let config = MemChainConfig {
            trusted_agents: vec![
                "aaaa0000000000000000000000000000000000000000000000000000000000aa".into(),
            ],
            ..Default::default()
        };
        let server = "bbbb0000000000000000000000000000000000000000000000000000000000bb";
        assert!(config.is_origin_trusted(server, server));
        assert!(config.is_origin_trusted(
            "aaaa0000000000000000000000000000000000000000000000000000000000aa", server));
        assert!(!config.is_origin_trusted(
            "cccc0000000000000000000000000000000000000000000000000000000000cc", server));
    }
}
