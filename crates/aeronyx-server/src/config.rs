// ============================================
// File: crates/aeronyx-server/src/config.rs
// ============================================
//! # Server Configuration
//!
//! ## Creation Reason
//! Central configuration for all AeroNyx server subsystems, loaded from TOML.
//!
//! ## Modification Reason
//! v2.1+MVF+Auth — Added `api_secret` field to MemChainConfig for MPI Bearer token auth.
//! v2.3.0+RemoteStorage — 🌟 Added `allow_remote_storage` and `max_remote_owners` fields
//!   to MemChainConfig for Phase 1 remote MPI Gateway support.
//!   Also added validation for `mvf_alpha`, `miner_interval_secs`, `embed_dim`,
//!   and `embed_max_tokens` (bug fixes from code review).
//!
//! ## Main Functionality
//! - ServerConfig: top-level config with network, vpn, tun, limits, logging, management, memchain
//! - MemChainConfig: memory system config with MVF parameters, DB paths, API auth, and remote storage
//! - Validation for all config sections
//!
//! ## Dependencies
//! - Used by server.rs to initialize all subsystems
//! - MemChainConfig consumed by MPI router (api/mpi.rs) for auth middleware + remote storage checks
//! - MemChainConfig consumed by storage.rs for DB path
//!
//! ## Main Logical Flow
//! 1. Load TOML file → deserialize into ServerConfig
//! 2. Validate all sections (network, vpn, tun, limits, management, memchain)
//! 3. Return validated config for server initialization
//!
//! ⚠️ Important Note for Next Developer:
//! - api_secret validation: must be >= 16 chars when set (prevents weak secrets)
//! - Empty/None api_secret = open access (backward compatible)
//! - All MemChain config fields have serde defaults for backward compatibility
//! - allow_remote_storage defaults to false — existing nodes are NOT affected
//! - When allow_remote_storage is true, Ed25519 signature auth is used for remote requests
//!   (parallel to Bearer token auth for local requests)
//! - max_remote_owners caps how many distinct remote users this node will serve
//! - mvf_alpha must be in [0.0, 1.0] — validated since v2.3.0
//! - miner_interval_secs must be > 0 — validated since v2.3.0
//! - embed_dim and embed_max_tokens must be > 0 when memchain is enabled — validated since v2.3.0
//!
//! ## Last Modified
//! v2.1.0 - Added db_path, compaction_threshold, mvf_alpha, mvf_enabled,
//!           cold_start_threshold, cold_start_until, rawlog_batch_threshold
//! v2.1.0+MVF - MVF fields added
//! v2.1.0+MVF+Auth - 🌟 Added api_secret for MPI Bearer token authentication
//! v2.3.0+RemoteStorage - 🌟 Added allow_remote_storage, max_remote_owners for Phase 1
//!   🐛 Added validation for mvf_alpha range, miner_interval_secs > 0,
//!      embed_dim > 0, embed_max_tokens > 0

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
    /// Path to the local embedding model directory.
    ///
    /// The directory must contain `model.onnx` and `tokenizer.json` for
    /// MiniLM-L6-v2 (or compatible BERT-family model).
    /// If the path does not exist or files are missing, embedding is disabled
    /// and the system falls back to OpenClaw Gateway `/v1/embeddings`.
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain]
    /// embed_model_path = "models/minilm-l6-v2"
    /// ```
    #[serde(default = "default_embed_model_path")]
    pub embed_model_path: String,

    /// Expected embedding dimension from the local model.
    /// MiniLM-L6-v2 produces 384-dim vectors. Used for validation only.
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,

    /// Maximum token sequence length for embedding inference.
    /// Inputs longer than this are truncated. MiniLM supports up to 512
    /// but 128 is optimal for MemChain's short content (preferences, facts).
    /// 512 would quadruple inference time with no quality benefit.
    #[serde(default = "default_embed_max_tokens")]
    pub embed_max_tokens: usize,

    /// MPI Bearer token secret for API authentication.
    ///
    /// When set (non-empty, >= 16 chars), all MPI endpoints require:
    ///   `Authorization: Bearer <api_secret>`
    /// When None or empty, MPI is open (backward compatible, loopback-only mitigates).
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain]
    /// api_secret = "your-secret-at-least-16-chars"
    /// ```
    #[serde(default)]
    pub api_secret: Option<String>,

    /// ## Phase 1: Remote Storage Configuration (v2.3.0)
    ///
    /// When `true`, this node accepts MPI requests from remote users
    /// authenticated via Ed25519 signature (not just local Bearer token).
    ///
    /// Remote requests use the signer's public key as the `owner` field,
    /// isolating their data from the node operator's own data.
    ///
    /// When `false` (default), only local requests (Bearer token auth) are
    /// accepted — existing behavior is completely unchanged.
    ///
    /// ## Security Model
    /// - Remote users cannot read/modify each other's data (owner isolation)
    /// - Remote users cannot access the node operator's data
    /// - Node operator can see encrypted content but cannot decrypt it
    ///   (record_key derived from user's private key via HKDF)
    /// - Embeddings are stored in plaintext (required for vector search)
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain]
    /// allow_remote_storage = true
    /// max_remote_owners = 100
    /// ```
    #[serde(default)]
    pub allow_remote_storage: bool,

    /// Maximum number of distinct remote owners this node will serve.
    ///
    /// Prevents a single node from being overwhelmed by too many remote users.
    /// Once the limit is reached, new remote users receive 503 Service Unavailable
    /// and should be reassigned to another node by CMS.
    ///
    /// Only effective when `allow_remote_storage = true`.
    /// Set to 0 for unlimited (not recommended for resource-constrained nodes).
    ///
    /// ## Configuration Example
    /// ```toml
    /// [memchain]
    /// max_remote_owners = 100
    /// ```
    #[serde(default = "default_max_remote_owners")]
    pub max_remote_owners: usize,
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
fn default_embed_model_path() -> String { "models/minilm-l6-v2".into() }
fn default_embed_dim() -> usize { 384 }
fn default_embed_max_tokens() -> usize { 128 }
fn default_max_remote_owners() -> usize { 100 }

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

            // Validate api_secret: if set, must be >= 16 characters
            // This prevents weak secrets that could be easily brute-forced.
            // None or empty string = open access (backward compatible).
            if let Some(ref secret) = self.api_secret {
                if !secret.is_empty() && secret.len() < 16 {
                    return Err(ServerError::config_invalid(
                        "memchain.api_secret",
                        format!("must be at least 16 characters, got {}", secret.len()),
                    ));
                }
            }

            // 🐛 v2.3.0: Validate mvf_alpha range [0.0, 1.0]
            // mvf_alpha is used in mvf::fuse_scores(vm, v_old, alpha) as a blend factor.
            // Values outside [0.0, 1.0] produce nonsensical recall scores.
            if self.mvf_alpha < 0.0 || self.mvf_alpha > 1.0 {
                return Err(ServerError::config_invalid(
                    "memchain.mvf_alpha",
                    format!("must be in [0.0, 1.0], got {}", self.mvf_alpha),
                ));
            }

            // 🐛 v2.3.0: Validate miner_interval_secs > 0
            // A zero interval would cause the Smart Miner timer to fire continuously,
            // consuming CPU with no benefit (no time for new data to accumulate).
            if self.miner_interval_secs == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.miner_interval_secs",
                    "must be > 0 (seconds between miner runs)",
                ));
            }

            // 🐛 v2.3.0: Validate embed_dim > 0
            // A zero-dimension embedding is meaningless and would cause
            // vector index operations to fail or produce empty results.
            if self.embed_dim == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.embed_dim",
                    "must be > 0",
                ));
            }

            // 🐛 v2.3.0: Validate embed_max_tokens > 0
            // Zero max tokens would truncate all input to nothing,
            // producing empty embeddings.
            if self.embed_max_tokens == 0 {
                return Err(ServerError::config_invalid(
                    "memchain.embed_max_tokens",
                    "must be > 0",
                ));
            }

            // v2.3.0: Validate remote storage configuration
            if self.allow_remote_storage {
                info!(
                    "[MEMCHAIN] Remote storage enabled (max_remote_owners: {})",
                    if self.max_remote_owners == 0 { "unlimited".to_string() }
                    else { self.max_remote_owners.to_string() }
                );

                // Warn if remote storage is enabled but no api_secret is set.
                // Remote auth uses Ed25519 signatures (not Bearer token), but having
                // api_secret protects local MPI from unauthorized access via loopback.
                // Without it, anyone on the same machine can call MPI without auth.
                if self.effective_api_secret().is_none() {
                    tracing::warn!(
                        "[MEMCHAIN] allow_remote_storage=true but api_secret is not set. \
                         Local MPI endpoints are unprotected. Consider setting api_secret \
                         to prevent unauthorized local access."
                    );
                }
            }
        }
        Ok(())
    }

    #[must_use] pub fn is_enabled(&self) -> bool { self.mode != MemChainMode::Off }
    #[must_use] pub fn is_p2p(&self) -> bool { self.mode == MemChainMode::P2p }

    /// Returns the effective API secret, or None if auth is disabled.
    ///
    /// Empty string is treated as None (no auth) for backward compatibility.
    #[must_use]
    pub fn effective_api_secret(&self) -> Option<&str> {
        self.api_secret.as_deref().filter(|s| !s.is_empty())
    }

    /// Check whether remote storage is enabled and accepting new owners.
    ///
    /// This is a config-level check only. The actual owner count check
    /// happens in mpi.rs middleware where the storage layer can be queried.
    ///
    /// ## Returns
    /// - `true` if `allow_remote_storage` is enabled
    /// - `false` otherwise
    #[must_use]
    pub fn is_remote_storage_enabled(&self) -> bool {
        self.allow_remote_storage
    }

    #[must_use]
    pub fn is_origin_trusted(&self, origin_hex: &str, server_pubkey_hex: &str) -> bool {
        // If origin is the server itself, always trusted
        if origin_hex == server_pubkey_hex { return true; }
        // If trusted_agents list is empty, all origins are trusted (backward compatible).
        // ⚠️ Note for Phase 1: this applies to local/P2P trust only.
        // Remote storage auth uses Ed25519 signatures, not this trust check.
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
            embed_model_path: default_embed_model_path(),
            embed_dim: default_embed_dim(),
            embed_max_tokens: default_embed_max_tokens(),
            api_secret: None,
            allow_remote_storage: false,
            max_remote_owners: default_max_remote_owners(),
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
        assert!(config.memchain.api_secret.is_none());
        // v2.3.0: remote storage defaults
        assert!(!config.memchain.allow_remote_storage);
        assert_eq!(config.memchain.max_remote_owners, 100);
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
        assert_eq!(mc.embed_model_path, "models/minilm-l6-v2");
        assert_eq!(mc.embed_dim, 384);
        assert_eq!(mc.embed_max_tokens, 128);
        assert!(mc.api_secret.is_none());
        assert!(mc.effective_api_secret().is_none());
        // v2.3.0
        assert!(!mc.allow_remote_storage);
        assert!(!mc.is_remote_storage_enabled());
        assert_eq!(mc.max_remote_owners, 100);
    }

    #[test]
    fn test_api_secret_validation_too_short() {
        let mc = MemChainConfig {
            api_secret: Some("short".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_api_secret_validation_valid() {
        let mc = MemChainConfig {
            api_secret: Some("this-is-a-long-secret-key-1234".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_api_secret_empty_is_none() {
        let mc = MemChainConfig {
            api_secret: Some(String::new()),
            ..Default::default()
        };
        // Empty string passes validation (treated as disabled)
        assert!(mc.validate().is_ok());
        // effective_api_secret returns None for empty string
        assert!(mc.effective_api_secret().is_none());
    }

    #[test]
    fn test_api_secret_none_is_open() {
        let mc = MemChainConfig {
            api_secret: None,
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.effective_api_secret().is_none());
    }

    #[test]
    fn test_effective_api_secret() {
        let mc = MemChainConfig {
            api_secret: Some("my-secure-secret-token-here".into()),
            ..Default::default()
        };
        assert_eq!(mc.effective_api_secret(), Some("my-secure-secret-token-here"));
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

    // ========================================
    // v2.3.0: Remote Storage Tests
    // ========================================

    #[test]
    fn test_remote_storage_disabled_by_default() {
        let mc = MemChainConfig::default();
        assert!(!mc.allow_remote_storage);
        assert!(!mc.is_remote_storage_enabled());
    }

    #[test]
    fn test_remote_storage_enabled() {
        let mc = MemChainConfig {
            allow_remote_storage: true,
            max_remote_owners: 50,
            ..Default::default()
        };
        assert!(mc.is_remote_storage_enabled());
        assert_eq!(mc.max_remote_owners, 50);
        // Validation passes (warn about missing api_secret is just a log warning)
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_remote_storage_with_api_secret() {
        let mc = MemChainConfig {
            allow_remote_storage: true,
            max_remote_owners: 200,
            api_secret: Some("my-secure-remote-secret".into()),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
        assert!(mc.is_remote_storage_enabled());
    }

    #[test]
    fn test_remote_storage_unlimited_owners() {
        let mc = MemChainConfig {
            allow_remote_storage: true,
            max_remote_owners: 0, // unlimited
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }

    #[test]
    fn test_remote_storage_toml_parsing() {
        let toml_str = r#"
[memchain]
mode = "local"
allow_remote_storage = true
max_remote_owners = 75
api_secret = "a-very-secure-secret-key"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(config.memchain.allow_remote_storage);
        assert_eq!(config.memchain.max_remote_owners, 75);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_remote_storage_toml_backward_compat() {
        // Old TOML without remote storage fields → defaults to false
        let toml_str = r#"
[memchain]
mode = "local"
db_path = "memchain.db"
"#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.memchain.allow_remote_storage);
        assert_eq!(config.memchain.max_remote_owners, 100);
        assert!(config.validate().is_ok());
    }

    // ========================================
    // v2.3.0: Bug Fix Validation Tests
    // ========================================

    #[test]
    fn test_mvf_alpha_out_of_range_rejected() {
        let mc = MemChainConfig {
            mvf_alpha: 1.5,
            ..Default::default()
        };
        assert!(mc.validate().is_err());

        let mc2 = MemChainConfig {
            mvf_alpha: -0.1,
            ..Default::default()
        };
        assert!(mc2.validate().is_err());
    }

    #[test]
    fn test_mvf_alpha_boundary_values() {
        let mc_zero = MemChainConfig { mvf_alpha: 0.0, ..Default::default() };
        assert!(mc_zero.validate().is_ok());

        let mc_one = MemChainConfig { mvf_alpha: 1.0, ..Default::default() };
        assert!(mc_one.validate().is_ok());
    }

    #[test]
    fn test_miner_interval_zero_rejected() {
        let mc = MemChainConfig {
            miner_interval_secs: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_embed_dim_zero_rejected() {
        let mc = MemChainConfig {
            embed_dim: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_embed_max_tokens_zero_rejected() {
        let mc = MemChainConfig {
            embed_max_tokens: 0,
            ..Default::default()
        };
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_memchain_off_skips_all_validation() {
        // When mode = Off, no validation is performed (even invalid values pass)
        let mc = MemChainConfig {
            mode: MemChainMode::Off,
            mvf_alpha: 999.0,
            miner_interval_secs: 0,
            embed_dim: 0,
            embed_max_tokens: 0,
            db_path: String::new(),
            ..Default::default()
        };
        assert!(mc.validate().is_ok());
    }
}
