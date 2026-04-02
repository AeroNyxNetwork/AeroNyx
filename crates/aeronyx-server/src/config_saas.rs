// ============================================
// File: crates/aeronyx-server/src/config_saas.rs
// ============================================
//! # SaaS Multi-Tenant Configuration
//!
//! ## Creation Reason
//! Extracted from config.rs as part of the v1.1.0-ChatRelay refactor split.
//! `SaasConfig` has its own lifecycle, defaults, and validation rules that
//! are entirely independent of other MemChain subsystems.
//!
//! ## Modification Reason
//! v1.0.0-MultiTenant — 🌟 Initial implementation (was inline in config.rs).
//! v1.1.0-ChatRelay   — 🌟 Extracted to config_saas.rs; zero logic changes.
//!
//! ## Main Functionality
//! - `SaasConfig` — StoragePool limits + MinerScheduler per-user quotas
//! - All validation is performed via `SaasConfig::validate()`
//! - `MemChainConfig::validate()` calls this only when `mode = "saas"`
//!
//! ## Dependencies
//! - `config_memchain.rs` — consumes `SaasConfig` via the `saas` field
//! - `server.rs`          — uses `effective_saas_config()` to init StoragePool
//!                          and MinerScheduler
//!
//! ## Main Logical Flow
//! 1. TOML `[memchain.saas]` deserializes into `SaasConfig`
//! 2. If section is absent in Saas mode, `SaasConfig::default()` is used at
//!    runtime (validated in `MemChainConfig::validate()` with a warn)
//! 3. `validate()` rejects zero-value fields that would break pool/scheduler
//!
//! ⚠️ Important Note for Next Developer:
//! - `data_root` must be writable by the server process.
//! - `pool_max_connections` limits simultaneously open SQLite files.
//!   Each connection uses ~1–2 MB RAM. 100 connections ≈ 100–200 MB.
//! - `pool_idle_timeout_secs`: lower = less RAM, more DB reopen cost.
//! - `miner_max_rounds_per_hour` is per-user. With 10 owners/tick × 6 rounds,
//!   the scheduler sustains ~60 cognitive cycles/hour across all users.
//! - `PartialEq` is required for `assert_eq!` in downstream tests.
//!
//! ## Last Modified
//! v1.0.0-MultiTenant — Added for SaaS multi-tenant mode.
//! v1.1.0-ChatRelay   — Extracted to own file.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Result, ServerError};

// ============================================
// SaasConfig
// ============================================

/// SaaS mode infrastructure configuration.
///
/// Controls the StoragePool connection limits, idle eviction,
/// and MinerScheduler per-user quotas.
///
/// All fields have safe defaults — a minimal config only needs `data_root`.
///
/// ## Configuration Example
/// ```toml
/// [memchain.saas]
/// data_root = "data"
/// pool_max_connections = 100
/// pool_idle_timeout_secs = 1800
/// miner_max_owners_per_tick = 10
/// miner_max_rounds_per_hour = 6
/// ```
// BUG FIX (v1.0.0-MT): Added PartialEq to enable structural equality
// assertions in tests. Without this, downstream test authors cannot use
// assert_eq! on SaasConfig instances.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaasConfig {
    /// Root directory for SaaS data files.
    ///
    /// Contains:
    /// - `system.db`    — global metadata (volume assignments, LLM usage, auth events)
    /// - `volumes.toml` — volume configuration (auto-generated if missing)
    /// - `volumes/vol-*/` — per-user SQLite DB files
    ///
    /// Created automatically on first SaaS startup.
    /// Default: `"data"` (relative to working directory).
    #[serde(default = "default_saas_data_root")]
    pub data_root: PathBuf,

    /// Maximum number of simultaneously open MemoryStorage connections.
    ///
    /// When the pool reaches this limit, the oldest idle connection is
    /// evicted before opening a new one.
    /// Default: 100.
    #[serde(default = "default_pool_max_connections")]
    pub pool_max_connections: usize,

    /// Seconds of inactivity before a connection is evicted from the pool.
    ///
    /// Eviction timer runs every 5 minutes (POOL_EVICTION_INTERVAL_SECS).
    /// Default: 1800 (30 minutes).
    #[serde(default = "default_pool_idle_timeout_secs")]
    pub pool_idle_timeout_secs: u64,

    /// Maximum number of users the MinerScheduler processes per 60-second tick.
    ///
    /// Users are selected by most-recently-active order.
    /// Default: 10.
    #[serde(default = "default_miner_max_owners_per_tick")]
    pub miner_max_owners_per_tick: usize,

    /// Maximum Miner cognitive cycles per user per hour.
    ///
    /// Prevents runaway LLM API costs for highly active users.
    /// Quota resets at the start of each hour.
    /// Default: 6 (one cycle every ~10 minutes).
    #[serde(default = "default_miner_max_rounds_per_hour")]
    pub miner_max_rounds_per_hour: usize,
}

// ── Default functions ──

fn default_saas_data_root() -> PathBuf { PathBuf::from("data") }
fn default_pool_max_connections() -> usize { 100 }
fn default_pool_idle_timeout_secs() -> u64 { 1800 }
fn default_miner_max_owners_per_tick() -> usize { 10 }
fn default_miner_max_rounds_per_hour() -> usize { 6 }

impl Default for SaasConfig {
    fn default() -> Self {
        Self {
            data_root: default_saas_data_root(),
            pool_max_connections: default_pool_max_connections(),
            pool_idle_timeout_secs: default_pool_idle_timeout_secs(),
            miner_max_owners_per_tick: default_miner_max_owners_per_tick(),
            miner_max_rounds_per_hour: default_miner_max_rounds_per_hour(),
        }
    }
}

impl SaasConfig {
    /// Validates all SaasConfig fields.
    ///
    /// Only called from `MemChainConfig::validate()` when `mode = "saas"`.
    /// All checks are "must be > 0" — zero values break pool/scheduler init.
    pub fn validate(&self) -> Result<()> {
        if self.pool_max_connections == 0 {
            return Err(ServerError::config_invalid(
                "memchain.saas.pool_max_connections",
                "must be > 0",
            ));
        }
        if self.pool_idle_timeout_secs == 0 {
            return Err(ServerError::config_invalid(
                "memchain.saas.pool_idle_timeout_secs",
                "must be > 0",
            ));
        }
        if self.miner_max_owners_per_tick == 0 {
            return Err(ServerError::config_invalid(
                "memchain.saas.miner_max_owners_per_tick",
                "must be > 0",
            ));
        }
        if self.miner_max_rounds_per_hour == 0 {
            return Err(ServerError::config_invalid(
                "memchain.saas.miner_max_rounds_per_hour",
                "must be > 0",
            ));
        }
        Ok(())
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saas_config_defaults() {
        let sc = SaasConfig::default();
        assert_eq!(sc.data_root, PathBuf::from("data"));
        assert_eq!(sc.pool_max_connections, 100);
        assert_eq!(sc.pool_idle_timeout_secs, 1800);
        assert_eq!(sc.miner_max_owners_per_tick, 10);
        assert_eq!(sc.miner_max_rounds_per_hour, 6);
    }

    #[test]
    fn test_saas_config_default_valid() {
        assert!(SaasConfig::default().validate().is_ok());
    }

    #[test]
    fn test_saas_pool_max_zero_rejected() {
        let sc = SaasConfig { pool_max_connections: 0, ..Default::default() };
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_saas_pool_idle_timeout_zero_rejected() {
        let sc = SaasConfig { pool_idle_timeout_secs: 0, ..Default::default() };
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_saas_owners_per_tick_zero_rejected() {
        let sc = SaasConfig { miner_max_owners_per_tick: 0, ..Default::default() };
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_saas_rounds_per_hour_zero_rejected() {
        let sc = SaasConfig { miner_max_rounds_per_hour: 0, ..Default::default() };
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_saas_partial_eq() {
        let a = SaasConfig::default();
        let b = SaasConfig::default();
        assert_eq!(a, b);
        let c = SaasConfig { pool_max_connections: 50, ..Default::default() };
        assert_ne!(a, c);
    }
}
