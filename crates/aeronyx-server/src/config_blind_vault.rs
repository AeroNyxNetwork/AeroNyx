// ============================================
// File: crates/aeronyx-server/src/config_blind_vault.rs
// ============================================
//! # Blind Vault Configuration
//!
//! ## Creation Reason
//! Configures the independent node-blind durable store used by encrypted
//! contact-vault and optional message-archive clients.
//!
//! ## Main Functionality
//! - Keeps the service disabled by default for backward compatibility.
//! - Bounds lease lifetime, object lifetime, per-lease count/bytes, page size,
//!   deletion tombstones, and maintenance work.
//! - Validates that one corrupted or malicious lease cannot consume unbounded
//!   storage or request work.
//!
//! ## Dependencies
//! - `config.rs`: owns this structure as a top-level subsystem.
//! - `services/blind_vault.rs`: consumes all runtime policy fields.
//! - `deploy/node/server.example.toml`: documents operator-safe defaults.
//!
//! ## Main Logical Flow
//! 1. TOML deserialization applies safe defaults.
//! 2. `ServerConfig::validate()` delegates to `validate()`.
//! 3. The server constructs `BlindVaultService` only when enabled.
//!
//! ## Important Note For The Next Developer
//! - Enabling storage must not automatically advertise a public capability.
//! - Lease admission remains a separate protocol policy; do not add account or
//!   wallet allowlists to this configuration.
//! - Keep byte/count limits finite even on official nodes.
//!
//! Last Modified: v1.0.0-BlindVaultService - Initial bounded configuration.
//! ============================================

use serde::{Deserialize, Serialize};

use crate::error::{Result, ServerError};

const MAX_LEASE_TTL_SECS_HARD: u64 = 365 * 24 * 60 * 60;
const MAX_OBJECTS_PER_LEASE_HARD: u64 = 1_000_000;
const MAX_BYTES_PER_LEASE_HARD: u64 = 64 * 1024 * 1024 * 1024;
const MAX_PULL_OBJECTS_HARD: usize = 256;
const LARGEST_PROTOCOL_OBJECT_BYTES: u64 = 256 * 1024;

/// Bounded, opt-in node policy for anonymous encrypted object storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindVaultConfig {
    /// Enables local Blind Vault persistence and client routes.
    #[serde(default)]
    pub enabled: bool,
    /// Dedicated SQLite path. It must not reuse MemChain, ChatRelay, or
    /// Directory Chain databases.
    #[serde(default = "BlindVaultConfig::default_db_path")]
    pub db_path: String,
    /// Maximum lifetime of one anonymous replica lease.
    #[serde(default = "BlindVaultConfig::default_max_lease_ttl_secs")]
    pub max_lease_ttl_secs: u64,
    /// Maximum lifetime accepted for one immutable ciphertext object.
    #[serde(default = "BlindVaultConfig::default_max_object_ttl_secs")]
    pub max_object_ttl_secs: u64,
    /// Maximum live objects retained by one lease.
    #[serde(default = "BlindVaultConfig::default_max_objects_per_lease")]
    pub max_objects_per_lease: u64,
    /// Maximum live ciphertext bytes retained by one lease.
    #[serde(default = "BlindVaultConfig::default_max_bytes_per_lease")]
    pub max_bytes_per_lease: u64,
    /// Maximum objects returned by one internal pull page.
    #[serde(default = "BlindVaultConfig::default_max_pull_objects")]
    pub max_pull_objects: usize,
    /// Retention for commitment-only deletion tombstones used by retries.
    #[serde(default = "BlindVaultConfig::default_tombstone_ttl_secs")]
    pub tombstone_ttl_secs: u64,
    /// Maximum client/server timestamp skew for signed mutations.
    #[serde(default = "BlindVaultConfig::default_mutation_clock_skew_secs")]
    pub mutation_clock_skew_secs: u64,
    /// Interval for bounded expiry cleanup.
    #[serde(default = "BlindVaultConfig::default_cleanup_interval_secs")]
    pub cleanup_interval_secs: u64,
}

impl BlindVaultConfig {
    fn default_db_path() -> String {
        "./data/blind_vault.db".to_string()
    }

    const fn default_max_lease_ttl_secs() -> u64 {
        90 * 24 * 60 * 60
    }

    const fn default_max_object_ttl_secs() -> u64 {
        30 * 24 * 60 * 60
    }

    const fn default_max_objects_per_lease() -> u64 {
        16_384
    }

    const fn default_max_bytes_per_lease() -> u64 {
        256 * 1024 * 1024
    }

    const fn default_max_pull_objects() -> usize {
        64
    }

    const fn default_tombstone_ttl_secs() -> u64 {
        7 * 24 * 60 * 60
    }

    const fn default_mutation_clock_skew_secs() -> u64 {
        5 * 60
    }

    const fn default_cleanup_interval_secs() -> u64 {
        5 * 60
    }

    /// Validates finite service limits when the subsystem is enabled.
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.db_path.trim().is_empty() {
            return Err(invalid("db_path", "must not be empty when enabled"));
        }
        if self.max_lease_ttl_secs == 0 || self.max_lease_ttl_secs > MAX_LEASE_TTL_SECS_HARD {
            return Err(invalid(
                "max_lease_ttl_secs",
                "must be between 1 second and 365 days",
            ));
        }
        if self.max_object_ttl_secs == 0 || self.max_object_ttl_secs > self.max_lease_ttl_secs {
            return Err(invalid(
                "max_object_ttl_secs",
                "must be non-zero and no greater than max_lease_ttl_secs",
            ));
        }
        if self.max_objects_per_lease == 0
            || self.max_objects_per_lease > MAX_OBJECTS_PER_LEASE_HARD
        {
            return Err(invalid(
                "max_objects_per_lease",
                "must be between 1 and 1000000",
            ));
        }
        if self.max_bytes_per_lease < LARGEST_PROTOCOL_OBJECT_BYTES
            || self.max_bytes_per_lease > MAX_BYTES_PER_LEASE_HARD
        {
            return Err(invalid(
                "max_bytes_per_lease",
                "must fit one 256 KiB object and not exceed 64 GiB",
            ));
        }
        if self.max_pull_objects == 0 || self.max_pull_objects > MAX_PULL_OBJECTS_HARD {
            return Err(invalid("max_pull_objects", "must be between 1 and 256"));
        }
        if self.tombstone_ttl_secs == 0 {
            return Err(invalid("tombstone_ttl_secs", "must be non-zero"));
        }
        if self.mutation_clock_skew_secs == 0 {
            return Err(invalid("mutation_clock_skew_secs", "must be non-zero"));
        }
        if self.cleanup_interval_secs == 0 {
            return Err(invalid("cleanup_interval_secs", "must be non-zero"));
        }
        Ok(())
    }

    /// Maximum lease lifetime in milliseconds.
    #[must_use]
    pub const fn max_lease_ttl_ms(&self) -> u64 {
        self.max_lease_ttl_secs.saturating_mul(1_000)
    }

    /// Maximum object lifetime in milliseconds.
    #[must_use]
    pub const fn max_object_ttl_ms(&self) -> u64 {
        self.max_object_ttl_secs.saturating_mul(1_000)
    }

    /// Signed-mutation clock-skew allowance in milliseconds.
    #[must_use]
    pub const fn mutation_clock_skew_ms(&self) -> u64 {
        self.mutation_clock_skew_secs.saturating_mul(1_000)
    }
}

impl Default for BlindVaultConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            db_path: Self::default_db_path(),
            max_lease_ttl_secs: Self::default_max_lease_ttl_secs(),
            max_object_ttl_secs: Self::default_max_object_ttl_secs(),
            max_objects_per_lease: Self::default_max_objects_per_lease(),
            max_bytes_per_lease: Self::default_max_bytes_per_lease(),
            max_pull_objects: Self::default_max_pull_objects(),
            tombstone_ttl_secs: Self::default_tombstone_ttl_secs(),
            mutation_clock_skew_secs: Self::default_mutation_clock_skew_secs(),
            cleanup_interval_secs: Self::default_cleanup_interval_secs(),
        }
    }
}

fn invalid(field: &'static str, message: &'static str) -> ServerError {
    ServerError::config_invalid(&format!("blind_vault.{field}"), message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled_and_bounded() {
        let config = BlindVaultConfig::default();
        assert!(!config.enabled);
        assert!(config.max_object_ttl_secs <= config.max_lease_ttl_secs);
        assert!(config.max_bytes_per_lease >= LARGEST_PROTOCOL_OBJECT_BYTES);
        config.validate().expect("disabled defaults remain valid");
    }

    #[test]
    fn enabled_configuration_rejects_cross_limit_inconsistency() {
        let config = BlindVaultConfig {
            enabled: true,
            max_object_ttl_secs: 100,
            max_lease_ttl_secs: 99,
            ..BlindVaultConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn milliseconds_accessors_do_not_use_floating_point() {
        let config = BlindVaultConfig::default();
        assert_eq!(config.max_lease_ttl_ms(), config.max_lease_ttl_secs * 1_000);
        assert_eq!(
            config.mutation_clock_skew_ms(),
            config.mutation_clock_skew_secs * 1_000
        );
    }
}
