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
//!   admission lifetime, deletion tombstones, and maintenance work.
//! - Keeps public routes fail-closed unless at least one Ed25519 admission
//!   issuer is explicitly pinned by the node operator.
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
//! - Admission issuer keys authorize anonymous bearer credentials only. Never
//!   replace them with account, wallet, device, or application allowlists.
//! - Keep byte/count limits finite even on official nodes.
//!
//! Last Modified: v1.1.0-BlindVaultAdmission - Added default-off public API and
//! pinned anonymous admission issuer policy.
//! v1.0.0-BlindVaultService - Initial bounded configuration.
//! ============================================

use std::collections::HashSet;

use aeronyx_core::crypto::keys::IdentityPublicKey;
use serde::{Deserialize, Serialize};

use crate::error::{Result, ServerError};

const MAX_LEASE_TTL_SECS_HARD: u64 = 365 * 24 * 60 * 60;
const MAX_OBJECTS_PER_LEASE_HARD: u64 = 1_000_000;
const MAX_BYTES_PER_LEASE_HARD: u64 = 64 * 1024 * 1024 * 1024;
const MAX_PULL_OBJECTS_HARD: usize = 256;
const LARGEST_PROTOCOL_OBJECT_BYTES: u64 = 256 * 1024;
const MAX_ADMISSION_ISSUERS: usize = 16;
const MAX_ADMISSION_TICKET_TTL_SECS_HARD: u64 = 7 * 24 * 60 * 60;

/// Bounded, opt-in node policy for anonymous encrypted object storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindVaultConfig {
    /// Enables local Blind Vault persistence and client routes.
    #[serde(default)]
    pub enabled: bool,
    /// Exposes client routes only when storage and admission issuer policy are
    /// both explicitly configured. Storage initialization alone remains local.
    #[serde(default)]
    pub public_api_enabled: bool,
    /// Operator-pinned Ed25519 issuer public keys, encoded as 64 hex digits.
    /// Tickets carry no account or application identity.
    #[serde(default)]
    pub admission_issuer_public_keys: Vec<String>,
    /// Maximum validity window accepted for one bearer admission ticket.
    #[serde(default = "BlindVaultConfig::default_max_admission_ticket_ttl_secs")]
    pub max_admission_ticket_ttl_secs: u64,
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

    const fn default_max_admission_ticket_ttl_secs() -> u64 {
        24 * 60 * 60
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
        if self.public_api_enabled && !self.enabled {
            return Err(invalid(
                "public_api_enabled",
                "requires blind_vault.enabled=true",
            ));
        }
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
        if self.max_admission_ticket_ttl_secs == 0
            || self.max_admission_ticket_ttl_secs > MAX_ADMISSION_TICKET_TTL_SECS_HARD
        {
            return Err(invalid(
                "max_admission_ticket_ttl_secs",
                "must be between 1 second and 7 days",
            ));
        }
        let issuer_keys = self.admission_issuer_key_bytes()?;
        if self.public_api_enabled && issuer_keys.is_empty() {
            return Err(invalid(
                "admission_issuer_public_keys",
                "must pin at least one issuer when the public API is enabled",
            ));
        }
        Ok(())
    }

    /// Parses and validates the bounded issuer allowlist.
    pub fn admission_issuer_key_bytes(&self) -> Result<Vec<[u8; 32]>> {
        if self.admission_issuer_public_keys.len() > MAX_ADMISSION_ISSUERS {
            return Err(invalid(
                "admission_issuer_public_keys",
                "must contain no more than 16 keys",
            ));
        }
        let mut unique = HashSet::with_capacity(self.admission_issuer_public_keys.len());
        let mut parsed = Vec::with_capacity(self.admission_issuer_public_keys.len());
        for encoded in &self.admission_issuer_public_keys {
            let mut bytes = [0_u8; 32];
            hex::decode_to_slice(encoded.trim(), &mut bytes).map_err(|_| {
                invalid(
                    "admission_issuer_public_keys",
                    "each key must be exactly 32 bytes of hexadecimal",
                )
            })?;
            IdentityPublicKey::from_bytes(&bytes).map_err(|_| {
                invalid(
                    "admission_issuer_public_keys",
                    "contains an invalid Ed25519 public key",
                )
            })?;
            if !unique.insert(bytes) {
                return Err(invalid(
                    "admission_issuer_public_keys",
                    "must not contain duplicate keys",
                ));
            }
            parsed.push(bytes);
        }
        Ok(parsed)
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

    /// Maximum admission-ticket validity window in milliseconds.
    #[must_use]
    pub const fn max_admission_ticket_ttl_ms(&self) -> u64 {
        self.max_admission_ticket_ttl_secs.saturating_mul(1_000)
    }
}

impl Default for BlindVaultConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            public_api_enabled: false,
            admission_issuer_public_keys: Vec::new(),
            max_admission_ticket_ttl_secs: Self::default_max_admission_ticket_ttl_secs(),
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
        assert_eq!(
            config.max_admission_ticket_ttl_ms(),
            config.max_admission_ticket_ttl_secs * 1_000
        );
    }

    #[test]
    fn public_api_requires_a_valid_unique_issuer_key() {
        let issuer =
            aeronyx_core::crypto::keys::IdentityKeyPair::from_bytes(&[19; 32]).expect("issuer key");
        let valid = BlindVaultConfig {
            enabled: true,
            public_api_enabled: true,
            admission_issuer_public_keys: vec![hex::encode(issuer.public_key_bytes())],
            ..BlindVaultConfig::default()
        };
        valid.validate().expect("valid public API policy");

        let duplicate = BlindVaultConfig {
            admission_issuer_public_keys: vec![
                hex::encode(issuer.public_key_bytes()),
                hex::encode(issuer.public_key_bytes()),
            ],
            ..valid
        };
        assert!(duplicate.validate().is_err());
    }
}
