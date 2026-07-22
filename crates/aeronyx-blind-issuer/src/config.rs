// ============================================
// File: crates/aeronyx-blind-issuer/src/config.rs
// ============================================
// [BLIND-ISSUER 2026-07-23 by Codex] Fail-closed configuration and secret-file
// policy for the isolated blind-admission signer.
//! # Blind Issuer Configuration
//!
//! ## Creation Reason
//! Private signing keys and backend authentication tokens require stricter
//! loading rules than ordinary node configuration.
//!
//! ## Main Functionality
//! - Rejects non-loopback listeners and unbounded key epochs.
//! - Reads secrets with `O_NOFOLLOW`, bounded allocation, and owner-only mode.
//! - Keeps secret bytes in zeroizing buffers until cryptographic parsing.
//! - Bounds how long HTTP callers wait for non-cancellable custody operations.
//! - Bounds circuit-breaker failure and recovery policy.
//!
//! ## Calling Relationships
//! `main.rs` loads this policy; `signer.rs` consumes key files; `api.rs`
//! consumes the backend token and pressure limits.
//!
//! ## Important Configuration
//! Every secret file must be a regular, single-link file with no group/other
//! permission bits. Issuer epochs may overlap but may not exceed 31 days.
//!
//! ## Next Developer Guide
//! Do not add inline private keys or bearer tokens to TOML/environment values.
//! Secret-manager integration should implement a separate custody backend.
//!
//! Last Modified: v0.4.0-BlindIssuerConfig - Defined key-only hot-reload policy.
//! ============================================

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Take};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use aeronyx_core::protocol::blind_vault::{
    MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS, MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS,
};
use serde::Deserialize;
use thiserror::Error;
use zeroize::Zeroizing;

const MAX_CONFIG_BYTES: u64 = 64 * 1024;
const MAX_PRIVATE_KEY_BYTES: u64 = 16 * 1024;
const MAX_AUTH_TOKEN_BYTES: u64 = 256;
const MIN_AUTH_TOKEN_BYTES: usize = 32;
const MAX_AUTH_TOKEN_LENGTH: usize = 128;
const MAX_LEASE_TTL_SECS: u64 = 365 * 24 * 60 * 60;
const MIN_SIGNING_TIMEOUT_MS: u64 = 100;
const MAX_SIGNING_TIMEOUT_MS: u64 = 120_000;
const MIN_CIRCUIT_COOLDOWN_MS: u64 = 1_000;
const MAX_CIRCUIT_COOLDOWN_MS: u64 = 300_000;
const MAX_CIRCUIT_FAILURE_THRESHOLD: u64 = 100;

/// Default caller wait bound for one private custody operation.
pub const DEFAULT_SIGNING_TIMEOUT_MS: u64 = 10_000;
/// Default consecutive backend failures before the signer circuit opens.
pub const DEFAULT_CIRCUIT_FAILURE_THRESHOLD: u64 = 5;
/// Default monotonic recovery delay after the signer circuit opens.
pub const DEFAULT_CIRCUIT_COOLDOWN_MS: u64 = 30_000;

/// Fail-closed issuer startup/configuration failures.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Configuration or secret file I/O failed.
    #[error("blind issuer file operation failed")]
    Io(#[from] std::io::Error),
    /// TOML was malformed or contained unknown fields.
    #[error("blind issuer configuration is invalid")]
    InvalidToml(#[from] toml::de::Error),
    /// TOML was not valid UTF-8.
    #[error("blind issuer configuration encoding is invalid")]
    InvalidEncoding,
    /// A listener, epoch, limit, or path violated policy.
    #[error("blind issuer configuration policy is invalid: {0}")]
    InvalidPolicy(&'static str),
    /// Secret file permissions, type, or hard-link count were unsafe.
    #[error("blind issuer secret file is not securely permissioned")]
    InsecureSecretFile,
    /// A bounded file exceeded its maximum accepted size.
    #[error("blind issuer file exceeds its size limit")]
    FileTooLarge,
    /// Secret files cannot be safely opened on this platform.
    #[error("secure issuer secret files are unsupported on this platform")]
    SecureFilesUnsupported,
    /// Backend bearer token content was malformed.
    #[error("blind issuer backend token is invalid")]
    InvalidAuthToken,
    /// RSA private key material was invalid or unsupported.
    #[error("blind issuer private key is invalid")]
    InvalidPrivateKey,
}

/// One rotating RFC 9474 private-key epoch.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlindIssuerKeyConfig {
    /// PKCS#8 or PKCS#1 DER file containing one RSA private key.
    pub private_key_der_file: PathBuf,
    /// Inclusive activation time as Unix seconds.
    pub not_before_unix_secs: u64,
    /// Exclusive expiry time as Unix seconds.
    pub expires_at_unix_secs: u64,
    /// Maximum lease lifetime nodes should pin for this key.
    pub max_lease_ttl_secs: u64,
}

/// Complete isolated signer process configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlindIssuerConfig {
    /// Loopback TCP socket used only by the authenticated central backend.
    pub listen_addr: String,
    /// Owner-only file containing the backend bearer token.
    pub auth_token_file: PathBuf,
    /// Maximum accepted signing requests per second for the whole process.
    #[serde(default = "default_max_requests_per_second")]
    pub max_requests_per_second: u64,
    /// Maximum concurrent private-key operations.
    #[serde(default = "default_max_in_flight")]
    pub max_in_flight: usize,
    /// Caller wait bound for one private operation; the operation retains its
    /// capacity permit if the HTTP response times out before an HSM returns.
    #[serde(default = "default_signing_timeout_ms")]
    pub signing_timeout_ms: u64,
    /// Consecutive backend failures/timeouts required to open the circuit.
    #[serde(default = "default_circuit_failure_threshold")]
    pub circuit_failure_threshold: u64,
    /// Monotonic delay before requests may probe an open circuit again.
    #[serde(default = "default_circuit_cooldown_ms")]
    pub circuit_cooldown_ms: u64,
    /// Rotating private-key epochs. Overlap is allowed for safe rollout.
    pub keys: Vec<BlindIssuerKeyConfig>,
}

impl BlindIssuerConfig {
    /// Loads and validates one bounded TOML file.
    ///
    /// # Errors
    /// Returns an I/O, TOML, or policy error without including secret content.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let bytes = read_bounded_file(path, MAX_CONFIG_BYTES)?;
        let text = std::str::from_utf8(&bytes).map_err(|_| ConfigError::InvalidEncoding)?;
        let config: Self = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates policy independently of secret-file contents.
    ///
    /// # Errors
    /// Returns `InvalidPolicy` for unsafe listeners, limits, paths, or epochs.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let listen_addr = self.listen_addr()?;
        if !listen_addr.ip().is_loopback() {
            return Err(ConfigError::InvalidPolicy(
                "listen_addr must use a loopback IP",
            ));
        }
        if !self.auth_token_file.is_absolute() {
            return Err(ConfigError::InvalidPolicy(
                "auth_token_file must be an absolute path",
            ));
        }
        if self.max_requests_per_second == 0 || self.max_requests_per_second > 10_000 {
            return Err(ConfigError::InvalidPolicy(
                "max_requests_per_second must be between 1 and 10000",
            ));
        }
        if self.max_in_flight == 0 || self.max_in_flight > 64 {
            return Err(ConfigError::InvalidPolicy(
                "max_in_flight must be between 1 and 64",
            ));
        }
        if !(MIN_SIGNING_TIMEOUT_MS..=MAX_SIGNING_TIMEOUT_MS).contains(&self.signing_timeout_ms) {
            return Err(ConfigError::InvalidPolicy(
                "signing_timeout_ms must be between 100 and 120000",
            ));
        }
        if self.circuit_failure_threshold == 0
            || self.circuit_failure_threshold > MAX_CIRCUIT_FAILURE_THRESHOLD
        {
            return Err(ConfigError::InvalidPolicy(
                "circuit_failure_threshold must be between 1 and 100",
            ));
        }
        if !(MIN_CIRCUIT_COOLDOWN_MS..=MAX_CIRCUIT_COOLDOWN_MS).contains(&self.circuit_cooldown_ms)
        {
            return Err(ConfigError::InvalidPolicy(
                "circuit_cooldown_ms must be between 1000 and 300000",
            ));
        }
        if self.keys.is_empty() || self.keys.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
            return Err(ConfigError::InvalidPolicy(
                "keys must contain between 1 and 16 epochs",
            ));
        }

        let mut unique_paths = HashSet::with_capacity(self.keys.len());
        for key in &self.keys {
            if !key.private_key_der_file.is_absolute()
                || !unique_paths.insert(key.private_key_der_file.clone())
            {
                return Err(ConfigError::InvalidPolicy(
                    "private key paths must be absolute and unique",
                ));
            }
            let epoch_secs = key
                .expires_at_unix_secs
                .checked_sub(key.not_before_unix_secs)
                .ok_or(ConfigError::InvalidPolicy(
                    "issuer expiry must be after activation",
                ))?;
            if epoch_secs == 0 || epoch_secs > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS / 1_000 {
                return Err(ConfigError::InvalidPolicy(
                    "issuer epochs must be between 1 second and 31 days",
                ));
            }
            if key.max_lease_ttl_secs == 0 || key.max_lease_ttl_secs > MAX_LEASE_TTL_SECS {
                return Err(ConfigError::InvalidPolicy(
                    "max_lease_ttl_secs must be between 1 second and 365 days",
                ));
            }
        }
        Ok(())
    }

    /// Parses the validated loopback listener.
    ///
    /// # Errors
    /// Returns `InvalidPolicy` when the socket address is malformed.
    pub fn listen_addr(&self) -> Result<SocketAddr, ConfigError> {
        self.listen_addr
            .parse()
            .map_err(|_| ConfigError::InvalidPolicy("listen_addr must be a socket address"))
    }

    /// Loads and validates the backend authentication token.
    ///
    /// # Errors
    /// Returns a secure-file or token-content error.
    pub fn load_auth_token(&self) -> Result<Zeroizing<Vec<u8>>, ConfigError> {
        let bytes = read_secure_file(&self.auth_token_file, MAX_AUTH_TOKEN_BYTES)?;
        let start = bytes
            .iter()
            .position(|byte| !byte.is_ascii_whitespace())
            .unwrap_or(bytes.len());
        let end = bytes
            .iter()
            .rposition(|byte| !byte.is_ascii_whitespace())
            .map_or(start, |index| index + 1);
        let token = Zeroizing::new(bytes[start..end].to_vec());
        if !(MIN_AUTH_TOKEN_BYTES..=MAX_AUTH_TOKEN_LENGTH).contains(&token.len())
            || !token.iter().all(u8::is_ascii_graphic)
        {
            return Err(ConfigError::InvalidAuthToken);
        }
        Ok(token)
    }

    /// Returns the validated caller wait bound for one signing operation.
    #[must_use]
    pub const fn signing_timeout(&self) -> Duration {
        Duration::from_millis(self.signing_timeout_ms)
    }

    /// Returns the validated monotonic circuit-breaker cooldown.
    #[must_use]
    pub const fn circuit_cooldown(&self) -> Duration {
        Duration::from_millis(self.circuit_cooldown_ms)
    }

    /// Returns whether `candidate` changes only rotating key epochs.
    ///
    /// [BLIND-ISSUER-RELOAD 2026-07-23 by Codex] Listener, authentication,
    /// pressure, timeout, and circuit policy remain startup-only. Silently
    /// applying only part of a changed configuration would leave operators
    /// believing limits or credentials had rotated when they had not.
    #[must_use]
    pub fn key_reload_compatible_with(&self, candidate: &Self) -> bool {
        self.listen_addr == candidate.listen_addr
            && self.auth_token_file == candidate.auth_token_file
            && self.max_requests_per_second == candidate.max_requests_per_second
            && self.max_in_flight == candidate.max_in_flight
            && self.signing_timeout_ms == candidate.signing_timeout_ms
            && self.circuit_failure_threshold == candidate.circuit_failure_threshold
            && self.circuit_cooldown_ms == candidate.circuit_cooldown_ms
    }
}

pub(crate) fn read_private_key_der(path: &Path) -> Result<Zeroizing<Vec<u8>>, ConfigError> {
    read_secure_file(path, MAX_PRIVATE_KEY_BYTES)
}

const fn default_max_requests_per_second() -> u64 {
    128
}

const fn default_max_in_flight() -> usize {
    8
}

const fn default_signing_timeout_ms() -> u64 {
    DEFAULT_SIGNING_TIMEOUT_MS
}

const fn default_circuit_failure_threshold() -> u64 {
    DEFAULT_CIRCUIT_FAILURE_THRESHOLD
}

const fn default_circuit_cooldown_ms() -> u64 {
    DEFAULT_CIRCUIT_COOLDOWN_MS
}

fn read_bounded_file(path: &Path, maximum_bytes: u64) -> Result<Vec<u8>, ConfigError> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(ConfigError::InvalidPolicy(
            "configuration path must be a regular file",
        ));
    }
    if metadata.len() > maximum_bytes {
        return Err(ConfigError::FileTooLarge);
    }
    read_bounded_reader(file.take(maximum_bytes + 1), maximum_bytes)
}

fn read_bounded_reader(mut reader: Take<File>, maximum_bytes: u64) -> Result<Vec<u8>, ConfigError> {
    let allocation = usize::try_from(maximum_bytes).map_err(|_| ConfigError::FileTooLarge)?;
    let mut bytes = Vec::with_capacity(allocation);
    reader.read_to_end(&mut bytes)?;
    let bytes_read = u64::try_from(bytes.len()).map_err(|_| ConfigError::FileTooLarge)?;
    if bytes_read > maximum_bytes {
        return Err(ConfigError::FileTooLarge);
    }
    Ok(bytes)
}

#[cfg(unix)]
fn read_secure_file(path: &Path, maximum_bytes: u64) -> Result<Zeroizing<Vec<u8>>, ConfigError> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let metadata = file.metadata()?;
    // [BLIND-ISSUER-HARDENING 2026-07-23 by Codex] Mode 0600 is not enough
    // when a privileged process can read a file owned by another account.
    if !metadata.is_file()
        || metadata.permissions().mode() & 0o077 != 0
        || metadata.nlink() != 1
        || metadata.uid() != nix::unistd::geteuid().as_raw()
    {
        return Err(ConfigError::InsecureSecretFile);
    }
    if metadata.len() > maximum_bytes {
        return Err(ConfigError::FileTooLarge);
    }
    Ok(Zeroizing::new(read_bounded_reader(
        file.take(maximum_bytes + 1),
        maximum_bytes,
    )?))
}

#[cfg(not(unix))]
fn read_secure_file(_path: &Path, _maximum_bytes: u64) -> Result<Zeroizing<Vec<u8>>, ConfigError> {
    Err(ConfigError::SecureFilesUnsupported)
}

#[cfg(all(test, unix))]
mod tests {
    // Panicking on fixture/setup errors is intentional in unit tests.
    #![allow(clippy::expect_used)]

    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn token_file_requires_owner_only_permissions() {
        let directory = tempfile::tempdir().expect("temp directory");
        let token_path = directory.path().join("backend.token");
        fs::write(&token_path, b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFG")
            .expect("write token");
        fs::set_permissions(&token_path, fs::Permissions::from_mode(0o600)).expect("secure mode");
        let config = BlindIssuerConfig {
            listen_addr: "127.0.0.1:9191".to_owned(),
            auth_token_file: token_path.clone(),
            max_requests_per_second: 128,
            max_in_flight: 8,
            signing_timeout_ms: DEFAULT_SIGNING_TIMEOUT_MS,
            circuit_failure_threshold: DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
            circuit_cooldown_ms: DEFAULT_CIRCUIT_COOLDOWN_MS,
            keys: vec![BlindIssuerKeyConfig {
                private_key_der_file: directory.path().join("issuer.der"),
                not_before_unix_secs: 1_800_000_000,
                expires_at_unix_secs: 1_800_086_400,
                max_lease_ttl_secs: 604_800,
            }],
        };
        assert!(config.load_auth_token().is_ok());

        fs::set_permissions(&token_path, fs::Permissions::from_mode(0o640)).expect("insecure mode");
        assert!(matches!(
            config.load_auth_token(),
            Err(ConfigError::InsecureSecretFile)
        ));
    }

    #[test]
    fn configuration_rejects_public_listener_and_long_epoch() {
        let config = BlindIssuerConfig {
            listen_addr: "0.0.0.0:9191".to_owned(),
            auth_token_file: PathBuf::from("/var/lib/aeronyx/backend.token"),
            max_requests_per_second: 128,
            max_in_flight: 8,
            signing_timeout_ms: DEFAULT_SIGNING_TIMEOUT_MS,
            circuit_failure_threshold: DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
            circuit_cooldown_ms: DEFAULT_CIRCUIT_COOLDOWN_MS,
            keys: vec![BlindIssuerKeyConfig {
                private_key_der_file: PathBuf::from("/var/lib/aeronyx/issuer.der"),
                not_before_unix_secs: 1,
                expires_at_unix_secs: 2,
                max_lease_ttl_secs: 1,
            }],
        };
        assert!(config.validate().is_err());

        let long_epoch = BlindIssuerConfig {
            listen_addr: "[::1]:9191".to_owned(),
            keys: vec![BlindIssuerKeyConfig {
                expires_at_unix_secs: 1 + MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS / 1_000 + 1,
                ..config.keys[0].clone()
            }],
            ..config
        };
        assert!(long_epoch.validate().is_err());
    }

    #[test]
    fn configuration_requires_absolute_secret_paths() {
        let mut config = BlindIssuerConfig {
            listen_addr: "127.0.0.1:9191".to_owned(),
            auth_token_file: PathBuf::from("backend.token"),
            max_requests_per_second: 128,
            max_in_flight: 8,
            signing_timeout_ms: DEFAULT_SIGNING_TIMEOUT_MS,
            circuit_failure_threshold: DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
            circuit_cooldown_ms: DEFAULT_CIRCUIT_COOLDOWN_MS,
            keys: vec![BlindIssuerKeyConfig {
                private_key_der_file: PathBuf::from("/var/lib/aeronyx/issuer.der"),
                not_before_unix_secs: 1,
                expires_at_unix_secs: 2,
                max_lease_ttl_secs: 1,
            }],
        };
        assert!(config.validate().is_err());

        config.auth_token_file = PathBuf::from("/var/lib/aeronyx/backend.token");
        config.keys[0].private_key_der_file = PathBuf::from("issuer.der");
        assert!(config.validate().is_err());
    }

    #[test]
    fn signing_timeout_is_bounded_and_defaults_for_old_configs() {
        let legacy_toml = r#"
listen_addr = "127.0.0.1:9191"
auth_token_file = "/var/lib/aeronyx/backend.token"
max_requests_per_second = 128
max_in_flight = 8

[[keys]]
private_key_der_file = "/var/lib/aeronyx/issuer.der"
not_before_unix_secs = 1
expires_at_unix_secs = 2
max_lease_ttl_secs = 1
"#;
        let mut config: BlindIssuerConfig = toml::from_str(legacy_toml).expect("legacy config");
        assert_eq!(config.signing_timeout_ms, DEFAULT_SIGNING_TIMEOUT_MS);
        assert_eq!(
            config.circuit_failure_threshold,
            DEFAULT_CIRCUIT_FAILURE_THRESHOLD
        );
        assert_eq!(config.circuit_cooldown_ms, DEFAULT_CIRCUIT_COOLDOWN_MS);
        assert_eq!(
            config.signing_timeout(),
            Duration::from_millis(DEFAULT_SIGNING_TIMEOUT_MS)
        );
        assert!(config.validate().is_ok());

        config.signing_timeout_ms = MIN_SIGNING_TIMEOUT_MS - 1;
        assert!(config.validate().is_err());
        config.signing_timeout_ms = MAX_SIGNING_TIMEOUT_MS + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn circuit_breaker_policy_is_bounded() {
        let mut config = BlindIssuerConfig {
            listen_addr: "127.0.0.1:9191".to_owned(),
            auth_token_file: PathBuf::from("/var/lib/aeronyx/backend.token"),
            max_requests_per_second: 128,
            max_in_flight: 8,
            signing_timeout_ms: DEFAULT_SIGNING_TIMEOUT_MS,
            circuit_failure_threshold: DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
            circuit_cooldown_ms: DEFAULT_CIRCUIT_COOLDOWN_MS,
            keys: vec![BlindIssuerKeyConfig {
                private_key_der_file: PathBuf::from("/var/lib/aeronyx/issuer.der"),
                not_before_unix_secs: 1,
                expires_at_unix_secs: 2,
                max_lease_ttl_secs: 1,
            }],
        };
        assert!(config.validate().is_ok());
        assert_eq!(
            config.circuit_cooldown(),
            Duration::from_millis(DEFAULT_CIRCUIT_COOLDOWN_MS)
        );

        let mut key_only_reload = config.clone();
        key_only_reload.keys[0].not_before_unix_secs += 1;
        assert!(config.key_reload_compatible_with(&key_only_reload));
        key_only_reload.max_in_flight += 1;
        assert!(!config.key_reload_compatible_with(&key_only_reload));

        config.circuit_failure_threshold = 0;
        assert!(config.validate().is_err());
        config.circuit_failure_threshold = DEFAULT_CIRCUIT_FAILURE_THRESHOLD;
        config.circuit_cooldown_ms = MIN_CIRCUIT_COOLDOWN_MS - 1;
        assert!(config.validate().is_err());
        config.circuit_cooldown_ms = MAX_CIRCUIT_COOLDOWN_MS + 1;
        assert!(config.validate().is_err());
    }
}
