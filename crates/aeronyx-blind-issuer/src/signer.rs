// ============================================
// File: crates/aeronyx-blind-issuer/src/signer.rs
// ============================================
// [BLIND-ISSUER 2026-07-23 by Codex] This module owns the only software RSA
// private-key operation; no account, lease, node, or network identity enters it.
//! # RFC 9474 Signing Boundary
//!
//! ## Creation Reason
//! Perform the only private RSA operation in a process that has no storage-node
//! database, user identity model, or redemption visibility.
//!
//! ## Main Functionality
//! - Loads validated private-key epochs from secure files.
//! - Derives canonical public DER and stable SHA-256 key IDs.
//! - Signs bounded blinded messages only while their key epoch is active.
//! - Publishes deterministic public epoch snapshots for configuration checks.
//!
//! ## Calling Relationships
//! `config.rs` supplies custody policy; `api.rs` supplies authenticated bounded
//! requests; storage nodes consume only the resulting public keys/signatures.
//!
//! ## Next Developer Guide
//! Replace `IssuerKey::secret_key` with an HSM/KMS handle behind this module;
//! do not leak request bodies or add caller metadata to this type.
//!
//! Last Modified: v0.1.0-BlindSigner - Initial software-key boundary.
//! ============================================

use std::collections::HashMap;

use aeronyx_core::protocol::blind_vault::{
    BlindVaultBlindIssuerEpoch, BLIND_VAULT_BLIND_ADMISSION_VERSION,
};
use blind_rsa_signatures::SecretKeySha384PSSRandomized;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::config::{read_private_key_der, BlindIssuerConfig, ConfigError};

/// Minimal identity-free signing request accepted by the custody boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindSignRequest {
    /// Blind-admission scheme version.
    pub version: u16,
    /// Public fingerprint selecting one configured key epoch.
    pub issuer_key_id: [u8; 32],
    /// RFC 9474 blinded message with RSA-modulus byte length.
    pub blinded_message: Vec<u8>,
}

/// Blind signature returned to the authenticated upstream backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindSignResponse {
    /// Blind-admission scheme version.
    pub version: u16,
    /// Key fingerprint that produced the signature.
    pub issuer_key_id: [u8; 32],
    /// Raw RSA blind signature for client-side finalization.
    pub blind_signature: Vec<u8>,
}

/// Privacy-safe signing failures mapped to coarse HTTP status buckets.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindSignError {
    /// Request version, key ID, or blinded-message shape was invalid.
    #[error("blind signing request is invalid")]
    InvalidRequest,
    /// Requested key ID is not configured by this signer.
    #[error("blind signing issuer is unavailable")]
    UnknownIssuer,
    /// Requested key exists but is outside its signing epoch.
    #[error("blind signing issuer is inactive")]
    InactiveIssuer,
    /// Private RSA operation failed without exposing cryptographic details.
    #[error("blind signing operation failed")]
    SigningFailed,
}

struct IssuerKey {
    secret_key: SecretKeySha384PSSRandomized,
    public_epoch: BlindVaultBlindIssuerEpoch,
    blind_message_bytes: usize,
}

/// In-memory private-key custody and active-epoch policy.
pub struct BlindSigner {
    keys: HashMap<[u8; 32], IssuerKey>,
}

impl BlindSigner {
    /// Loads validated private keys and rejects duplicate canonical key IDs.
    ///
    /// # Errors
    /// Returns a configuration, secure-file, or private-key parsing error.
    pub fn from_config(config: &BlindIssuerConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let mut keys = HashMap::with_capacity(config.keys.len());
        for key_config in &config.keys {
            let private_der = read_private_key_der(&key_config.private_key_der_file)?;
            let secret_key = SecretKeySha384PSSRandomized::from_der(&private_der)
                .map_err(|_| ConfigError::InvalidPrivateKey)?;
            let public_key = secret_key
                .public_key()
                .map_err(|_| ConfigError::InvalidPrivateKey)?;
            let public_der = public_key
                .to_der()
                .map_err(|_| ConfigError::InvalidPrivateKey)?;
            let issuer_key_id: [u8; 32] = Sha256::digest(&public_der).into();
            let not_before_ms = key_config
                .not_before_unix_secs
                .checked_mul(1_000)
                .ok_or(ConfigError::InvalidPolicy("issuer timestamp is too large"))?;
            let expires_at_ms = key_config
                .expires_at_unix_secs
                .checked_mul(1_000)
                .ok_or(ConfigError::InvalidPolicy("issuer timestamp is too large"))?;
            let max_lease_ttl_ms = key_config
                .max_lease_ttl_secs
                .checked_mul(1_000)
                .ok_or(ConfigError::InvalidPolicy("lease TTL is too large"))?;
            let public_epoch = BlindVaultBlindIssuerEpoch::new(
                public_der,
                not_before_ms,
                expires_at_ms,
                max_lease_ttl_ms,
            );
            if public_epoch.issuer_key_id != issuer_key_id {
                return Err(ConfigError::InvalidPrivateKey);
            }
            let blind_message_bytes = public_key.components().n().len();
            if keys
                .insert(
                    issuer_key_id,
                    IssuerKey {
                        secret_key,
                        public_epoch,
                        blind_message_bytes,
                    },
                )
                .is_some()
            {
                return Err(ConfigError::InvalidPolicy(
                    "configured private keys must be unique",
                ));
            }
        }
        Ok(Self { keys })
    }

    /// Returns the number of loaded key epochs without exposing private state.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Returns canonical non-expired public epochs in key-ID order.
    #[must_use]
    pub fn public_epochs(&self, now_ms: u64) -> Vec<BlindVaultBlindIssuerEpoch> {
        let mut epochs = self
            .keys
            .values()
            .filter(|key| key.public_epoch.expires_at_ms > now_ms)
            .map(|key| key.public_epoch.clone())
            .collect::<Vec<_>>();
        epochs.sort_by_key(|epoch| epoch.issuer_key_id);
        epochs
    }

    /// Signs one blinded message under an active selected key.
    ///
    /// # Errors
    /// Returns a coarse request, key availability, epoch, or signing failure.
    pub fn sign(
        &self,
        request: &BlindSignRequest,
        now_ms: u64,
    ) -> Result<BlindSignResponse, BlindSignError> {
        if request.version != BLIND_VAULT_BLIND_ADMISSION_VERSION
            || request.issuer_key_id.iter().all(|byte| *byte == 0)
        {
            return Err(BlindSignError::InvalidRequest);
        }
        let key = self
            .keys
            .get(&request.issuer_key_id)
            .ok_or(BlindSignError::UnknownIssuer)?;
        if now_ms < key.public_epoch.not_before_ms || now_ms >= key.public_epoch.expires_at_ms {
            return Err(BlindSignError::InactiveIssuer);
        }
        if request.blinded_message.len() != key.blind_message_bytes {
            return Err(BlindSignError::InvalidRequest);
        }
        let blind_signature = key
            .secret_key
            .blind_sign(&request.blinded_message)
            .map_err(|_| BlindSignError::SigningFailed)?;
        Ok(BlindSignResponse {
            version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id: request.issuer_key_id,
            blind_signature: blind_signature.0,
        })
    }
}
