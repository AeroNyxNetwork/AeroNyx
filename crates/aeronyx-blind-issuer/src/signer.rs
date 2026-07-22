// ============================================
// File: crates/aeronyx-blind-issuer/src/signer.rs
// ============================================
// [BLIND-ISSUER 2026-07-23 by Codex] This module owns the identity-free signing
// boundary; private-key backends never receive account, lease, node, or network
// identity. Software RSA is the first backend, not part of the policy layer.
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
//! - Isolates private operations behind an HSM/KMS-ready backend contract.
//!
//! ## Calling Relationships
//! `config.rs` supplies custody policy; `api.rs` supplies authenticated bounded
//! requests; storage nodes consume only the resulting public keys/signatures.
//!
//! ## Next Developer Guide
//! Add HSM/KMS adapters by implementing `BlindSigningBackend`; do not extend
//! that trait with caller metadata or leak provider errors across this module.
//!
//! Last Modified: v0.2.0-BlindSigner - Separated signing policy from custody
//! backends and added canonical public-key/output-shape validation.
//! ============================================

use std::collections::HashMap;

use aeronyx_core::protocol::blind_vault::{
    BlindVaultBlindIssuerEpoch, BLIND_VAULT_BLIND_ADMISSION_VERSION,
    MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES, MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS,
    MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS, MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
    MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
};
use blind_rsa_signatures::{PublicKeySha384PSSRandomized, SecretKeySha384PSSRandomized};
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

/// Coarse failure returned by a private-key backend.
///
/// Provider-specific errors must be handled inside the adapter and must never
/// include blinded input bytes, key material, or remote caller metadata.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("blind signing backend operation failed")]
pub struct BlindSigningBackendError;

/// Minimal synchronous custody contract for software, HSM, or KMS signers.
///
/// The HTTP layer already moves calls onto Tokio's blocking pool. Keeping this
/// boundary synchronous permits native PKCS#11 drivers without coupling key
/// custody to the web runtime. The only input is a validated blinded message.
pub trait BlindSigningBackend: Send + Sync {
    /// Performs one private blind-signing operation.
    ///
    /// # Errors
    /// Returns a deliberately coarse error after adapter-local diagnostics.
    fn sign_blinded(&self, blinded_message: &[u8]) -> Result<Vec<u8>, BlindSigningBackendError>;
}

/// Validation failures while assembling signer policy around custody handles.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BlindSignerBuildError {
    /// No key or more keys than the protocol directory permits were supplied.
    #[error("blind signer key count is invalid")]
    InvalidKeyCount,
    /// Public epoch metadata or canonical RSA DER was invalid.
    #[error("blind signer public epoch is invalid")]
    InvalidPublicEpoch,
    /// Two custody handles advertised the same canonical public key.
    #[error("blind signer issuer keys must be unique")]
    DuplicateIssuer,
}

struct SoftwareRsaBlindSigningBackend {
    secret_key: SecretKeySha384PSSRandomized,
}

impl BlindSigningBackend for SoftwareRsaBlindSigningBackend {
    fn sign_blinded(&self, blinded_message: &[u8]) -> Result<Vec<u8>, BlindSigningBackendError> {
        self.secret_key
            .blind_sign(blinded_message)
            .map(|signature| signature.0)
            .map_err(|_| BlindSigningBackendError)
    }
}

/// One validated public epoch paired with an opaque private-key backend.
///
/// [BLIND-ISSUER-BACKEND 2026-07-23 by Codex] Construction derives the exact
/// RSA modulus length from canonical public DER. An adapter cannot weaken frame
/// bounds by reporting its own message size.
pub struct BlindSigningKey {
    backend: Box<dyn BlindSigningBackend>,
    public_epoch: BlindVaultBlindIssuerEpoch,
    blind_message_bytes: usize,
}

impl BlindSigningKey {
    /// Validates canonical RSA public material before accepting a custody handle.
    ///
    /// # Errors
    /// Returns an invalid-public-epoch error for unsupported versions, bad
    /// fingerprints, malformed/non-canonical DER, or RSA sizes outside 2048–4096.
    pub fn new(
        public_epoch: BlindVaultBlindIssuerEpoch,
        backend: Box<dyn BlindSigningBackend>,
    ) -> Result<Self, BlindSignerBuildError> {
        let lifetime_ms = public_epoch
            .expires_at_ms
            .checked_sub(public_epoch.not_before_ms)
            .ok_or(BlindSignerBuildError::InvalidPublicEpoch)?;
        let derived_key_id: [u8; 32] = Sha256::digest(&public_epoch.public_key_der).into();
        if public_epoch.admission_version != BLIND_VAULT_BLIND_ADMISSION_VERSION
            || public_epoch.issuer_key_id.iter().all(|byte| *byte == 0)
            || public_epoch.issuer_key_id != derived_key_id
            || public_epoch.public_key_der.is_empty()
            || public_epoch.public_key_der.len() > MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES
            || lifetime_ms == 0
            || lifetime_ms > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS
            || public_epoch.max_lease_ttl_ms == 0
        {
            return Err(BlindSignerBuildError::InvalidPublicEpoch);
        }

        let public_key = PublicKeySha384PSSRandomized::from_der(&public_epoch.public_key_der)
            .map_err(|_| BlindSignerBuildError::InvalidPublicEpoch)?;
        let canonical_der = public_key
            .to_der()
            .map_err(|_| BlindSignerBuildError::InvalidPublicEpoch)?;
        let blind_message_bytes = public_key.components().n().len();
        if canonical_der != public_epoch.public_key_der
            || !(MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES..=MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES)
                .contains(&blind_message_bytes)
        {
            return Err(BlindSignerBuildError::InvalidPublicEpoch);
        }

        Ok(Self {
            backend,
            public_epoch,
            blind_message_bytes,
        })
    }
}

struct IssuerKey {
    backend: Box<dyn BlindSigningBackend>,
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
        let mut signing_keys = Vec::with_capacity(config.keys.len());
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
            let signing_key = BlindSigningKey::new(
                public_epoch,
                Box::new(SoftwareRsaBlindSigningBackend { secret_key }),
            )
            .map_err(|_| ConfigError::InvalidPrivateKey)?;
            signing_keys.push(signing_key);
        }
        Self::from_signing_keys(signing_keys).map_err(|error| match error {
            BlindSignerBuildError::DuplicateIssuer => {
                ConfigError::InvalidPolicy("configured private keys must be unique")
            }
            BlindSignerBuildError::InvalidKeyCount => {
                ConfigError::InvalidPolicy("configured private key count is invalid")
            }
            BlindSignerBuildError::InvalidPublicEpoch => ConfigError::InvalidPrivateKey,
        })
    }

    /// Builds policy around already-provisioned software, HSM, or KMS handles.
    ///
    /// # Errors
    /// Returns an error when the key count is outside protocol bounds or public
    /// key fingerprints collide. Each `BlindSigningKey` has already validated
    /// its public epoch and modulus length.
    pub fn from_signing_keys(
        signing_keys: Vec<BlindSigningKey>,
    ) -> Result<Self, BlindSignerBuildError> {
        if signing_keys.is_empty() || signing_keys.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
            return Err(BlindSignerBuildError::InvalidKeyCount);
        }
        let mut keys = HashMap::with_capacity(signing_keys.len());
        for signing_key in signing_keys {
            let issuer_key_id = signing_key.public_epoch.issuer_key_id;
            if keys
                .insert(
                    issuer_key_id,
                    IssuerKey {
                        backend: signing_key.backend,
                        public_epoch: signing_key.public_epoch,
                        blind_message_bytes: signing_key.blind_message_bytes,
                    },
                )
                .is_some()
            {
                return Err(BlindSignerBuildError::DuplicateIssuer);
            }
        }
        Ok(Self { keys })
    }

    /// Returns the number of loaded key epochs without exposing private state.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Returns whether at least one configured key can sign at `now_ms`.
    #[must_use]
    pub fn has_active_key(&self, now_ms: u64) -> bool {
        self.keys.values().any(|key| {
            now_ms >= key.public_epoch.not_before_ms && now_ms < key.public_epoch.expires_at_ms
        })
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
            .backend
            .sign_blinded(&request.blinded_message)
            .map_err(|_| BlindSignError::SigningFailed)?;
        if blind_signature.len() != key.blind_message_bytes {
            return Err(BlindSignError::SigningFailed);
        }
        Ok(BlindSignResponse {
            version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id: request.issuer_key_id,
            blind_signature,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use blind_rsa_signatures::{DefaultRng, KeyPairSha384PSSRandomized};

    struct FixedBackend {
        result: Result<Vec<u8>, BlindSigningBackendError>,
    }

    impl BlindSigningBackend for FixedBackend {
        fn sign_blinded(
            &self,
            _blinded_message: &[u8],
        ) -> Result<Vec<u8>, BlindSigningBackendError> {
            self.result.clone()
        }
    }

    fn test_epoch() -> BlindVaultBlindIssuerEpoch {
        let key_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
        BlindVaultBlindIssuerEpoch::new(
            key_pair.pk.to_der().expect("public DER"),
            1_000,
            10_000,
            1_000,
        )
    }

    #[test]
    fn backend_failures_and_malformed_outputs_remain_coarse() {
        let epoch = test_epoch();
        let request = BlindSignRequest {
            version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id: epoch.issuer_key_id,
            blinded_message: vec![7; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES],
        };
        let failing = BlindSigner::from_signing_keys(vec![BlindSigningKey::new(
            epoch.clone(),
            Box::new(FixedBackend {
                result: Err(BlindSigningBackendError),
            }),
        )
        .expect("failing signing key")])
        .expect("failing signer");
        assert_eq!(
            failing.sign(&request, 2_000),
            Err(BlindSignError::SigningFailed)
        );

        let truncated = BlindSigner::from_signing_keys(vec![BlindSigningKey::new(
            epoch,
            Box::new(FixedBackend {
                result: Ok(vec![0; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES - 1]),
            }),
        )
        .expect("truncated signing key")])
        .expect("truncated signer");
        assert_eq!(
            truncated.sign(&request, 2_000),
            Err(BlindSignError::SigningFailed)
        );
    }

    #[test]
    fn signer_rejects_duplicate_and_noncanonical_public_keys() {
        let epoch = test_epoch();
        let first = BlindSigningKey::new(
            epoch.clone(),
            Box::new(FixedBackend {
                result: Ok(vec![0; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES]),
            }),
        )
        .expect("first signing key");
        let second = BlindSigningKey::new(
            epoch.clone(),
            Box::new(FixedBackend {
                result: Ok(vec![0; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES]),
            }),
        )
        .expect("second signing key");
        assert!(matches!(
            BlindSigner::from_signing_keys(vec![first, second]),
            Err(BlindSignerBuildError::DuplicateIssuer)
        ));

        let mut malformed = epoch;
        malformed.public_key_der.push(0);
        malformed.issuer_key_id = Sha256::digest(&malformed.public_key_der).into();
        assert!(matches!(
            BlindSigningKey::new(
                malformed,
                Box::new(FixedBackend {
                    result: Ok(vec![0; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES]),
                }),
            ),
            Err(BlindSignerBuildError::InvalidPublicEpoch)
        ));
    }
}
