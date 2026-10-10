// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/issuer_directory.rs
// ============================================
//! # Blind-admission issuer epochs, directory, and updates
//!
//! Owns the public RFC 9474 issuer-epoch type, the node-signed issuer
//! directory used for authenticated key discovery and rotation, the
//! authority-signed issuer update, and their shared epoch signing helpers.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use serde::{Deserialize, Serialize};

use crate::crypto::keys::{IdentityKeyPair, IdentityPublicKey};

use super::error::BlindVaultError;
use super::{
    require_non_zero, require_version, serde_bytes64, sha256,
    BLIND_ISSUER_DIRECTORY_SIGNING_DOMAIN, BLIND_ISSUER_UPDATE_SIGNING_DOMAIN,
    BLIND_VAULT_BLIND_ADMISSION_VERSION, BLIND_VAULT_PROTOCOL_VERSION,
};

/// Maximum rotating blind-admission keys advertised by one storage node.
pub const MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS: usize = 16;

/// Maximum canonical RSA public-key DER accepted in an issuer directory.
pub const MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES: usize = 800;

/// Maximum lifetime of one advertised issuer epoch.
pub const MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS: u64 = 31 * 24 * 60 * 60 * 1_000;

/// One public RFC 9474 issuer key and its node-enforced coarse policy.
///
/// The key is public by design. No issuer URL, account scope, product tier, or
/// issuance transcript belongs in this storage-node directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindIssuerEpoch {
    /// Blind-admission scheme version accepted under this key.
    pub admission_version: u16,
    /// SHA-256 fingerprint of `public_key_der`.
    pub issuer_key_id: [u8; 32],
    /// Canonical SPKI DER for the RSA public key.
    pub public_key_der: Vec<u8>,
    /// Inclusive activation time in Unix milliseconds.
    pub not_before_ms: u64,
    /// Exclusive expiry time in Unix milliseconds.
    pub expires_at_ms: u64,
    /// Maximum anonymous lease lifetime authorized by this key epoch.
    pub max_lease_ttl_ms: u64,
}

impl BlindVaultBlindIssuerEpoch {
    /// Builds an epoch and derives its stable key fingerprint.
    #[must_use]
    pub fn new(
        public_key_der: Vec<u8>,
        not_before_ms: u64,
        expires_at_ms: u64,
        max_lease_ttl_ms: u64,
    ) -> Self {
        let issuer_key_id = sha256(&public_key_der);
        Self {
            admission_version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id,
            public_key_der,
            not_before_ms,
            expires_at_ms,
            max_lease_ttl_ms,
        }
    }

    fn validate_at(&self, generated_at_ms: u64) -> Result<(), BlindVaultError> {
        if self.admission_version != BLIND_VAULT_BLIND_ADMISSION_VERSION {
            return Err(BlindVaultError::UnsupportedBlindAdmissionVersion(
                self.admission_version,
            ));
        }
        require_non_zero("blind_issuer_key_id", &self.issuer_key_id)?;
        if self.public_key_der.is_empty()
            || self.public_key_der.len() > MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES
        {
            return Err(BlindVaultError::InvalidBlindIssuerKeyLength {
                actual: self.public_key_der.len(),
            });
        }
        if sha256(&self.public_key_der) != self.issuer_key_id {
            return Err(BlindVaultError::BlindIssuerKeyIdMismatch);
        }
        let epoch_lifetime_ms = self
            .expires_at_ms
            .checked_sub(self.not_before_ms)
            .ok_or(BlindVaultError::InvalidBlindIssuerEpochPolicy)?;
        if epoch_lifetime_ms == 0
            || epoch_lifetime_ms > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS
            || self.max_lease_ttl_ms == 0
            || self.expires_at_ms <= generated_at_ms
        {
            return Err(BlindVaultError::InvalidBlindIssuerEpochPolicy);
        }
        Ok(())
    }
}

/// Authenticated discovery response for blind-admission key rotation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindIssuerDirectory {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Response creation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Descriptor identity of the responding storage node.
    pub node_id: [u8; 32],
    /// Strictly key-ID-sorted active and pre-announced future epochs.
    pub epochs: Vec<BlindVaultBlindIssuerEpoch>,
    /// Ed25519 signature by `node_id` over all directory fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultBlindIssuerDirectory {
    /// Builds an unsigned issuer directory.
    #[must_use]
    pub const fn new(
        generated_at_ms: u64,
        node_id: [u8; 32],
        epochs: Vec<BlindVaultBlindIssuerEpoch>,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            generated_at_ms,
            node_id,
            epochs,
            signature: [0; 64],
        }
    }

    /// Canonical key-directory signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let epoch_bytes = blind_issuer_epoch_bytes_capacity(&self.epochs);
        let mut bytes =
            Vec::with_capacity(BLIND_ISSUER_DIRECTORY_SIGNING_DOMAIN.len() + 44 + epoch_bytes);
        bytes.extend_from_slice(BLIND_ISSUER_DIRECTORY_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.generated_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.node_id);
        append_blind_issuer_epochs(&mut bytes, &self.epochs);
        bytes
    }

    /// Validates and signs the directory with the node descriptor identity.
    ///
    /// # Errors
    /// Returns an invariant error for malformed epochs or a node-identity
    /// mismatch.
    pub fn sign(&mut self, node_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.public_key_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        self.signature = node_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Verifies bounds, freshness, expected node identity, and signature.
    ///
    /// # Errors
    /// Returns an invariant, freshness, identity, or signature error when the
    /// directory cannot be trusted.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_age_ms: u64,
        maximum_clock_skew_ms: u64,
        node_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if maximum_age_ms == 0
            || self.generated_at_ms > now_ms.saturating_add(maximum_clock_skew_ms)
            || now_ms.saturating_sub(self.generated_at_ms) > maximum_age_ms
        {
            return Err(BlindVaultError::IssuerDirectoryTimestampOutsideWindow);
        }
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("blind_issuer_directory_node_id", &self.node_id)?;
        if self.epochs.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
            return Err(BlindVaultError::TooManyBlindIssuerEpochs);
        }
        for epoch in &self.epochs {
            epoch.validate_at(self.generated_at_ms)?;
        }
        if self
            .epochs
            .windows(2)
            .any(|pair| pair[0].issuer_key_id >= pair[1].issuer_key_id)
        {
            return Err(BlindVaultError::BlindIssuerEpochOrderInvalid);
        }
        Ok(())
    }
}

/// [BLIND-VAULT-ISSUER-UPDATE 2026-07-23 by Codex]
/// Authority-signed public issuer generation accepted by storage nodes.
///
/// This object is transport-independent: a backend management channel, an
/// offline operator tool, or a future node synchronization layer may carry the
/// exact same signed bytes. It contains no issuer private key or issuance,
/// account, wallet, lease, storage, or client-network identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindIssuerUpdate {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Strictly increasing control-plane generation, starting at one.
    pub generation: u64,
    /// Update creation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Ed25519 authority identity explicitly pinned by the node operator.
    pub authority_id: [u8; 32],
    /// Strictly key-ID-sorted active and pre-announced future epochs.
    pub epochs: Vec<BlindVaultBlindIssuerEpoch>,
    /// Ed25519 authority signature over every preceding field.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultBlindIssuerUpdate {
    /// Builds one unsigned issuer update.
    #[must_use]
    pub const fn new(
        generation: u64,
        generated_at_ms: u64,
        authority_id: [u8; 32],
        epochs: Vec<BlindVaultBlindIssuerEpoch>,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            generation,
            generated_at_ms,
            authority_id,
            epochs,
            signature: [0; 64],
        }
    }

    /// Canonical authority-signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let epoch_bytes = blind_issuer_epoch_bytes_capacity(&self.epochs);
        let mut bytes =
            Vec::with_capacity(BLIND_ISSUER_UPDATE_SIGNING_DOMAIN.len() + 52 + epoch_bytes);
        bytes.extend_from_slice(BLIND_ISSUER_UPDATE_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.generation.to_be_bytes());
        bytes.extend_from_slice(&self.generated_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.authority_id);
        append_blind_issuer_epochs(&mut bytes, &self.epochs);
        bytes
    }

    /// Validates and signs this update with its declared authority identity.
    ///
    /// # Errors
    /// Returns a bounded invariant or identity error for malformed input.
    pub fn sign(&mut self, authority_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.authority_id != authority_key.public_key_bytes() {
            return Err(BlindVaultError::BlindIssuerAuthorityMismatch);
        }
        self.signature = authority_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Verifies bounds, freshness, pinned authority identity, and signature.
    ///
    /// # Errors
    /// Returns a fail-closed protocol error for stale, forged, or malformed
    /// updates.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_age_ms: u64,
        maximum_clock_skew_ms: u64,
        authority_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if maximum_age_ms == 0
            || self.generated_at_ms > now_ms.saturating_add(maximum_clock_skew_ms)
            || now_ms.saturating_sub(self.generated_at_ms) > maximum_age_ms
        {
            return Err(BlindVaultError::BlindIssuerUpdateTimestampOutsideWindow);
        }
        if self.authority_id != authority_key.to_bytes() {
            return Err(BlindVaultError::BlindIssuerAuthorityMismatch);
        }
        authority_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("blind_issuer_update_authority_id", &self.authority_id)?;
        if self.generation == 0 {
            return Err(BlindVaultError::InvalidBlindIssuerUpdateGeneration);
        }
        if self.epochs.is_empty() {
            return Err(BlindVaultError::BlindIssuerUpdateHasNoEpochs);
        }
        if self.epochs.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
            return Err(BlindVaultError::TooManyBlindIssuerEpochs);
        }
        for epoch in &self.epochs {
            epoch.validate_at(self.generated_at_ms)?;
        }
        if self
            .epochs
            .windows(2)
            .any(|pair| pair[0].issuer_key_id >= pair[1].issuer_key_id)
        {
            return Err(BlindVaultError::BlindIssuerEpochOrderInvalid);
        }
        Ok(())
    }
}

fn blind_issuer_epoch_bytes_capacity(epochs: &[BlindVaultBlindIssuerEpoch]) -> usize {
    epochs
        .iter()
        .map(|epoch| 90usize.saturating_add(epoch.public_key_der.len()))
        .sum()
}

fn append_blind_issuer_epochs(bytes: &mut Vec<u8>, epochs: &[BlindVaultBlindIssuerEpoch]) {
    let epoch_count = u16::try_from(epochs.len()).unwrap_or(u16::MAX);
    bytes.extend_from_slice(&epoch_count.to_be_bytes());
    for epoch in epochs {
        bytes.extend_from_slice(&epoch.admission_version.to_be_bytes());
        bytes.extend_from_slice(&epoch.issuer_key_id);
        let der_length = u16::try_from(epoch.public_key_der.len()).unwrap_or(u16::MAX);
        bytes.extend_from_slice(&der_length.to_be_bytes());
        bytes.extend_from_slice(&epoch.public_key_der);
        bytes.extend_from_slice(&epoch.not_before_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.max_lease_ttl_ms.to_be_bytes());
    }
}

#[cfg(test)]
mod tests;
