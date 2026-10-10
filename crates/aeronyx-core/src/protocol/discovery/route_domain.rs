// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/route_domain.rs
// ============================================
//! # Portable route-domain attestations
//!
//! Owns opaque time-bounded route-domain attestations, their signing digest,
//! portable certificates verified against a local pinned-attestor policy,
//! and the bounded magic-prefixed certificate frame codec.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

use super::sync_signing::directory_sync_signing_digest;
use super::{
    serde_bytes64, AERONYX_DIRECTORY_MAINNET_CHAIN_ID, MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS,
};

/// Stable route-domain attestation statement version.
pub const ROUTE_DOMAIN_ATTESTATION_VERSION_V1: u16 = 1;

/// Stable portable route-domain certificate version.
pub const ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1: u16 = 1;

/// Maximum distinct attestations retained in one portable certificate.
pub const MAX_ROUTE_DOMAIN_ATTESTATIONS_V1: usize = 16;

/// Maximum lifetime of one route-domain attestation.
///
/// [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] Short-lived evidence limits
/// stale infrastructure/operator mappings without requiring revocation lists
/// in the first protocol version.
pub const MAX_ROUTE_DOMAIN_ATTESTATION_LIFETIME_SECS_V1: u64 = 30 * 24 * 60 * 60;

/// One-byte discriminator prepended to portable route-domain certificates.
pub const ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC: u8 = 0xc8;

/// Maximum encoded route-domain certificate payload, excluding its magic byte.
const MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES: u64 = 16 * 1024;

/// Maximum complete portable route-domain certificate frame size.
pub const MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES: usize =
    MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES as usize + 1;

// ============================================
// Portable Route-Domain Attestation V1
// ============================================

/// One independently signed, time-bounded opaque route-domain assignment.
///
/// [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] The 128-bit domain token is
/// deliberately opaque. A verifier learns only that one pinned attestor signed
/// the exact node/token pair for this interval; the statement does not encode
/// an operator, provider, ASN, geography, ownership claim, or public label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDomainAttestationV1 {
    /// Stable statement contract version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Node identity assigned to the opaque route domain.
    pub subject_node_id: [u8; 32],
    /// Opaque 128-bit route-domain token.
    pub route_domain: [u8; 16],
    /// Independent attestor identity.
    pub attestor_node_id: [u8; 32],
    /// Statement creation time in Unix epoch seconds.
    pub issued_at: u64,
    /// Exclusive statement expiry in Unix epoch seconds.
    pub expires_at: u64,
    /// Attestor signature over every preceding field.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

/// Fail-closed route-domain attestation validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDomainAttestationValidationError {
    /// The statement contract version is unsupported.
    UnsupportedVersion,
    /// The statement belongs to another Directory Chain.
    WrongChain,
    /// The subject identity is malformed.
    InvalidSubject,
    /// The route-domain token is the reserved zero value.
    InvalidRouteDomain,
    /// The attestor is malformed or self-attests the subject.
    InvalidAttestor,
    /// The issue/expiry interval is malformed, too long, or too far ahead.
    InvalidTimestamp,
    /// The statement is expired at the verifier's current time.
    Expired,
    /// The attestor signature is invalid.
    InvalidSignature,
}

impl std::fmt::Display for RouteDomainAttestationValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported route-domain attestation version",
            Self::WrongChain => "route-domain attestation belongs to another chain",
            Self::InvalidSubject => "route-domain attestation subject is invalid",
            Self::InvalidRouteDomain => "route-domain attestation token is invalid",
            Self::InvalidAttestor => "route-domain attestation signer is invalid",
            Self::InvalidTimestamp => "route-domain attestation interval is invalid",
            Self::Expired => "route-domain attestation is expired",
            Self::InvalidSignature => "route-domain attestation signature is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RouteDomainAttestationValidationError {}

impl RouteDomainAttestationV1 {
    /// Builds and signs one bounded opaque route-domain assignment.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationValidationError`] for a malformed
    /// subject, zero token, self-attestation, or invalid validity interval.
    pub fn new_signed(
        subject_node_id: [u8; 32],
        route_domain: [u8; 16],
        issued_at: u64,
        expires_at: u64,
        attestor: &IdentityKeyPair,
    ) -> Result<Self, RouteDomainAttestationValidationError> {
        let mut attestation = Self {
            protocol_version: ROUTE_DOMAIN_ATTESTATION_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            subject_node_id,
            route_domain,
            attestor_node_id: attestor.public_key_bytes(),
            issued_at,
            expires_at,
            signature: [0u8; 64],
        };
        attestation.validate_unsigned_at(&attestation.chain_id, issued_at)?;
        attestation.signature = attestor.sign(&attestation.signing_bytes());
        Ok(attestation)
    }

    /// Returns the canonical digest signed by the attestor.
    #[must_use]
    pub fn signing_bytes(&self) -> [u8; 32] {
        route_domain_attestation_signing_bytes(
            &self.chain_id,
            &self.subject_node_id,
            &self.route_domain,
            &self.attestor_node_id,
            self.issued_at,
            self.expires_at,
        )
    }

    /// Computes a stable identity including the exact signature.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-RouteDomainAttestationHash-v1");
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }

    /// Verifies structure, bounded validity, identity keys, and signature.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationValidationError`] when any field,
    /// timestamp, chain binding, or signature is invalid.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), RouteDomainAttestationValidationError> {
        self.validate_unsigned_at(expected_chain_id, verifier_observed_at)?;
        IdentityPublicKey::from_bytes(&self.attestor_node_id)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidAttestor)?
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidSignature)
    }

    fn validate_unsigned_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), RouteDomainAttestationValidationError> {
        if self.protocol_version != ROUTE_DOMAIN_ATTESTATION_VERSION_V1 {
            return Err(RouteDomainAttestationValidationError::UnsupportedVersion);
        }
        if &self.chain_id != expected_chain_id {
            return Err(RouteDomainAttestationValidationError::WrongChain);
        }
        IdentityPublicKey::from_bytes(&self.subject_node_id)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidSubject)?;
        if self.route_domain == [0u8; 16] {
            return Err(RouteDomainAttestationValidationError::InvalidRouteDomain);
        }
        IdentityPublicKey::from_bytes(&self.attestor_node_id)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidAttestor)?;
        if self.attestor_node_id == self.subject_node_id {
            return Err(RouteDomainAttestationValidationError::InvalidAttestor);
        }
        let lifetime = self
            .expires_at
            .checked_sub(self.issued_at)
            .ok_or(RouteDomainAttestationValidationError::InvalidTimestamp)?;
        if self.issued_at == 0
            || lifetime == 0
            || lifetime > MAX_ROUTE_DOMAIN_ATTESTATION_LIFETIME_SECS_V1
            || self.issued_at
                > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(RouteDomainAttestationValidationError::InvalidTimestamp);
        }
        if self.expires_at <= verifier_observed_at {
            return Err(RouteDomainAttestationValidationError::Expired);
        }
        Ok(())
    }
}

/// Portable, bounded evidence for one opaque route-domain assignment.
///
/// Every statement retains its original attestor signature. The package has no
/// aggregator signature and carries no network-wide threshold; each verifier
/// must apply its own local pinned-attestor policy with
/// [`Self::verify_with_policy_at`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDomainAttestationCertificateV1 {
    /// Stable portable certificate version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Node identity covered by every statement.
    pub subject_node_id: [u8; 32],
    /// Opaque route-domain token covered by every statement.
    pub route_domain: [u8; 16],
    /// Canonically attestor-sorted signed statements.
    pub attestations: Vec<RouteDomainAttestationV1>,
}

/// Fail-closed portable route-domain certificate validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDomainAttestationCertificateValidationError {
    /// The certificate contract version is unsupported.
    UnsupportedVersion,
    /// The certificate belongs to another Directory Chain.
    WrongChain,
    /// The certificate subject is malformed.
    InvalidSubject,
    /// The certificate route-domain token is malformed.
    InvalidRouteDomain,
    /// The statement count is zero or exceeds the protocol bound.
    InvalidAttestationCount,
    /// Statements are not canonically sorted by attestor identity.
    NonCanonicalAttestorOrder,
    /// The same attestor appears more than once.
    DuplicateAttestor,
    /// A statement does not bind the exact certificate subject and token.
    InvalidAttestationContract,
    /// One statement has an invalid time, identity, chain, or signature.
    InvalidAttestation,
    /// The verifier's local pin set or threshold is malformed.
    InvalidPolicy,
    /// Too few currently valid statements came from locally pinned attestors.
    InsufficientTrustedAttestations,
}

impl std::fmt::Display for RouteDomainAttestationCertificateValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported route-domain certificate version",
            Self::WrongChain => "route-domain certificate belongs to another chain",
            Self::InvalidSubject => "route-domain certificate subject is invalid",
            Self::InvalidRouteDomain => "route-domain certificate token is invalid",
            Self::InvalidAttestationCount => "route-domain certificate statement count is invalid",
            Self::NonCanonicalAttestorOrder => {
                "route-domain certificate attestors are not canonically ordered"
            }
            Self::DuplicateAttestor => "route-domain certificate contains a duplicate attestor",
            Self::InvalidAttestationContract => {
                "route-domain certificate statement contract is invalid"
            }
            Self::InvalidAttestation => "route-domain certificate statement is invalid",
            Self::InvalidPolicy => "route-domain attestor policy is invalid",
            Self::InsufficientTrustedAttestations => {
                "route-domain certificate does not satisfy the pinned attestor threshold"
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RouteDomainAttestationCertificateValidationError {}

impl RouteDomainAttestationCertificateV1 {
    /// Builds a canonical certificate and verifies every statement signature.
    ///
    /// This does not establish trust in the attestors. Call
    /// [`Self::verify_with_policy_at`] with local pins before routing.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationCertificateValidationError`] for an
    /// empty/oversized set, invalid binding, timestamp, identity, or signature.
    pub fn new_verified(
        subject_node_id: [u8; 32],
        route_domain: [u8; 16],
        mut attestations: Vec<RouteDomainAttestationV1>,
        verifier_observed_at: u64,
    ) -> Result<Self, RouteDomainAttestationCertificateValidationError> {
        attestations.sort_unstable_by_key(|attestation| attestation.attestor_node_id);
        let certificate = Self {
            protocol_version: ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            subject_node_id,
            route_domain,
            attestations,
        };
        certificate.verify_at(&certificate.chain_id, verifier_observed_at)?;
        Ok(certificate)
    }

    /// Computes the stable identity of the exact canonical statement set.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-RouteDomainAttestationCertificate-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.subject_node_id);
        hasher.update(self.route_domain);
        hasher.update(
            u64::try_from(self.attestations.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for attestation in &self.attestations {
            hasher.update(attestation.hash());
        }
        hasher.finalize().into()
    }

    /// Verifies the certificate structure and every independent signature.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationCertificateValidationError`] for any
    /// invalid version, chain, binding, ordering, duplicate, time, or signature.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), RouteDomainAttestationCertificateValidationError> {
        if self.protocol_version != ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1 {
            return Err(RouteDomainAttestationCertificateValidationError::UnsupportedVersion);
        }
        if &self.chain_id != expected_chain_id {
            return Err(RouteDomainAttestationCertificateValidationError::WrongChain);
        }
        IdentityPublicKey::from_bytes(&self.subject_node_id)
            .map_err(|_| RouteDomainAttestationCertificateValidationError::InvalidSubject)?;
        if self.route_domain == [0u8; 16] {
            return Err(RouteDomainAttestationCertificateValidationError::InvalidRouteDomain);
        }
        if !(1..=MAX_ROUTE_DOMAIN_ATTESTATIONS_V1).contains(&self.attestations.len()) {
            return Err(RouteDomainAttestationCertificateValidationError::InvalidAttestationCount);
        }
        if self
            .attestations
            .windows(2)
            .any(|items| items[0].attestor_node_id > items[1].attestor_node_id)
        {
            return Err(
                RouteDomainAttestationCertificateValidationError::NonCanonicalAttestorOrder,
            );
        }
        if self
            .attestations
            .windows(2)
            .any(|items| items[0].attestor_node_id == items[1].attestor_node_id)
        {
            return Err(RouteDomainAttestationCertificateValidationError::DuplicateAttestor);
        }
        for attestation in &self.attestations {
            if attestation.chain_id != self.chain_id
                || attestation.subject_node_id != self.subject_node_id
                || attestation.route_domain != self.route_domain
            {
                return Err(
                    RouteDomainAttestationCertificateValidationError::InvalidAttestationContract,
                );
            }
            attestation
                .verify_at(expected_chain_id, verifier_observed_at)
                .map_err(|_| {
                    RouteDomainAttestationCertificateValidationError::InvalidAttestation
                })?;
        }
        Ok(())
    }

    /// Applies one verifier-local pinned-attestor policy.
    ///
    /// The returned count includes only valid statements whose identities are
    /// present in `allowed_attestors`. Unpinned statements remain portable but
    /// do not contribute to the local threshold.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationCertificateValidationError`] when the
    /// certificate is invalid, pins/threshold are malformed, or too few pinned
    /// attestors signed the exact assignment.
    pub fn verify_with_policy_at(
        &self,
        expected_chain_id: &[u8; 32],
        allowed_attestors: &[[u8; 32]],
        minimum_attestors: usize,
        verifier_observed_at: u64,
    ) -> Result<usize, RouteDomainAttestationCertificateValidationError> {
        self.verify_at(expected_chain_id, verifier_observed_at)?;
        let mut canonical_allowed = allowed_attestors.to_vec();
        canonical_allowed.sort_unstable();
        let original_count = canonical_allowed.len();
        canonical_allowed.dedup();
        if canonical_allowed.len() != original_count
            || !(1..=MAX_ROUTE_DOMAIN_ATTESTATIONS_V1).contains(&canonical_allowed.len())
            || minimum_attestors == 0
            || minimum_attestors > canonical_allowed.len()
            || canonical_allowed.iter().any(|attestor| {
                *attestor == [0u8; 32]
                    || *attestor == self.subject_node_id
                    || IdentityPublicKey::from_bytes(attestor).is_err()
            })
        {
            return Err(RouteDomainAttestationCertificateValidationError::InvalidPolicy);
        }
        let trusted = self
            .attestations
            .iter()
            .filter(|attestation| {
                canonical_allowed
                    .binary_search(&attestation.attestor_node_id)
                    .is_ok()
            })
            .count();
        if trusted < minimum_attestors {
            return Err(
                RouteDomainAttestationCertificateValidationError::InsufficientTrustedAttestations,
            );
        }
        Ok(trusted)
    }
}

/// Canonical digest signed by one opaque route-domain attestation.
#[must_use]
pub fn route_domain_attestation_signing_bytes(
    chain_id: &[u8; 32],
    subject_node_id: &[u8; 32],
    route_domain: &[u8; 16],
    attestor_node_id: &[u8; 32],
    issued_at: u64,
    expires_at: u64,
) -> [u8; 32] {
    let issued_at = issued_at.to_le_bytes();
    let expires_at = expires_at.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-RouteDomainAttestation-v1",
        [
            chain_id.as_slice(),
            subject_node_id.as_slice(),
            route_domain.as_slice(),
            attestor_node_id.as_slice(),
            issued_at.as_slice(),
            expires_at.as_slice(),
        ],
    )
}

/// Encodes one canonical bounded portable route-domain certificate.
///
/// Encoding does not establish attestor trust. Call
/// [`RouteDomainAttestationCertificateV1::verify_with_policy_at`] before using
/// a certificate for path admission.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization exceeds the bound
/// or fails.
pub fn encode_route_domain_attestation_certificate(
    certificate: &RouteDomainAttestationCertificateV1,
) -> Result<Vec<u8>, CoreError> {
    let payload =
        encode_bincode_bounded(certificate, MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES)
            .map_err(|error| {
                CoreError::malformed(format!("route-domain certificate encode: {error}"))
            })?;
    let mut frame = Vec::with_capacity(payload.len() + 1);
    frame.push(ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes one canonical bounded portable route-domain certificate.
///
/// Decoding does not establish trust. Call
/// [`RouteDomainAttestationCertificateV1::verify_with_policy_at`] before using
/// the evidence.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` for a wrong magic byte, trailing data,
/// oversized payload, or malformed certificate.
pub fn decode_route_domain_attestation_certificate(
    bytes: &[u8],
) -> Result<RouteDomainAttestationCertificateV1, CoreError> {
    if bytes.first().copied() != Some(ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC) {
        return Err(CoreError::malformed(
            "route-domain attestation certificate magic mismatch",
        ));
    }
    decode_bincode_bounded(
        &bytes[1..],
        MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| CoreError::malformed(format!("route-domain certificate decode: {error}")))
}

#[cfg(test)]
mod tests;
