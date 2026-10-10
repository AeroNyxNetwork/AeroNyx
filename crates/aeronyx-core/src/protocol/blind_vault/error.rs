// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/error.rs
// ============================================
//! # Blind Vault validation and codec errors
//!
//! Owns `BlindVaultError`, the shared validation and wire-codec failure type.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use thiserror::Error;

/// Validation and bounded wire-codec failures for Blind Vault v1.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultError {
    /// Frame or object uses a version unsupported by this implementation.
    #[error("unsupported blind-vault protocol version {0}")]
    UnsupportedVersion(u16),
    /// A security-sensitive random identifier was the all-zero sentinel.
    #[error("{0} must not be all zero")]
    ZeroIdentifier(&'static str),
    /// Ciphertext did not use one of the protocol's coarse padding classes.
    #[error("ciphertext length {actual} is not an allowed padded size class")]
    InvalidCiphertextSize {
        /// Received ciphertext length in bytes.
        actual: usize,
    },
    /// Declared commitment did not match the exact ciphertext bytes.
    #[error("ciphertext commitment does not match ciphertext")]
    CommitmentMismatch,
    /// Requested expiry was not later than node time.
    #[error("object expiry is not in the future")]
    Expired,
    /// Requested retention exceeded the accepting node's policy.
    #[error("object lifetime exceeds node policy")]
    LifetimeTooLong,
    /// Ed25519 verification failed.
    #[error("signature verification failed")]
    InvalidSignature,
    /// Embedded Ed25519 key bytes were not a valid public key.
    #[error("invalid blind-vault public key")]
    InvalidPublicKey,
    /// Lease reused one key for write and administration authority.
    #[error("lease write and administration keys must be distinct")]
    LeaseKeyReuse,
    /// Lease request signer did not match its declared administration key.
    #[error("lease administration identity does not match its signing key")]
    AdminIdentityMismatch,
    /// A signed mutation timestamp was outside the configured replay window.
    #[error("request timestamp is outside the accepted clock-skew window")]
    RequestTimestampOutsideWindow,
    /// A lease-retirement receipt carried inconsistent aggregate deletion data.
    #[error("lease retirement receipt summary is inconsistent")]
    InvalidRetirementSummary,
    /// A lease renewal did not strictly extend one currently live generation.
    #[error("lease renewal window is invalid")]
    InvalidLeaseRenewalWindow,
    /// A signed lease-status receipt carried impossible live-usage data.
    #[error("lease status receipt summary is inconsistent")]
    InvalidLeaseStatusSummary,
    /// A private inventory entry violated its fixed-width commitment contract.
    #[error("blind vault inventory entry is invalid")]
    InvalidInventoryEntry,
    /// Inventory entries were duplicated or not in canonical object-id order.
    #[error("blind vault inventory entries are not in strict object-id order")]
    InvalidInventoryOrder,
    /// Inventory aggregate counters exceeded their fixed-width representation.
    #[error("blind vault inventory aggregate overflowed")]
    InventoryOverflow,
    /// A signed inventory receipt carried impossible aggregate data.
    #[error("blind vault inventory receipt summary is inconsistent")]
    InvalidInventorySummary,
    /// Receipt signer did not match the declared descriptor identity.
    #[error("receipt node identity does not match its signing key")]
    NodeIdentityMismatch,
    /// Admission ticket signer did not match its declared issuer identity.
    #[error("admission issuer identity does not match its signing key")]
    AdmissionIssuerMismatch,
    /// Admission ticket cannot be redeemed before its validity window.
    #[error("admission ticket is not yet valid")]
    AdmissionNotYetValid,
    /// Admission ticket carried an invalid time or lease policy.
    #[error("admission ticket policy is invalid")]
    InvalidAdmissionPolicy,
    /// Blind-admission credential version is unsupported.
    #[error("unsupported blind-admission credential version {0}")]
    UnsupportedBlindAdmissionVersion(u16),
    /// Finalized RSA signature did not fit the RSA-2048 through RSA-4096 bound.
    #[error("blind-admission signature length {actual} is invalid")]
    InvalidBlindAdmissionSignatureLength {
        /// Received finalized signature length in bytes.
        actual: usize,
    },
    /// An advertised RSA public key exceeded the issuer-directory bound.
    #[error("blind issuer public-key DER length {actual} is invalid")]
    InvalidBlindIssuerKeyLength {
        /// Received canonical DER length in bytes.
        actual: usize,
    },
    /// Advertised RSA key bytes did not match their signed key fingerprint.
    #[error("blind issuer key fingerprint does not match public-key DER")]
    BlindIssuerKeyIdMismatch,
    /// An issuer epoch had an empty, expired, reversed, or overlong policy.
    #[error("blind issuer epoch policy is invalid")]
    InvalidBlindIssuerEpochPolicy,
    /// A signed directory exceeded the protocol-wide epoch count ceiling.
    #[error("blind issuer directory contains too many epochs")]
    TooManyBlindIssuerEpochs,
    /// Epochs were duplicated or not in canonical key-ID order.
    #[error("blind issuer epochs are not in strict key-ID order")]
    BlindIssuerEpochOrderInvalid,
    /// Signed issuer discovery data was stale or implausibly far in the future.
    #[error("blind issuer directory timestamp is outside the accepted window")]
    IssuerDirectoryTimestampOutsideWindow,
    /// An authority update used generation zero, which is reserved for local
    /// static bootstrap state.
    #[error("blind issuer update generation must start at one")]
    InvalidBlindIssuerUpdateGeneration,
    /// An authority update attempted to install no usable issuer epochs.
    #[error("blind issuer update contains no epochs")]
    BlindIssuerUpdateHasNoEpochs,
    /// Declared update authority did not match the pinned verification key.
    #[error("blind issuer update authority does not match its signing key")]
    BlindIssuerAuthorityMismatch,
    /// Authority update was stale or implausibly far in the future.
    #[error("blind issuer update timestamp is outside the accepted window")]
    BlindIssuerUpdateTimestampOutsideWindow,
    /// Pull page size was zero or exceeded the protocol-wide ceiling.
    #[error("blind-vault pull limit is invalid")]
    InvalidPullLimit,
    /// Opaque pull cursor exceeded the protocol-wide ceiling.
    #[error("blind-vault pull cursor length is invalid")]
    InvalidPullCursorLength,
    /// Receipt promised no positive retention interval.
    #[error("receipt storage window is invalid")]
    InvalidReceiptWindow,
    /// Frame did not start with the Blind Vault magic bytes.
    #[error("invalid blind-vault frame magic")]
    InvalidMagic,
    /// Frame did not contain a complete header.
    #[error("blind-vault frame is truncated")]
    TruncatedFrame,
    /// Encoded frame exceeded the hard protocol allocation limit.
    #[error("blind-vault frame exceeds the protocol limit")]
    FrameTooLarge,
    /// Frame kind is not implemented by this protocol version.
    #[error("unknown blind-vault frame kind {0}")]
    UnknownFrameKind(u8),
    /// A typed value could not be serialized within the protocol bound.
    #[error("blind-vault frame serialization failed")]
    Serialization,
    /// A frame body was malformed, oversized, or contained trailing bytes.
    #[error("blind-vault frame deserialization failed")]
    Deserialization,
}
