// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/mailbox_work_policy.rs
// ============================================
//! # Anonymous Mailbox admission-work policy
//!
//! Owns the target-signed, descriptor-bound admission-ticket proof-of-work
//! policy: its frozen constants, fixed-width codec, build-metadata token
//! embedding helpers, and fail-closed verification against one descriptor.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};

use super::descriptor::NodeDescriptor;
use super::protocol_feature::NodeProtocolFeature;

// [ANONYMOUS-MAILBOX-WORK-POLICY 2026-09-07 by Codex] Keep the public work
// policy inside the already signed SemVer metadata field, with a fixed codec
// that upgraded clients can validate without changing descriptor schema v2.
const ANONYMOUS_MAILBOX_WORK_POLICY_MAGIC: [u8; 4] = *b"AMWP";
const ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_PREFIX_V1: &str = "anmp1-";
const ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_FAMILY: &str = "anmp";
const ANONYMOUS_MAILBOX_WORK_POLICY_SIGNATURE_DOMAIN_V1: &[u8] =
    b"AeroNyx-AnonymousMailbox-WorkPolicy-v1";
/// Frozen version of the signed Anonymous Mailbox work-policy codec.
pub const ANONYMOUS_MAILBOX_WORK_POLICY_VERSION_V1: u16 = 1;
/// Exact encoded bytes in one signed Anonymous Mailbox work-policy token.
pub const ANONYMOUS_MAILBOX_WORK_POLICY_WIRE_BYTES_V1: usize = 127;
/// Minimum accepted target-bound admission-ticket proof-of-work bits.
pub const MIN_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1: u8 = 1;
/// Maximum accepted target-bound admission-ticket proof-of-work bits.
pub const MAX_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1: u8 = 24;

// ============================================
// Anonymous Mailbox admission-work policy
// ============================================

/// Coarse validation failures for signed Anonymous Mailbox work-policy data.
///
/// Variants deliberately omit node ids, signatures, and descriptor contents so
/// callers can fail closed without leaking routing metadata into diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymousMailboxWorkPolicyError {
    /// A descriptor does not carry the required policy token.
    MissingPolicy,
    /// The policy token or its fixed-width payload is not canonical.
    MalformedPolicy,
    /// The token names an unsupported work-policy version.
    UnsupportedVersion,
    /// More than one policy exists or authenticated claims do not agree.
    ClaimsConflict,
    /// The policy or its enclosing descriptor is outside its validity window.
    NotCurrentlyValid,
    /// A descriptor or nested work-policy signature did not verify.
    SignatureRejected,
}

impl std::fmt::Display for AnonymousMailboxWorkPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::MissingPolicy => "anonymous mailbox work policy is missing",
            Self::MalformedPolicy => "anonymous mailbox work policy is malformed",
            Self::UnsupportedVersion => "anonymous mailbox work policy version is unsupported",
            Self::ClaimsConflict => "anonymous mailbox work policy claims conflict",
            Self::NotCurrentlyValid => "anonymous mailbox work policy is not currently valid",
            Self::SignatureRejected => "anonymous mailbox work policy signature was rejected",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for AnonymousMailboxWorkPolicyError {}

/// Target-signed, descriptor-bound admission-ticket proof-of-work policy.
///
/// The fixed representation is embedded as lowercase hexadecimal in one
/// SemVer build-metadata identifier. Fields stay private so construction cannot
/// bypass the canonical transcript, range checks, or descriptor binding.
#[derive(Clone, PartialEq, Eq)]
pub struct SignedAnonymousMailboxWorkPolicyV1 {
    target_node_id: [u8; 32],
    descriptor_sequence: u64,
    issued_at: u64,
    expires_at: u64,
    work_bits: u8,
    signature: [u8; 64],
}

impl std::fmt::Debug for SignedAnonymousMailboxWorkPolicyV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SignedAnonymousMailboxWorkPolicyV1")
            .field("descriptor_sequence", &self.descriptor_sequence)
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .field("work_bits", &self.work_bits)
            .finish_non_exhaustive()
    }
}

impl SignedAnonymousMailboxWorkPolicyV1 {
    /// Returns the target node identity authenticated by the nested signature.
    #[must_use]
    pub const fn target_node_id(&self) -> [u8; 32] {
        self.target_node_id
    }

    /// Returns the exact enclosing descriptor sequence.
    #[must_use]
    pub const fn descriptor_sequence(&self) -> u64 {
        self.descriptor_sequence
    }

    /// Returns the policy issue time in Unix epoch seconds.
    #[must_use]
    pub const fn issued_at(&self) -> u64 {
        self.issued_at
    }

    /// Returns the policy expiry time in Unix epoch seconds.
    #[must_use]
    pub const fn expires_at(&self) -> u64 {
        self.expires_at
    }

    /// Returns the required target-bound proof-of-work bits.
    #[must_use]
    pub const fn work_bits(&self) -> u8 {
        self.work_bits
    }

    fn signing_bytes(
        target_node_id: [u8; 32],
        descriptor_sequence: u64,
        issued_at: u64,
        expires_at: u64,
        work_bits: u8,
    ) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            ANONYMOUS_MAILBOX_WORK_POLICY_SIGNATURE_DOMAIN_V1.len() + 2 + 32 + 8 + 8 + 8 + 1,
        );
        bytes.extend_from_slice(ANONYMOUS_MAILBOX_WORK_POLICY_SIGNATURE_DOMAIN_V1);
        bytes.extend_from_slice(&ANONYMOUS_MAILBOX_WORK_POLICY_VERSION_V1.to_le_bytes());
        bytes.extend_from_slice(&target_node_id);
        bytes.extend_from_slice(&descriptor_sequence.to_le_bytes());
        bytes.extend_from_slice(&issued_at.to_le_bytes());
        bytes.extend_from_slice(&expires_at.to_le_bytes());
        bytes.push(work_bits);
        bytes
    }

    pub(super) fn issue(
        descriptor: &NodeDescriptor,
        work_bits: u8,
        identity: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxWorkPolicyError> {
        if descriptor.node_id != identity.public_key_bytes()
            || descriptor.sequence == 0
            || descriptor.issued_at >= descriptor.expires_at
        {
            return Err(AnonymousMailboxWorkPolicyError::ClaimsConflict);
        }
        if !descriptor.advertises_protocol_feature(NodeProtocolFeature::AnonymousMailboxV1) {
            return Err(AnonymousMailboxWorkPolicyError::ClaimsConflict);
        }
        if !(MIN_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1..=MAX_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1)
            .contains(&work_bits)
        {
            return Err(AnonymousMailboxWorkPolicyError::MalformedPolicy);
        }
        let signature = identity.sign(&Self::signing_bytes(
            descriptor.node_id,
            descriptor.sequence,
            descriptor.issued_at,
            descriptor.expires_at,
            work_bits,
        ));
        Ok(Self {
            target_node_id: descriptor.node_id,
            descriptor_sequence: descriptor.sequence,
            issued_at: descriptor.issued_at,
            expires_at: descriptor.expires_at,
            work_bits,
            signature,
        })
    }

    fn encode_fixed(&self) -> [u8; ANONYMOUS_MAILBOX_WORK_POLICY_WIRE_BYTES_V1] {
        let mut encoded = [0u8; ANONYMOUS_MAILBOX_WORK_POLICY_WIRE_BYTES_V1];
        encoded[..4].copy_from_slice(&ANONYMOUS_MAILBOX_WORK_POLICY_MAGIC);
        encoded[4..6].copy_from_slice(&ANONYMOUS_MAILBOX_WORK_POLICY_VERSION_V1.to_le_bytes());
        encoded[6..38].copy_from_slice(&self.target_node_id);
        encoded[38..46].copy_from_slice(&self.descriptor_sequence.to_le_bytes());
        encoded[46..54].copy_from_slice(&self.issued_at.to_le_bytes());
        encoded[54..62].copy_from_slice(&self.expires_at.to_le_bytes());
        encoded[62] = self.work_bits;
        encoded[63..].copy_from_slice(&self.signature);
        encoded
    }

    pub(super) fn semver_build_token(&self) -> String {
        format!(
            "{ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_PREFIX_V1}{}",
            hex::encode(self.encode_fixed())
        )
    }

    pub(super) fn decode_semver_build_token(
        token: &str,
    ) -> Result<Self, AnonymousMailboxWorkPolicyError> {
        if !token.starts_with(ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_PREFIX_V1) {
            if token
                .get(..ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_FAMILY.len())
                .is_some_and(|prefix| {
                    prefix.eq_ignore_ascii_case(ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_FAMILY)
                })
            {
                return if token.starts_with("anmp") {
                    Err(AnonymousMailboxWorkPolicyError::UnsupportedVersion)
                } else {
                    Err(AnonymousMailboxWorkPolicyError::MalformedPolicy)
                };
            }
            return Err(AnonymousMailboxWorkPolicyError::MalformedPolicy);
        }
        let hex_payload = &token[ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_PREFIX_V1.len()..];
        if hex_payload.len() != ANONYMOUS_MAILBOX_WORK_POLICY_WIRE_BYTES_V1 * 2
            || !hex_payload
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(AnonymousMailboxWorkPolicyError::MalformedPolicy);
        }
        let encoded = hex::decode(hex_payload)
            .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?;
        if encoded[..4] != ANONYMOUS_MAILBOX_WORK_POLICY_MAGIC {
            return Err(AnonymousMailboxWorkPolicyError::MalformedPolicy);
        }
        let version = u16::from_le_bytes(
            encoded[4..6]
                .try_into()
                .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?,
        );
        if version != ANONYMOUS_MAILBOX_WORK_POLICY_VERSION_V1 {
            return Err(AnonymousMailboxWorkPolicyError::UnsupportedVersion);
        }
        let target_node_id = encoded[6..38]
            .try_into()
            .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?;
        let descriptor_sequence = u64::from_le_bytes(
            encoded[38..46]
                .try_into()
                .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?,
        );
        let issued_at = u64::from_le_bytes(
            encoded[46..54]
                .try_into()
                .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?,
        );
        let expires_at = u64::from_le_bytes(
            encoded[54..62]
                .try_into()
                .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?,
        );
        let work_bits = encoded[62];
        let signature = encoded[63..]
            .try_into()
            .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?;
        Ok(Self {
            target_node_id,
            descriptor_sequence,
            issued_at,
            expires_at,
            work_bits,
            signature,
        })
    }

    pub(super) fn verify_for_descriptor(
        &self,
        descriptor: &NodeDescriptor,
        now: u64,
    ) -> Result<(), AnonymousMailboxWorkPolicyError> {
        if self.target_node_id != descriptor.node_id
            || self.descriptor_sequence != descriptor.sequence
            || self.issued_at != descriptor.issued_at
            || self.expires_at != descriptor.expires_at
        {
            return Err(AnonymousMailboxWorkPolicyError::ClaimsConflict);
        }
        if self.descriptor_sequence == 0
            || self.issued_at >= self.expires_at
            || !(MIN_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1
                ..=MAX_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1)
                .contains(&self.work_bits)
        {
            return Err(AnonymousMailboxWorkPolicyError::MalformedPolicy);
        }
        if !descriptor.advertises_protocol_feature(NodeProtocolFeature::AnonymousMailboxV1) {
            return Err(AnonymousMailboxWorkPolicyError::ClaimsConflict);
        }
        if now < self.issued_at || now >= self.expires_at {
            return Err(AnonymousMailboxWorkPolicyError::NotCurrentlyValid);
        }
        let identity = IdentityPublicKey::from_bytes(&self.target_node_id)
            .map_err(|_| AnonymousMailboxWorkPolicyError::SignatureRejected)?;
        identity
            .verify(
                &Self::signing_bytes(
                    self.target_node_id,
                    self.descriptor_sequence,
                    self.issued_at,
                    self.expires_at,
                    self.work_bits,
                ),
                &self.signature,
            )
            .map_err(|_| AnonymousMailboxWorkPolicyError::SignatureRejected)
    }
}

pub(super) fn append_semver_build_identifier(version: &str, identifier: &str) -> String {
    if version.split_once('+').is_some() {
        format!("{version}.{identifier}")
    } else {
        format!("{version}+{identifier}")
    }
}

pub(super) fn anonymous_mailbox_work_policy_tokens(version: &str) -> impl Iterator<Item = &str> {
    version
        .split_once('+')
        .map(|(_, metadata)| metadata)
        .unwrap_or_default()
        .split('.')
        .filter(|identifier| {
            identifier
                .get(..ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_FAMILY.len())
                .is_some_and(|prefix| {
                    prefix.eq_ignore_ascii_case(ANONYMOUS_MAILBOX_WORK_POLICY_TOKEN_FAMILY)
                })
        })
}

#[cfg(test)]
mod tests;
