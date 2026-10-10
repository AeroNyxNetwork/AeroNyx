// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/terminal_failure.rs
// ============================================
//! # Encrypted terminal failures
//!
//! Owns the stable one-byte `BlindVaultTerminalOperation` and
//! `BlindVaultTerminalFailureCode` values and the minimal
//! `BlindVaultTerminalFailure` body carried inside a sealed onion reply.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Intra-doc link target only: [`OnionReplySession`] in the docs below.
#[cfg(doc)]
use crate::protocol::onion_reply::OnionReplySession;

/// Blind Vault operation answered by one encrypted terminal failure.
///
/// [BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex] Stable one-byte
/// operation values keep the bincode body independent from Rust enum ordering.
/// They identify only the operation already bound into the signed onion reply;
/// lease, object, route, endpoint, and application metadata remain absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlindVaultTerminalOperation {
    /// Immutable ciphertext write.
    Put = 1,
    /// Capability-authenticated ciphertext recovery.
    Pull = 2,
    /// Administration-authorized object deletion.
    Delete = 3,
    /// Blind-issued anonymous lease admission.
    LeaseAdmission = 4,
    /// Blind-authorized lease renewal.
    LeaseRenewal = 5,
    /// Private lease status observation.
    LeaseStatus = 6,
    /// Private encrypted-object inventory observation.
    LeaseInventory = 7,
    /// Complete anonymous lease retirement.
    LeaseRetire = 8,
}

impl TryFrom<u8> for BlindVaultTerminalOperation {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Put),
            2 => Ok(Self::Pull),
            3 => Ok(Self::Delete),
            4 => Ok(Self::LeaseAdmission),
            5 => Ok(Self::LeaseRenewal),
            6 => Ok(Self::LeaseStatus),
            7 => Ok(Self::LeaseInventory),
            8 => Ok(Self::LeaseRetire),
            _ => Err("unknown blind vault terminal operation"),
        }
    }
}

impl Serialize for BlindVaultTerminalOperation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for BlindVaultTerminalOperation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

/// Stable source-action class for an encrypted terminal failure.
///
/// Codes are deliberately coarse. In particular, every capability, lease,
/// signature, cursor, object-state, and replay rejection collapses to
/// [`Self::Rejected`] so a client cannot turn the storage node into an oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlindVaultTerminalFailureCode {
    /// The request is invalid or permanently unauthorized; do not retry it.
    Rejected = 1,
    /// The selected replica lacks capacity; choose another audited terminal.
    Capacity = 2,
    /// The terminal is temporarily unavailable; bounded retry is permitted.
    Unavailable = 3,
    /// The selected inline response class cannot carry the requested result.
    ResponseTooLarge = 4,
}

impl BlindVaultTerminalFailureCode {
    /// Stable operator-safe label without request or storage details.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rejected => "rejected",
            Self::Capacity => "capacity",
            Self::Unavailable => "unavailable",
            Self::ResponseTooLarge => "response_too_large",
        }
    }
}

impl std::fmt::Display for BlindVaultTerminalFailureCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl TryFrom<u8> for BlindVaultTerminalFailureCode {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Rejected),
            2 => Ok(Self::Capacity),
            3 => Ok(Self::Unavailable),
            4 => Ok(Self::ResponseTooLarge),
            _ => Err("unknown blind vault terminal failure code"),
        }
    }
}

impl Serialize for BlindVaultTerminalFailureCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for BlindVaultTerminalFailureCode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

/// Authenticated workload failure carried inside a sealed onion reply.
///
/// The outer [`OnionReplySession`] verifies the terminal signature and binds
/// this body to the exact request commitment before a caller can inspect it.
/// No separate inner signature or request identifier is needed, and adding
/// either would create unnecessary correlation material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultTerminalFailure {
    operation: BlindVaultTerminalOperation,
    code: BlindVaultTerminalFailureCode,
}

impl BlindVaultTerminalFailure {
    /// Creates one minimal encrypted terminal failure.
    #[must_use]
    pub const fn new(
        operation: BlindVaultTerminalOperation,
        code: BlindVaultTerminalFailureCode,
    ) -> Self {
        Self { operation, code }
    }

    /// Operation already bound into the encrypted request and signed reply.
    #[must_use]
    pub const fn operation(&self) -> BlindVaultTerminalOperation {
        self.operation
    }

    /// Coarse source action; never a storage-engine or authorization detail.
    #[must_use]
    pub const fn code(&self) -> BlindVaultTerminalFailureCode {
        self.code
    }
}
