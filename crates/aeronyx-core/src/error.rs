// ============================================
// File: crates/aeronyx-core/src/error.rs
// ============================================
//! # Core Error Types
//!
//! ## Creation Reason
//! Defines error types specific to protocol and cryptographic operations
//! in the AeroNyx core crate.
//!
//! ## Main Functionality
//! - `CoreError`: Primary error enum for core operations
//! - `CryptoError`: Cryptography-specific errors
//! - `ProtocolError`: Protocol parsing and validation errors
//!
//! ## Error Categories
//! 1. **Crypto Errors**: Key generation, signing, encryption failures
//! 2. **Protocol Errors**: Message parsing, validation, version mismatch
//! 3. **State Errors**: Invalid operation order, missing data
//!
//! ## ⚠️ Important Note for Next Developer
//! - NEVER include key material in error messages
//! - Keep error messages informative but secure
//! - All errors should be loggable without leaking secrets
//!
//! ## Last Modified
//! v0.1.0 - Initial error definitions

use thiserror::Error;

use aeronyx_common::error::CommonError;

// ============================================
// Result Type Alias
// ============================================

/// Result type for core operations.
pub type Result<T> = std::result::Result<T, CoreError>;

// ============================================
// CoreError
// ============================================

/// Core error types for protocol and cryptographic operations.
///
/// # Security Note
/// Error messages are designed to be informative for debugging
/// without revealing sensitive information like key material.
#[derive(Error, Debug)]
pub enum CoreError {
    // ========================================
    // Cryptographic Errors
    // ========================================

    /// Failed to generate cryptographic key.
    #[error("Key generation failed: {context}")]
    KeyGeneration {
        /// What key was being generated
        context: String,
    },

    /// Signature verification failed.
    #[error("Signature verification failed")]
    SignatureVerification,

    /// Signature creation failed.
    #[error("Failed to create signature: {reason}")]
    SignatureCreation {
        /// Why signing failed
        reason: String,
    },

    /// Key exchange operation failed.
    #[error("Key exchange failed: {reason}")]
    KeyExchange {
        /// Why key exchange failed
        reason: String,
    },

    /// Encryption operation failed.
    #[error("Encryption failed: {context}")]
    Encryption {
        /// What was being encrypted
        context: String,
    },

    /// Decryption operation failed (authentication failure).
    #[error("Decryption failed: authentication error")]
    Decryption,

    /// Key derivation failed.
    #[error("Key derivation failed: {reason}")]
    KeyDerivation {
        /// Why derivation failed
        reason: String,
    },

    // ========================================
    // Protocol Errors
    // ========================================

    /// Unknown or unsupported message type.
    #[error("Unknown message type: 0x{0:02x}")]
    UnknownMessageType(u8),

    /// Protocol version mismatch.
    #[error("Unsupported protocol version: {got}, expected {expected}")]
    UnsupportedVersion {
        /// Version received
        got: u8,
        /// Version expected
        expected: u8,
    },

    /// Message is malformed or truncated.
    #[error("Malformed message: {reason}")]
    MalformedMessage {
        /// What's wrong with the message
        reason: String,
    },

    /// Message is too short to be valid.
    #[error("Message too short: expected at least {expected} bytes, got {actual}")]
    MessageTooShort {
        /// Minimum expected length
        expected: usize,
        /// Actual length received
        actual: usize,
    },

    /// Message exceeds maximum allowed size.
    #[error("Message too large: max {max} bytes, got {actual}")]
    MessageTooLarge {
        /// Maximum allowed size
        max: usize,
        /// Actual size received
        actual: usize,
    },

    /// Timestamp validation failed.
    #[error("Invalid timestamp: {reason}")]
    InvalidTimestamp {
        /// Why timestamp is invalid
        reason: String,
    },

    /// Replay attack detected (counter not advancing).
    #[error("Replay detected: counter {received} not greater than {expected}")]
    ReplayDetected {
        /// Counter value received
        received: u64,
        /// Minimum expected counter
        expected: u64,
    },

    // ========================================
    // State Errors
    // ========================================

    /// Operation not valid in current state.
    #[error("Invalid state for operation: {operation} requires {required_state}")]
    InvalidState {
        /// What operation was attempted
        operation: String,
        /// What state was required
        required_state: String,
    },

    /// Required data is missing.
    #[error("Missing required data: {field}")]
    MissingData {
        /// What data is missing
        field: String,
    },

    // ========================================
    // Wrapped Errors
    // ========================================

    /// Error from common crate.
    #[error(transparent)]
    Common(#[from] CommonError),
}

impl CoreError {
    // ========================================
    // Convenience Constructors
    // ========================================

    /// Creates a `KeyGeneration` error.
    pub fn key_generation(context: impl Into<String>) -> Self {
        Self::KeyGeneration {
            context: context.into(),
        }
    }

    /// Creates a `MalformedMessage` error.
    pub fn malformed(reason: impl Into<String>) -> Self {
        Self::MalformedMessage {
            reason: reason.into(),
        }
    }

    /// Creates a `MessageTooShort` error.
    pub const fn too_short(expected: usize, actual: usize) -> Self {
        Self::MessageTooShort { expected, actual }
    }

    /// Creates an `InvalidTimestamp` error.
    pub fn invalid_timestamp(reason: impl Into<String>) -> Self {
        Self::InvalidTimestamp {
            reason: reason.into(),
        }
    }

    /// Creates a `ReplayDetected` error.
    pub const fn replay(received: u64, expected: u64) -> Self {
        Self::ReplayDetected { received, expected }
    }

    /// Creates an `InvalidState` error.
    pub fn invalid_state(
        operation: impl Into<String>,
        required_state: impl Into<String>,
    ) -> Self {
        Self::InvalidState {
            operation: operation.into(),
            required_state: required_state.into(),
        }
    }

    // ========================================
    // Error Classification
    // ========================================

    /// Returns `true` if this is a cryptographic error.
    ///
    /// Crypto errors might indicate an attack or implementation bug.
    #[must_use]
    pub const fn is_crypto_error(&self) -> bool {
        matches!(
            self,
            Self::KeyGeneration { .. }
                | Self::SignatureVerification
                | Self::SignatureCreation { .. }
                | Self::KeyExchange { .. }
                | Self::Encryption { .. }
                | Self::Decryption
                | Self::KeyDerivation { .. }
        )
    }

    /// Returns `true` if this is a protocol error.
    ///
    /// Protocol errors indicate malformed or invalid messages.
    #[must_use]
    pub const fn is_protocol_error(&self) -> bool {
        matches!(
            self,
            Self::UnknownMessageType(_)
                | Self::UnsupportedVersion { .. }
                | Self::MalformedMessage { .. }
                | Self::MessageTooShort { .. }
                | Self::MessageTooLarge { .. }
                | Self::InvalidTimestamp { .. }
                | Self::ReplayDetected { .. }
        )
    }

    /// Returns `true` if this error might indicate an attack.
    ///
    /// These errors warrant additional logging/monitoring.
    #[must_use]
    pub const fn is_suspicious(&self) -> bool {
        matches!(
            self,
            Self::SignatureVerification
                | Self::Decryption
                | Self::ReplayDetected { .. }
                | Self::InvalidTimestamp { .. }
        )
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::SignatureVerification;
        assert!(err.to_string().contains("Signature"));

        let err = CoreError::too_short(100, 50);
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_error_classification() {
        assert!(CoreError::SignatureVerification.is_crypto_error());
        assert!(CoreError::SignatureVerification.is_suspicious());
        
        assert!(CoreError::UnknownMessageType(0xFF).is_protocol_error());
        
        let replay = CoreError::replay(5, 10);
        assert!(replay.is_protocol_error());
        assert!(replay.is_suspicious());
    }

    #[test]
    fn test_common_error_conversion() {
        let common = CommonError::invalid_input("field", "bad value");
        let core: CoreError = common.into();
        assert!(matches!(core, CoreError::Common(_)));
    }
}
