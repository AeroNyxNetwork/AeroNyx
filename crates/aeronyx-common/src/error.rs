// ============================================
// File: crates/aeronyx-common/src/error.rs
// ============================================
//! # Common Error Types
//!
//! ## Creation Reason
//! Provides foundational error types and result aliases used across
//! all AeroNyx crates, enabling consistent error handling.
//!
//! ## Main Functionality
//! - `CommonError`: Base error enum for common operations
//! - `Result<T>`: Type alias using `CommonError`
//! - Error conversion traits for interoperability
//!
//! ## Design Philosophy
//! - Use `thiserror` for ergonomic error definitions
//! - Each crate may define its own error types that wrap `CommonError`
//! - Errors should be informative without leaking sensitive information
//!
//! ## ⚠️ Important Note for Next Developer
//! - Never include sensitive data (keys, IPs) in error messages
//! - Keep error variants specific but not too granular
//! - Implement `From` traits for seamless error propagation
//!
//! ## Last Modified
//! v0.1.0 - Initial error definitions

use std::fmt;
use thiserror::Error;

// ============================================
// Result Type Alias
// ============================================

/// Common result type for operations that may fail.
pub type Result<T> = std::result::Result<T, CommonError>;

// ============================================
// CommonError
// ============================================

/// Common error types shared across AeroNyx crates.
///
/// # Categories
/// - **Validation**: Input validation failures
/// - **IO**: System I/O errors
/// - **Encoding**: Serialization/deserialization errors
/// - **Internal**: Unexpected internal state
///
/// # Example
/// ```
/// use aeronyx_common::error::{CommonError, Result};
///
/// fn validate_input(data: &[u8]) -> Result<()> {
///     if data.is_empty() {
///         return Err(CommonError::InvalidInput {
///             field: "data".into(),
///             reason: "cannot be empty".into(),
///         });
///     }
///     Ok(())
/// }
/// ```
#[derive(Error, Debug)]
pub enum CommonError {
    // ========================================
    // Validation Errors
    // ========================================
    
    /// Invalid input data provided.
    #[error("Invalid input for '{field}': {reason}")]
    InvalidInput {
        /// Name of the field or parameter
        field: String,
        /// Description of what's wrong
        reason: String,
    },

    /// Data length doesn't match expected size.
    #[error("Invalid length: expected {expected}, got {actual}")]
    InvalidLength {
        /// Expected length in bytes
        expected: usize,
        /// Actual length received
        actual: usize,
    },

    /// Value is out of acceptable range.
    #[error("Value out of range: {value} not in [{min}, {max}]")]
    OutOfRange {
        /// The value that was out of range
        value: String,
        /// Minimum acceptable value
        min: String,
        /// Maximum acceptable value
        max: String,
    },

    // ========================================
    // Resource Errors
    // ========================================

    /// Requested resource was not found.
    #[error("Resource not found: {resource_type} with id '{id}'")]
    NotFound {
        /// Type of resource (e.g., "session", "route")
        resource_type: String,
        /// Identifier that wasn't found
        id: String,
    },

    /// Resource already exists.
    #[error("Resource already exists: {resource_type} with id '{id}'")]
    AlreadyExists {
        /// Type of resource
        resource_type: String,
        /// Identifier that already exists
        id: String,
    },

    /// Resource limit exceeded.
    #[error("Resource exhausted: {resource} (limit: {limit})")]
    ResourceExhausted {
        /// Name of the resource
        resource: String,
        /// The limit that was exceeded
        limit: String,
    },

    // ========================================
    // IO Errors
    // ========================================

    /// System I/O error occurred.
    #[error("I/O error: {context}")]
    Io {
        /// What operation was being performed
        context: String,
        /// Underlying IO error
        #[source]
        source: std::io::Error,
    },

    // ========================================
    // Encoding Errors
    // ========================================

    /// Failed to encode/serialize data.
    #[error("Encoding error: {context}")]
    Encoding {
        /// What was being encoded
        context: String,
        /// Error details
        details: String,
    },

    /// Failed to decode/deserialize data.
    #[error("Decoding error: {context}")]
    Decoding {
        /// What was being decoded
        context: String,
        /// Error details
        details: String,
    },

    // ========================================
    // State Errors
    // ========================================

    /// Operation not valid in current state.
    #[error("Invalid state: expected {expected}, found {current}")]
    InvalidState {
        /// Expected state
        expected: String,
        /// Current state
        current: String,
    },

    /// Operation timed out.
    #[error("Operation timed out: {operation} after {duration_ms}ms")]
    Timeout {
        /// What operation timed out
        operation: String,
        /// How long we waited
        duration_ms: u64,
    },

    // ========================================
    // Internal Errors
    // ========================================

    /// Internal error (bug or unexpected condition).
    #[error("Internal error: {message}")]
    Internal {
        /// Description of what went wrong
        message: String,
    },
}

impl CommonError {
    // ========================================
    // Convenience Constructors
    // ========================================

    /// Creates an `InvalidInput` error.
    pub fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Creates an `InvalidLength` error.
    pub const fn invalid_length(expected: usize, actual: usize) -> Self {
        Self::InvalidLength { expected, actual }
    }

    /// Creates a `NotFound` error.
    pub fn not_found(resource_type: impl Into<String>, id: impl Into<String>) -> Self {
        Self::NotFound {
            resource_type: resource_type.into(),
            id: id.into(),
        }
    }

    /// Creates an `Io` error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Creates an `Internal` error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Creates a `ResourceExhausted` error.
    pub fn resource_exhausted(
        resource: impl Into<String>,
        limit: impl fmt::Display,
    ) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            limit: limit.to_string(),
        }
    }

    /// Creates a `Timeout` error.
    pub fn timeout(operation: impl Into<String>, duration_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            duration_ms,
        }
    }

    // ========================================
    // Error Classification
    // ========================================

    /// Returns `true` if this error is retryable.
    ///
    /// Retryable errors are transient and the operation might
    /// succeed if attempted again.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Io { .. } | Self::Timeout { .. } | Self::ResourceExhausted { .. }
        )
    }

    /// Returns `true` if this error indicates a client mistake.
    ///
    /// Client errors are caused by invalid input or requests,
    /// not by server-side issues.
    #[must_use]
    pub const fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidInput { .. }
                | Self::InvalidLength { .. }
                | Self::OutOfRange { .. }
                | Self::NotFound { .. }
                | Self::InvalidState { .. }
        )
    }

    /// Returns `true` if this error indicates a server-side issue.
    #[must_use]
    pub const fn is_server_error(&self) -> bool {
        matches!(self, Self::Internal { .. })
    }
}

// ============================================
// Error Conversions
// ============================================

impl From<std::io::Error> for CommonError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            context: "unspecified I/O operation".into(),
            source: err,
        }
    }
}

impl From<base64::DecodeError> for CommonError {
    fn from(err: base64::DecodeError) -> Self {
        Self::Decoding {
            context: "base64 decode".into(),
            details: err.to_string(),
        }
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
        let err = CommonError::invalid_input("session_id", "must be 16 bytes");
        assert!(err.to_string().contains("session_id"));
        assert!(err.to_string().contains("16 bytes"));
    }

    #[test]
    fn test_error_classification() {
        let client_err = CommonError::invalid_input("field", "bad");
        assert!(client_err.is_client_error());
        assert!(!client_err.is_server_error());
        assert!(!client_err.is_retryable());

        let server_err = CommonError::internal("bug");
        assert!(server_err.is_server_error());
        assert!(!server_err.is_client_error());

        let retryable = CommonError::timeout("handshake", 5000);
        assert!(retryable.is_retryable());
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        );
        let common_err: CommonError = io_err.into();
        assert!(matches!(common_err, CommonError::Io { .. }));
    }
}
