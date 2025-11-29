// ============================================
// File: crates/aeronyx-transport/src/error.rs
// ============================================
//! # Transport Error Types
//!
//! ## Creation Reason
//! Defines error types specific to transport layer operations
//! including network I/O and TUN device errors.
//!
//! ## Main Functionality
//! - `TransportError`: Primary error enum for transport operations
//! - Error conversion from system errors
//! - Categorization of retryable vs fatal errors
//!
//! ## Error Categories
//! 1. **Network Errors**: UDP send/receive failures
//! 2. **TUN Errors**: Device creation, read/write failures
//! 3. **Configuration Errors**: Invalid addresses, ports
//! 4. **System Errors**: Permission denied, resource limits
//!
//! ## ⚠️ Important Note for Next Developer
//! - Network errors are often transient and retryable
//! - TUN errors may require elevated privileges
//! - System errors should be logged with context
//!
//! ## Last Modified
//! v0.1.0 - Initial error definitions

use std::io;
use std::net::SocketAddr;

use thiserror::Error;

use aeronyx_common::error::CommonError;

// ============================================
// Result Type Alias
// ============================================

/// Result type for transport operations.
pub type Result<T> = std::result::Result<T, TransportError>;

// ============================================
// TransportError
// ============================================

/// Transport layer error types.
///
/// # Categories
/// - **Network**: Socket and network-related errors
/// - **Tun**: TUN device specific errors
/// - **Config**: Configuration and setup errors
/// - **System**: OS-level errors
#[derive(Error, Debug)]
pub enum TransportError {
    // ========================================
    // Network Errors
    // ========================================

    /// Failed to bind to address.
    #[error("Failed to bind to {addr}: {reason}")]
    BindFailed {
        /// Address we tried to bind to
        addr: SocketAddr,
        /// Why binding failed
        reason: String,
    },

    /// Send operation failed.
    #[error("Failed to send to {dest}: {reason}")]
    SendFailed {
        /// Destination address
        dest: SocketAddr,
        /// Why send failed
        reason: String,
    },

    /// Receive operation failed.
    #[error("Failed to receive: {reason}")]
    ReceiveFailed {
        /// Why receive failed
        reason: String,
    },

    /// Socket is not connected.
    #[error("Socket not connected")]
    NotConnected,

    /// Address already in use.
    #[error("Address {addr} already in use")]
    AddressInUse {
        /// The address that's in use
        addr: SocketAddr,
    },

    // ========================================
    // TUN Device Errors
    // ========================================

    /// Failed to create TUN device.
    #[error("Failed to create TUN device '{name}': {reason}")]
    TunCreateFailed {
        /// Requested device name
        name: String,
        /// Why creation failed
        reason: String,
    },

    /// Failed to configure TUN device.
    #[error("Failed to configure TUN device '{name}': {reason}")]
    TunConfigFailed {
        /// Device name
        name: String,
        /// Why configuration failed
        reason: String,
    },

    /// TUN device read failed.
    #[error("TUN read failed: {reason}")]
    TunReadFailed {
        /// Why read failed
        reason: String,
    },

    /// TUN device write failed.
    #[error("TUN write failed: {reason}")]
    TunWriteFailed {
        /// Why write failed
        reason: String,
    },

    /// TUN device not found.
    #[error("TUN device '{name}' not found")]
    TunNotFound {
        /// Device name that wasn't found
        name: String,
    },

    // ========================================
    // Configuration Errors
    // ========================================

    /// Invalid configuration.
    #[error("Invalid configuration: {field} - {reason}")]
    InvalidConfig {
        /// Configuration field name
        field: String,
        /// Why it's invalid
        reason: String,
    },

    /// Invalid IP address.
    #[error("Invalid IP address: {addr}")]
    InvalidAddress {
        /// The invalid address string
        addr: String,
    },

    /// Invalid port number.
    #[error("Invalid port: {port}")]
    InvalidPort {
        /// The invalid port
        port: u16,
    },

    // ========================================
    // System Errors
    // ========================================

    /// Permission denied for operation.
    #[error("Permission denied: {operation}")]
    PermissionDenied {
        /// What operation was denied
        operation: String,
    },

    /// Resource limit exceeded.
    #[error("Resource limit exceeded: {resource}")]
    ResourceLimit {
        /// Which resource limit was hit
        resource: String,
    },

    /// Operation timed out.
    #[error("Operation timed out: {operation}")]
    Timeout {
        /// What operation timed out
        operation: String,
    },

    /// Device is shutting down.
    #[error("Transport is shutting down")]
    ShuttingDown,

    // ========================================
    // Wrapped Errors
    // ========================================

    /// I/O error from the system.
    #[error("I/O error: {context}")]
    Io {
        /// What was happening when the error occurred
        context: String,
        /// Underlying I/O error
        #[source]
        source: io::Error,
    },

    /// Error from common crate.
    #[error(transparent)]
    Common(#[from] CommonError),
}

impl TransportError {
    // ========================================
    // Convenience Constructors
    // ========================================

    /// Creates a `BindFailed` error.
    pub fn bind_failed(addr: SocketAddr, reason: impl Into<String>) -> Self {
        Self::BindFailed {
            addr,
            reason: reason.into(),
        }
    }

    /// Creates a `TunCreateFailed` error.
    pub fn tun_create_failed(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::TunCreateFailed {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Creates an `Io` error with context.
    pub fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Creates an `InvalidConfig` error.
    pub fn invalid_config(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidConfig {
            field: field.into(),
            reason: reason.into(),
        }
    }

    // ========================================
    // Error Classification
    // ========================================

    /// Returns `true` if this error is transient and retryable.
    ///
    /// Transient errors may succeed if the operation is retried.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Timeout { .. } => true,
            Self::Io { source, .. } => matches!(
                source.kind(),
                io::ErrorKind::WouldBlock
                    | io::ErrorKind::Interrupted
                    | io::ErrorKind::TimedOut
            ),
            Self::SendFailed { .. } => true,
            Self::ReceiveFailed { .. } => true,
            _ => false,
        }
    }

    /// Returns `true` if this error requires elevated privileges.
    #[must_use]
    pub const fn requires_privileges(&self) -> bool {
        matches!(
            self,
            Self::PermissionDenied { .. } | Self::TunCreateFailed { .. }
        )
    }

    /// Returns `true` if this is a network-related error.
    #[must_use]
    pub const fn is_network_error(&self) -> bool {
        matches!(
            self,
            Self::BindFailed { .. }
                | Self::SendFailed { .. }
                | Self::ReceiveFailed { .. }
                | Self::NotConnected
                | Self::AddressInUse { .. }
        )
    }

    /// Returns `true` if this is a TUN device error.
    #[must_use]
    pub const fn is_tun_error(&self) -> bool {
        matches!(
            self,
            Self::TunCreateFailed { .. }
                | Self::TunConfigFailed { .. }
                | Self::TunReadFailed { .. }
                | Self::TunWriteFailed { .. }
                | Self::TunNotFound { .. }
        )
    }
}

// ============================================
// Error Conversions
// ============================================

impl From<io::Error> for TransportError {
    fn from(err: io::Error) -> Self {
        Self::Io {
            context: "unspecified I/O operation".into(),
            source: err,
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
        let err = TransportError::bind_failed(
            "127.0.0.1:8080".parse().unwrap(),
            "address in use",
        );
        assert!(err.to_string().contains("127.0.0.1:8080"));
        assert!(err.to_string().contains("address in use"));
    }

    #[test]
    fn test_error_classification() {
        let network_err = TransportError::SendFailed {
            dest: "127.0.0.1:8080".parse().unwrap(),
            reason: "timeout".into(),
        };
        assert!(network_err.is_network_error());
        assert!(network_err.is_retryable());

        let tun_err = TransportError::tun_create_failed("tun0", "permission denied");
        assert!(tun_err.is_tun_error());
        assert!(tun_err.requires_privileges());
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::WouldBlock, "would block");
        let transport_err: TransportError = io_err.into();
        assert!(transport_err.is_retryable());
    }
}
