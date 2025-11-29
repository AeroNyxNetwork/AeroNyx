// ============================================
// File: crates/aeronyx-server/src/error.rs
// ============================================
//! # Server Error Types
//!
//! ## Creation Reason
//! Defines error types specific to server operations, including
//! configuration, session management, and runtime errors.
//!
//! ## Main Functionality
//! - `ServerError`: Primary error enum for server operations
//! - Error conversion from underlying crates
//! - Error categorization for handling decisions
//!
//! ## Error Categories
//! 1. **Configuration**: Invalid config, missing files
//! 2. **Session**: Session creation, lookup failures
//! 3. **Routing**: Packet routing errors
//! 4. **Runtime**: Server lifecycle errors
//!
//! ## ⚠️ Important Note for Next Developer
//! - Errors are logged at appropriate levels
//! - Some errors trigger session cleanup
//! - Configuration errors prevent startup
//!
//! ## Last Modified
//! v0.1.0 - Initial error definitions
//! v0.1.1 - Fixed: SocketAddr cannot be #[source], changed to String

use std::net::SocketAddr;

use thiserror::Error;

use aeronyx_common::error::CommonError;
use aeronyx_common::SessionId;
use aeronyx_core::error::CoreError;
use aeronyx_transport::error::TransportError;

// ============================================
// Result Type Alias
// ============================================

/// Result type for server operations.
pub type Result<T> = std::result::Result<T, ServerError>;

// ============================================
// ServerError
// ============================================

/// Server error types.
///
/// # Categories
/// - **Config**: Configuration and setup errors
/// - **Session**: Session management errors
/// - **Routing**: Packet routing errors
/// - **Runtime**: Server lifecycle errors
#[derive(Error, Debug)]
pub enum ServerError {
    // ========================================
    // Configuration Errors
    // ========================================

    /// Failed to load configuration file.
    #[error("Failed to load configuration from '{path}': {reason}")]
    ConfigLoad {
        /// Path to config file
        path: String,
        /// Why loading failed
        reason: String,
    },

    /// Invalid configuration value.
    #[error("Invalid configuration: {field} - {reason}")]
    ConfigInvalid {
        /// Configuration field
        field: String,
        /// Why it's invalid
        reason: String,
    },

    /// Missing required configuration.
    #[error("Missing required configuration: {field}")]
    ConfigMissing {
        /// Missing field
        field: String,
    },

    // ========================================
    // Session Errors
    // ========================================

    /// Session not found.
    #[error("Session not found: {0}")]
    SessionNotFound(SessionId),

    /// Session creation failed.
    #[error("Failed to create session: {reason}")]
    SessionCreationFailed {
        /// Why creation failed
        reason: String,
    },

    /// Session limit reached.
    #[error("Session limit reached: max {limit} sessions")]
    SessionLimitReached {
        /// Maximum allowed sessions
        limit: usize,
    },

    /// Session already exists.
    #[error("Session already exists for client")]
    SessionExists,

    // ========================================
    // IP Pool Errors
    // ========================================

    /// IP pool exhausted.
    #[error("IP address pool exhausted")]
    IpPoolExhausted,

    /// IP address already assigned.
    #[error("IP address {0} already assigned")]
    IpAlreadyAssigned(std::net::Ipv4Addr),

    // ========================================
    // Routing Errors
    // ========================================

    /// No route found for destination.
    #[error("No route found for {destination}")]
    NoRoute {
        /// Destination IP
        destination: std::net::Ipv4Addr,
    },

    /// Invalid packet received.
    #[error("Invalid packet from {source}: {reason}")]
    InvalidPacket {
        /// Packet source (as string to avoid #[source] issue)
        source: String,
        /// Why it's invalid
        reason: String,
    },

    // ========================================
    // Runtime Errors
    // ========================================

    /// Server failed to start.
    #[error("Server failed to start: {reason}")]
    StartupFailed {
        /// Why startup failed
        reason: String,
    },

    /// Server is shutting down.
    #[error("Server is shutting down")]
    ShuttingDown,

    /// Internal server error.
    #[error("Internal error: {message}")]
    Internal {
        /// Error description
        message: String,
    },

    // ========================================
    // Wrapped Errors
    // ========================================

    /// Error from common crate.
    #[error(transparent)]
    Common(#[from] CommonError),

    /// Error from core crate.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Error from transport crate.
    #[error(transparent)]
    Transport(#[from] TransportError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl ServerError {
    // ========================================
    // Convenience Constructors
    // ========================================

    /// Creates a `ConfigLoad` error.
    pub fn config_load(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConfigLoad {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Creates a `ConfigInvalid` error.
    pub fn config_invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConfigInvalid {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Creates a `SessionCreationFailed` error.
    pub fn session_creation_failed(reason: impl Into<String>) -> Self {
        Self::SessionCreationFailed {
            reason: reason.into(),
        }
    }

    /// Creates an `InvalidPacket` error.
    pub fn invalid_packet(source: SocketAddr, reason: impl Into<String>) -> Self {
        Self::InvalidPacket {
            source: source.to_string(),
            reason: reason.into(),
        }
    }

    /// Creates a `StartupFailed` error.
    pub fn startup_failed(reason: impl Into<String>) -> Self {
        Self::StartupFailed {
            reason: reason.into(),
        }
    }

    /// Creates an `Internal` error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    // ========================================
    // Error Classification
    // ========================================

    /// Returns `true` if this is a configuration error.
    #[must_use]
    pub const fn is_config_error(&self) -> bool {
        matches!(
            self,
            Self::ConfigLoad { .. } | Self::ConfigInvalid { .. } | Self::ConfigMissing { .. }
        )
    }

    /// Returns `true` if this is a session error.
    #[must_use]
    pub const fn is_session_error(&self) -> bool {
        matches!(
            self,
            Self::SessionNotFound(_)
                | Self::SessionCreationFailed { .. }
                | Self::SessionLimitReached { .. }
                | Self::SessionExists
        )
    }

    /// Returns `true` if this error should trigger session cleanup.
    #[must_use]
    pub const fn should_cleanup_session(&self) -> bool {
        matches!(
            self,
            Self::Core(CoreError::Decryption)
                | Self::Core(CoreError::SignatureVerification)
                | Self::Core(CoreError::ReplayDetected { .. })
        )
    }

    /// Returns `true` if this error is fatal and server should stop.
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::ConfigLoad { .. }
                | Self::ConfigMissing { .. }
                | Self::StartupFailed { .. }
        )
    }

    /// Returns `true` if this error is transient and retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport(e) => e.is_retryable(),
            Self::IpPoolExhausted => true,
            Self::SessionLimitReached { .. } => true,
            _ => false,
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
        let err = ServerError::config_load("/etc/aeronyx.toml", "file not found");
        assert!(err.to_string().contains("/etc/aeronyx.toml"));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_classification() {
        let config_err = ServerError::config_invalid("port", "must be > 0");
        assert!(config_err.is_config_error());
        assert!(config_err.is_fatal());

        let session_err = ServerError::SessionNotFound(SessionId::generate());
        assert!(session_err.is_session_error());
        assert!(!session_err.is_fatal());

        let limit_err = ServerError::SessionLimitReached { limit: 1000 };
        assert!(limit_err.is_retryable());
    }

    #[test]
    fn test_error_conversion() {
        let core_err = CoreError::Decryption;
        let server_err: ServerError = core_err.into();
        assert!(server_err.should_cleanup_session());
    }
}
