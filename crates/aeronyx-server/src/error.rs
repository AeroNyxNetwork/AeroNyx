// ============================================
// File: crates/aeronyx-server/src/error.rs
// ============================================
//! # Server Error Types
//!
//! ## Last Modified
//! v0.1.2 - Fixed: InvalidPacket source field renamed to avoid thiserror issue

use std::net::SocketAddr;

use thiserror::Error;

use aeronyx_common::error::CommonError;
use aeronyx_common::SessionId;
use aeronyx_core::error::CoreError;
use aeronyx_transport::error::TransportError;

/// Result type for server operations.
pub type Result<T> = std::result::Result<T, ServerError>;

/// Server error types.
#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Failed to load configuration from '{path}': {reason}")]
    ConfigLoad {
        path: String,
        reason: String,
    },

    #[error("Invalid configuration: {field} - {reason}")]
    ConfigInvalid {
        field: String,
        reason: String,
    },

    #[error("Missing required configuration: {field}")]
    ConfigMissing {
        field: String,
    },

    #[error("Session not found: {0}")]
    SessionNotFound(SessionId),

    #[error("Failed to create session: {reason}")]
    SessionCreationFailed {
        reason: String,
    },

    #[error("Session limit reached: max {limit} sessions")]
    SessionLimitReached {
        limit: usize,
    },

    #[error("Session already exists for client")]
    SessionExists,

    #[error("IP address pool exhausted")]
    IpPoolExhausted,

    #[error("IP address {0} already assigned")]
    IpAlreadyAssigned(std::net::Ipv4Addr),

    #[error("No route found for {destination}")]
    NoRoute {
        destination: std::net::Ipv4Addr,
    },

    #[error("Invalid packet from {from_addr}: {reason}")]
    InvalidPacket {
        from_addr: String,
        reason: String,
    },

    #[error("Server failed to start: {reason}")]
    StartupFailed {
        reason: String,
    },

    #[error("Server is shutting down")]
    ShuttingDown,

    #[error("Internal error: {message}")]
    Internal {
        message: String,
    },

    #[error(transparent)]
    Common(#[from] CommonError),

    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    Transport(#[from] TransportError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl ServerError {
    pub fn config_load(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConfigLoad {
            path: path.into(),
            reason: reason.into(),
        }
    }

    pub fn config_invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConfigInvalid {
            field: field.into(),
            reason: reason.into(),
        }
    }

    pub fn session_creation_failed(reason: impl Into<String>) -> Self {
        Self::SessionCreationFailed {
            reason: reason.into(),
        }
    }

    pub fn invalid_packet(source: SocketAddr, reason: impl Into<String>) -> Self {
        Self::InvalidPacket {
            from_addr: source.to_string(),
            reason: reason.into(),
        }
    }

    pub fn startup_failed(reason: impl Into<String>) -> Self {
        Self::StartupFailed {
            reason: reason.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn is_config_error(&self) -> bool {
        matches!(
            self,
            Self::ConfigLoad { .. } | Self::ConfigInvalid { .. } | Self::ConfigMissing { .. }
        )
    }

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

    #[must_use]
    pub const fn should_cleanup_session(&self) -> bool {
        matches!(
            self,
            Self::Core(CoreError::Decryption)
                | Self::Core(CoreError::SignatureVerification)
                | Self::Core(CoreError::ReplayDetected { .. })
        )
    }

    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::ConfigLoad { .. }
                | Self::ConfigMissing { .. }
                | Self::StartupFailed { .. }
        )
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ServerError::config_load("/etc/aeronyx.toml", "file not found");
        assert!(err.to_string().contains("/etc/aeronyx.toml"));
    }

    #[test]
    fn test_error_classification() {
        let config_err = ServerError::config_invalid("port", "must be > 0");
        assert!(config_err.is_config_error());
        assert!(config_err.is_fatal());
    }
}
