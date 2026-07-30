// ============================================
// File: crates/aeronyx-server/src/error.rs
// ============================================
// Version: 1.2.0-JoinFailurePrivacy
//
// Modification Reason:
//   Added WalletDenied variant for deny list rejection at handshake time.
//   Called by HandshakeService::process() when a wallet is on the deny list.
//   Caller (server.rs UDP task) treats this the same as any other handshake
//   error — sends 0xFF RESET to the client.
//   [RUNTIME-SUPERVISION 2026-07-29 by Codex] Added an explicit fatal runtime
//   error so required service loss is distinguishable from startup/config
//   failure and can terminate the process for systemd recovery.
//   [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] Centralized privacy-safe Tokio
//   task join classification for runtime and storage error boundaries.
//
// What changed:
//   - Added WalletDenied { reason: String } variant
//   - Added is_wallet_denied() helper method
//
// Last Modified:
//   v1.2.0-JoinFailurePrivacy - Fixed typed Tokio join failure categories
//   v1.1.0-RuntimeSupervision - Fatal required-task failure classification
//   v0.1.2               - InvalidPacket source field rename
//   v1.0.0-Voice+SessionFix - is_session_not_found() helper
//   v1.0.0-Membership    - WalletDenied variant + is_wallet_denied()
// ============================================

use std::fmt;
use std::net::SocketAddr;

use thiserror::Error;
use tokio::task::JoinError;

use aeronyx_common::error::CommonError;
use aeronyx_common::SessionId;
use aeronyx_core::error::CoreError;
use aeronyx_transport::error::TransportError;

/// Result type for server operations.
pub type Result<T> = std::result::Result<T, ServerError>;

/// Privacy-safe category for a Tokio task that failed to join.
///
/// [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] `JoinError` may retain a panic
/// payload. Runtime status and storage errors use this fixed category whenever
/// they need diagnostics, preventing that payload from crossing into API,
/// management, or structured log fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeTaskJoinFailureKind {
    /// The task panicked.
    Panicked,
    /// The task was cancelled before it completed.
    Cancelled,
    /// Tokio reported another join failure category.
    Failed,
}

impl RuntimeTaskJoinFailureKind {
    /// Classifies a Tokio join failure without retaining its payload.
    pub(crate) fn classify(error: &JoinError) -> Self {
        if error.is_panic() {
            Self::Panicked
        } else if error.is_cancelled() {
            Self::Cancelled
        } else {
            Self::Failed
        }
    }

    /// Returns the stable process-health reason for a required runtime task.
    pub(crate) const fn required_task_reason(self) -> &'static str {
        match self {
            Self::Panicked => "required runtime task panicked",
            Self::Cancelled => "required runtime task was cancelled unexpectedly",
            Self::Failed => "required runtime task join failed",
        }
    }

    /// Returns the stable process-health reason for a required API listener.
    pub(crate) const fn required_api_listener_reason(self) -> &'static str {
        match self {
            Self::Panicked => "required API listener task panicked",
            Self::Cancelled => "required API listener task was cancelled unexpectedly",
            Self::Failed => "required API listener task join failed",
        }
    }

    /// Returns the stable startup reason for a blocking task.
    pub(crate) const fn blocking_task_reason(self) -> &'static str {
        match self {
            Self::Panicked => "blocking task panicked",
            Self::Cancelled => "blocking task was cancelled",
            Self::Failed => "blocking task join failed",
        }
    }
}

impl fmt::Display for RuntimeTaskJoinFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self {
            Self::Panicked => "task panicked",
            Self::Cancelled => "task was cancelled",
            Self::Failed => "task join failed",
        };
        formatter.write_str(reason)
    }
}

/// Server error types.
#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Failed to load configuration from '{path}': {reason}")]
    ConfigLoad { path: String, reason: String },

    #[error("Invalid configuration: {field} - {reason}")]
    ConfigInvalid { field: String, reason: String },

    #[error("Missing required configuration: {field}")]
    ConfigMissing { field: String },

    #[error("Session not found: {0}")]
    SessionNotFound(SessionId),

    #[error("Failed to create session: {reason}")]
    SessionCreationFailed { reason: String },

    #[error("Session limit reached: max {limit} sessions")]
    SessionLimitReached { limit: usize },

    #[error("Session already exists for client")]
    SessionExists,

    #[error("IP address pool exhausted")]
    IpPoolExhausted,

    #[error("IP address {0} already assigned")]
    IpAlreadyAssigned(std::net::Ipv4Addr),

    #[error("No route found for {destination}")]
    NoRoute { destination: std::net::Ipv4Addr },

    #[error("Invalid packet from {from_addr}: {reason}")]
    InvalidPacket { from_addr: String, reason: String },

    #[error("Server failed to start: {reason}")]
    StartupFailed { reason: String },

    /// A required long-running task exited after startup.
    ///
    /// [RUNTIME-SUPERVISION 2026-07-29 by Codex] This is intentionally fatal:
    /// continuing would leave systemd and node operators with a false-positive
    /// healthy process while a required protocol surface is unavailable.
    #[error("Required runtime task '{task}' failed: {reason}")]
    RuntimeFailed {
        /// Stable privacy-safe runtime role, such as `node_api`.
        task: String,
        /// Local operational failure reason without client or traffic data.
        reason: String,
    },

    #[error("Server is shutting down")]
    ShuttingDown,

    #[error("Internal error: {message}")]
    Internal { message: String },

    #[error("Node policy rejected request: {reason}")]
    NodePolicyRejected { reason: String },

    // ── v1.0.0-Membership ────────────────────────────────────────────────
    /// Wallet is on the deny list — handshake rejected immediately.
    ///
    /// Reasons:
    ///   - "no_premium_access": tier cannot access this premium node
    ///   - "quota_exceeded": Free tier monthly traffic quota exhausted
    ///
    /// Caller sends 0xFF RESET; client will back off and retry later.
    #[error("Wallet denied: {reason}")]
    WalletDenied { reason: String },

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

    /// Builds a fatal error for a required task that exited after startup.
    pub fn runtime_failed(task: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::RuntimeFailed {
            task: task.into(),
            reason: reason.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    pub fn node_policy_rejected(reason: impl Into<String>) -> Self {
        Self::NodePolicyRejected {
            reason: reason.into(),
        }
    }

    /// Returns true if this error indicates the requested session does not
    /// exist in the session manager.
    ///
    /// Used by server.rs::spawn_udp_task to detect packets from clients
    /// that are using a stale session key (e.g. after a server restart).
    #[must_use]
    pub fn is_session_not_found(&self) -> bool {
        matches!(self, Self::SessionNotFound(_))
    }

    /// Returns true if this error indicates a wallet was rejected by the
    /// deny list check in HandshakeService::process().
    ///
    /// Caller should send 0xFF RESET — same as other handshake errors.
    #[must_use]
    pub fn is_wallet_denied(&self) -> bool {
        matches!(self, Self::WalletDenied { .. })
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
                | Self::ConfigInvalid { .. }
                | Self::ConfigMissing { .. }
                | Self::StartupFailed { .. }
                | Self::RuntimeFailed { .. }
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

        // [RUNTIME-SUPERVISION 2026-07-29 by Codex] Required task loss must
        // remain fatal so the process exits into systemd's restart policy.
        let runtime_err = ServerError::runtime_failed("node_api", "listener exited");
        assert!(runtime_err.is_fatal());
        assert!(!runtime_err.is_config_error());
        assert!(runtime_err.to_string().contains("node_api"));
    }

    #[test]
    fn test_is_session_not_found() {
        let sid = aeronyx_common::types::SessionId::generate();
        let not_found = ServerError::SessionNotFound(sid);
        assert!(not_found.is_session_not_found());

        let other = ServerError::startup_failed("test");
        assert!(!other.is_session_not_found());

        let invalid_pkt = ServerError::invalid_packet("0.0.0.0:0".parse().unwrap(), "test");
        assert!(!invalid_pkt.is_session_not_found());
    }

    #[test]
    fn test_is_wallet_denied() {
        let err = ServerError::WalletDenied {
            reason: "quota_exceeded".to_string(),
        };
        assert!(err.is_wallet_denied());
        assert!(!err.is_session_not_found());

        let other = ServerError::startup_failed("test");
        assert!(!other.is_wallet_denied());
    }
}
