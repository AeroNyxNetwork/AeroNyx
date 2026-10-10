// ============================================
// File: crates/aeronyx-server/src/server/peer_http.rs
// ============================================
//! # Peer HTTP transport profiles
//!
//! Owns the named, process-lifetime HTTP transport budgets and the
//! `PeerHttpClients` bundle used for authenticated node-to-node traffic.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `server.rs`; bodies unchanged.

use std::sync::Arc;
use std::time::Duration;

use crate::api::directory_replica_sync::{
    DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS, DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS,
};
use crate::api::privacy_safe_peer_http_client_builder;
use crate::config::ServerConfig;
use crate::error::{Result, ServerError};

/// One immutable process-lifetime peer HTTP transport budget.
///
/// [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Transport budgets are named
/// values rather than duplicated builder chains. This keeps production,
/// tests, startup logs, and standalone Directory constructors on one contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PeerHttpProfileSpec {
    name: &'static str,
    pub(super) connect_timeout_secs: Option<u64>,
    pub(super) request_timeout_secs: u64,
    pool_max_idle_per_host: usize,
    pool_idle_timeout_secs: Option<u64>,
}

impl PeerHttpProfileSpec {
    fn build(&self) -> Result<Arc<reqwest::Client>> {
        let mut builder = privacy_safe_peer_http_client_builder()
            .timeout(Duration::from_secs(self.request_timeout_secs))
            .pool_max_idle_per_host(self.pool_max_idle_per_host);
        if let Some(connect_timeout_secs) = self.connect_timeout_secs {
            builder = builder.connect_timeout(Duration::from_secs(connect_timeout_secs));
        }
        if let Some(pool_idle_timeout_secs) = self.pool_idle_timeout_secs {
            builder = builder.pool_idle_timeout(Duration::from_secs(pool_idle_timeout_secs));
        }
        builder.build().map(Arc::new).map_err(|error| {
            ServerError::startup_failed(format!(
                "privacy-safe peer HTTP {} client initialization failed: {error}",
                self.name
            ))
        })
    }
}

const PEER_CONTROL_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "control",
    connect_timeout_secs: Some(3),
    request_timeout_secs: 5,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
pub(super) const DIRECTORY_SYNC_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "directory_sync",
    connect_timeout_secs: Some(DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS),
    request_timeout_secs: DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
pub(super) const DIRECTORY_OPERATOR_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "directory_operator",
    connect_timeout_secs: Some(3),
    request_timeout_secs: 12,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
pub(super) const MEMCHAIN_SYNC_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "memchain_sync",
    connect_timeout_secs: Some(5),
    request_timeout_secs: 15,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
// [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Replica availability recovery
// must fail over before the operator-only diagnostic request budget expires.
const _: () = assert!(
    DIRECTORY_SYNC_HTTP_PROFILE.request_timeout_secs
        < DIRECTORY_OPERATOR_HTTP_PROFILE.request_timeout_secs
);

/// Process-lifetime HTTP transports for authenticated node-to-node traffic.
///
/// [PEER-TRANSPORT-RUNTIME 2026-07-28 by Codex] Each profile preserves its
/// historical timeout and pool budget while sharing one connection pool for
/// the complete server lifetime. Building every profile before mutable
/// services start prevents a configured protocol capability from disappearing
/// later because one background task failed to initialize its own client.
#[derive(Clone)]
pub(super) struct PeerHttpClients {
    /// Short control requests: relay, witnesses, leases, and cache anchors.
    pub(super) control: Arc<reqwest::Client>,
    /// Directory replica synchronization with its historical failover budget.
    pub(super) directory_sync: Arc<reqwest::Client>,
    /// Operator-only Directory diagnostics with a larger bounded deadline.
    pub(super) directory_operator: Arc<reqwest::Client>,
    /// Long-running MemChain page and checkpoint synchronization requests.
    pub(super) sync: Arc<reqwest::Client>,
    /// Discovery gossip with its operator-configured fetch timeout and pool.
    pub(super) gossip: Arc<reqwest::Client>,
}

impl PeerHttpClients {
    pub(super) fn build(config: &ServerConfig) -> Result<Self> {
        let control = PEER_CONTROL_HTTP_PROFILE.build()?;
        let directory_sync = DIRECTORY_SYNC_HTTP_PROFILE.build()?;
        let directory_operator = DIRECTORY_OPERATOR_HTTP_PROFILE.build()?;
        let sync = MEMCHAIN_SYNC_HTTP_PROFILE.build()?;
        let gossip = PeerHttpProfileSpec {
            name: "gossip",
            connect_timeout_secs: None,
            request_timeout_secs: config.discovery.fetch_timeout_secs,
            pool_max_idle_per_host: 8,
            pool_idle_timeout_secs: Some(90),
        }
        .build()?;
        Ok(Self {
            control,
            directory_sync,
            directory_operator,
            sync,
            gossip,
        })
    }
}
