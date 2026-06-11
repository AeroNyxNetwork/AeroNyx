//! Runtime node operator policy.
//!
//! Source path:
//!   /root/a/AeroNyx/crates/aeronyx-server/src/services/node_policy.rs
//!
//! The CMS sends `node_policy` in every heartbeat response. This module keeps a
//! local copy for the hot VPN paths so handshake and packet handling can enforce
//! nodeboard Settings without SSH or process restart.

use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

#[derive(Debug, Clone)]
/// Snapshot of operator-controlled VPN node policy from the CMS heartbeat.
pub struct NodePolicySnapshot {
    /// Operator tier label used by nodeboard and the CMS.
    pub node_tier: String,
    /// When enabled, new handshakes are rejected while existing sessions stay alive.
    pub maintenance_mode: bool,
    /// Maximum active VPN sessions allowed by nodeboard; zero means unlimited.
    pub max_sessions: u32,
    /// Node-wide traffic cap in megabits per second; zero means unlimited.
    pub bandwidth_limit_mbps: u32,
    /// CMS-requested heartbeat interval for this node.
    pub heartbeat_interval_seconds: u64,
    /// CMS policy update timestamp, carried for diagnostics.
    pub updated_at: Option<String>,
}

impl Default for NodePolicySnapshot {
    fn default() -> Self {
        Self {
            node_tier: "public".to_string(),
            maintenance_mode: false,
            max_sessions: 0,
            bandwidth_limit_mbps: 0,
            heartbeat_interval_seconds: 30,
            updated_at: None,
        }
    }
}

#[derive(Debug)]
struct BandwidthWindow {
    started_at: Instant,
    bytes: u64,
}

impl Default for BandwidthWindow {
    fn default() -> Self {
        Self {
            started_at: Instant::now(),
            bytes: 0,
        }
    }
}

#[derive(Debug, Default)]
/// Thread-safe runtime policy cache used by handshake and packet hot paths.
pub struct NodePolicyRuntime {
    policy: RwLock<NodePolicySnapshot>,
    bandwidth: Mutex<BandwidthWindow>,
}

impl NodePolicyRuntime {
    /// Replace the local policy snapshot after a successful CMS heartbeat.
    pub fn update(&self, next: NodePolicySnapshot) {
        *self.policy.write() = next;
    }

    #[must_use]
    /// Return a clone of the current policy for diagnostics or tests.
    pub fn snapshot(&self) -> NodePolicySnapshot {
        self.policy.read().clone()
    }

    /// Check whether a new VPN session may be admitted under operator policy.
    pub fn validate_new_session(&self, active_sessions: usize) -> Result<(), &'static str> {
        let policy = self.policy.read();
        if policy.maintenance_mode {
            return Err("maintenance_mode");
        }
        if policy.max_sessions > 0 && active_sessions >= policy.max_sessions as usize {
            return Err("max_sessions");
        }
        Ok(())
    }

    /// Reserve packet bytes against the node-wide one-second bandwidth window.
    pub fn allow_traffic_bytes(&self, bytes: usize) -> bool {
        let limit_mbps = self.policy.read().bandwidth_limit_mbps;
        if limit_mbps == 0 {
            return true;
        }

        let limit_bytes_per_sec = (limit_mbps as u64).saturating_mul(1_000_000) / 8;
        if limit_bytes_per_sec == 0 {
            return false;
        }

        let mut window = self.bandwidth.lock();
        if window.started_at.elapsed() >= Duration::from_secs(1) {
            window.started_at = Instant::now();
            window.bytes = 0;
        }

        let next = window.bytes.saturating_add(bytes as u64);
        if next > limit_bytes_per_sec {
            return false;
        }

        window.bytes = next;
        true
    }
}
