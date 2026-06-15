//! Runtime node operator policy.
//!
//! Source path:
//!   /root/open/AeroNyx/crates/aeronyx-server/src/services/node_policy.rs
//!
//! The CMS sends `node_policy` in every heartbeat response. This module keeps a
//! local copy for the hot VPN paths so handshake and packet handling can enforce
//! nodeboard Settings without SSH or process restart.
//!
//! Commercial placement readiness is also computed here because the Rust node
//! is the authoritative runtime for maintenance mode, max_sessions, and the
//! node-wide bandwidth limiter. Nodeboard and the CMS should render this
//! aggregate decision instead of re-implementing admission policy.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Default, Serialize)]
/// Privacy-safe aggregate counters for operator policy enforcement.
pub struct NodePolicyEnforcementSnapshot {
    /// Unix timestamp when these in-process enforcement counters started.
    pub counters_started_at: u64,
    /// New handshakes rejected while maintenance mode was enabled.
    pub maintenance_rejections: u64,
    /// New handshakes rejected because active sessions reached max_sessions.
    pub max_sessions_rejections: u64,
    /// VPN packets rejected by the node-wide bandwidth limiter.
    pub bandwidth_drops: u64,
    /// Total plaintext VPN bytes rejected by the node-wide bandwidth limiter.
    pub bandwidth_drop_bytes: u64,
    /// Current node-wide limiter in bytes per second; zero means unlimited.
    pub bandwidth_limit_bytes_per_second: u64,
    /// Bytes already admitted in the active one-second limiter window.
    pub bandwidth_window_bytes: u64,
    /// Last policy reason that rejected a handshake or packet.
    pub last_rejection_reason: Option<String>,
    /// Unix timestamp of the last policy rejection.
    pub last_rejection_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
/// Privacy-safe runtime admission summary for commercial placement.
pub struct NodePolicyPlacementSnapshot {
    /// True when this node can accept a new VPN session under local policy.
    pub accepting_new_sessions: bool,
    /// Compact machine status for backend and nodeboard placement triage.
    pub status: String,
    /// Primary admission reason: accepting, maintenance_mode, or max_sessions.
    pub reason: String,
    /// Operator-facing detail safe for heartbeat and nodeboard.
    pub detail: String,
    /// Current active VPN sessions on this Rust process.
    pub active_sessions: usize,
    /// Operator configured max sessions; zero means unlimited.
    pub max_sessions: u32,
    /// Remaining bounded session slots; null means unlimited.
    pub session_capacity_remaining: Option<u32>,
    /// Bounded session usage percent; null means unlimited.
    pub session_capacity_used_percent: Option<f64>,
    /// Whether nodeboard maintenance mode is active locally.
    pub maintenance_mode: bool,
    /// Operator configured bandwidth cap; zero means unlimited.
    pub bandwidth_limit_mbps: u32,
    /// Current bandwidth cap in bytes per second; zero means unlimited.
    pub bandwidth_limit_bytes_per_second: u64,
    /// Bytes already admitted in the active one-second limiter window.
    pub bandwidth_window_bytes: u64,
    /// Current one-second limiter usage percent; null means unlimited.
    pub bandwidth_window_used_percent: Option<f64>,
    /// Compact traffic-capacity status for backend commercial placement UI.
    pub traffic_capacity_status: String,
    /// Source path for backend/nodeboard contract tracing.
    pub source: &'static str,
    /// Explicit privacy boundary for heartbeat consumers.
    pub privacy_boundary: &'static str,
}

#[derive(Debug)]
struct PolicyEnforcementStats {
    counters_started_at: u64,
    maintenance_rejections: u64,
    max_sessions_rejections: u64,
    bandwidth_drops: u64,
    bandwidth_drop_bytes: u64,
    last_rejection_reason: Option<&'static str>,
    last_rejection_at: Option<u64>,
}

impl Default for PolicyEnforcementStats {
    fn default() -> Self {
        Self {
            counters_started_at: unix_now_secs(),
            maintenance_rejections: 0,
            max_sessions_rejections: 0,
            bandwidth_drops: 0,
            bandwidth_drop_bytes: 0,
            last_rejection_reason: None,
            last_rejection_at: None,
        }
    }
}

#[derive(Debug, Default)]
/// Thread-safe runtime policy cache used by handshake and packet hot paths.
pub struct NodePolicyRuntime {
    policy: RwLock<NodePolicySnapshot>,
    bandwidth: Mutex<BandwidthWindow>,
    enforcement: Mutex<PolicyEnforcementStats>,
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

    #[must_use]
    /// Return privacy-safe aggregate policy enforcement counters.
    pub fn enforcement_snapshot(&self) -> NodePolicyEnforcementSnapshot {
        let limit_mbps = self.policy.read().bandwidth_limit_mbps;
        let limit_bytes_per_second = bandwidth_limit_bytes_per_second(limit_mbps);
        let window_bytes = {
            let window = self.bandwidth.lock();
            if window.started_at.elapsed() < Duration::from_secs(1) {
                window.bytes
            } else {
                0
            }
        };
        let stats = self.enforcement.lock();
        NodePolicyEnforcementSnapshot {
            counters_started_at: stats.counters_started_at,
            maintenance_rejections: stats.maintenance_rejections,
            max_sessions_rejections: stats.max_sessions_rejections,
            bandwidth_drops: stats.bandwidth_drops,
            bandwidth_drop_bytes: stats.bandwidth_drop_bytes,
            bandwidth_limit_bytes_per_second: limit_bytes_per_second,
            bandwidth_window_bytes: window_bytes,
            last_rejection_reason: stats.last_rejection_reason.map(str::to_string),
            last_rejection_at: stats.last_rejection_at,
        }
    }

    #[must_use]
    /// Return the local runtime decision for admitting new commercial traffic.
    pub fn placement_snapshot(&self, active_sessions: usize) -> NodePolicyPlacementSnapshot {
        let policy = self.policy.read().clone();
        let limit_bytes_per_second = bandwidth_limit_bytes_per_second(policy.bandwidth_limit_mbps);
        let bandwidth_window_bytes = {
            let window = self.bandwidth.lock();
            if window.started_at.elapsed() < Duration::from_secs(1) {
                window.bytes
            } else {
                0
            }
        };
        let session_capacity_remaining = if policy.max_sessions > 0 {
            Some((policy.max_sessions as usize).saturating_sub(active_sessions) as u32)
        } else {
            None
        };
        let session_capacity_used_percent = if policy.max_sessions > 0 {
            Some(round_percent((active_sessions as f64 / policy.max_sessions as f64) * 100.0))
        } else {
            None
        };
        let bandwidth_window_used_percent = if limit_bytes_per_second > 0 {
            Some(round_percent((bandwidth_window_bytes as f64 / limit_bytes_per_second as f64) * 100.0))
        } else {
            None
        };
        let traffic_capacity_status = if limit_bytes_per_second == 0 {
            "unlimited"
        } else if bandwidth_window_bytes >= limit_bytes_per_second {
            "saturated"
        } else if bandwidth_window_used_percent.unwrap_or(0.0) >= 80.0 {
            "near_limit"
        } else {
            "available"
        };
        let (accepting_new_sessions, status, reason, detail) = if policy.maintenance_mode {
            (
                false,
                "blocked",
                "maintenance_mode",
                "Maintenance mode is enabled; new VPN handshakes are rejected while existing sessions can drain.".to_string(),
            )
        } else if policy.max_sessions > 0 && active_sessions >= policy.max_sessions as usize {
            (
                false,
                "blocked",
                "max_sessions",
                format!(
                    "Active sessions ({}) reached max_sessions ({}); new handshakes are rejected.",
                    active_sessions, policy.max_sessions
                ),
            )
        } else {
            (
                true,
                if traffic_capacity_status == "near_limit" { "watch" } else { "accepting" },
                "accepting",
                "Node policy allows new VPN handshakes under the current maintenance and session limits.".to_string(),
            )
        };

        NodePolicyPlacementSnapshot {
            accepting_new_sessions,
            status: status.to_string(),
            reason: reason.to_string(),
            detail,
            active_sessions,
            max_sessions: policy.max_sessions,
            session_capacity_remaining,
            session_capacity_used_percent,
            maintenance_mode: policy.maintenance_mode,
            bandwidth_limit_mbps: policy.bandwidth_limit_mbps,
            bandwidth_limit_bytes_per_second: limit_bytes_per_second,
            bandwidth_window_bytes,
            bandwidth_window_used_percent,
            traffic_capacity_status: traffic_capacity_status.to_string(),
            source: "/root/open/AeroNyx/crates/aeronyx-server/src/services/node_policy.rs",
            privacy_boundary: concat!(
                "aggregate node policy and capacity state only; no client ",
                "public IPs, destinations, DNS contents, packet payloads, ",
                "domains, URLs, browsing history, voucher secrets, or ",
                "wallet-level traffic"
            ),
        }
    }

    /// Check whether a new VPN session may be admitted under operator policy.
    pub fn validate_new_session(&self, active_sessions: usize) -> Result<(), &'static str> {
        let rejection = {
            let policy = self.policy.read();
            if policy.maintenance_mode {
                Some("maintenance_mode")
            } else if policy.max_sessions > 0 && active_sessions >= policy.max_sessions as usize {
                Some("max_sessions")
            } else {
                None
            }
        };

        if let Some(reason) = rejection {
            self.record_rejection(reason, 0);
            return Err(reason);
        }

        Ok(())
    }

    /// Reserve packet bytes against the node-wide one-second bandwidth window.
    pub fn allow_traffic_bytes(&self, bytes: usize) -> bool {
        let limit_mbps = self.policy.read().bandwidth_limit_mbps;
        if limit_mbps == 0 {
            return true;
        }

        let rejected_bytes = bytes as u64;
        let limit_bytes_per_sec = bandwidth_limit_bytes_per_second(limit_mbps);
        if limit_bytes_per_sec == 0 {
            self.record_rejection("bandwidth_limit_mbps", rejected_bytes);
            return false;
        }

        let mut window = self.bandwidth.lock();
        if window.started_at.elapsed() >= Duration::from_secs(1) {
            window.started_at = Instant::now();
            window.bytes = 0;
        }

        let next = window.bytes.saturating_add(bytes as u64);
        if next > limit_bytes_per_sec {
            self.record_rejection("bandwidth_limit_mbps", rejected_bytes);
            return false;
        }

        window.bytes = next;
        true
    }

    fn record_rejection(&self, reason: &'static str, rejected_bytes: u64) {
        let mut stats = self.enforcement.lock();
        match reason {
            "maintenance_mode" => {
                stats.maintenance_rejections = stats.maintenance_rejections.saturating_add(1);
            }
            "max_sessions" => {
                stats.max_sessions_rejections = stats.max_sessions_rejections.saturating_add(1);
            }
            "bandwidth_limit_mbps" => {
                stats.bandwidth_drops = stats.bandwidth_drops.saturating_add(1);
                stats.bandwidth_drop_bytes = stats.bandwidth_drop_bytes.saturating_add(rejected_bytes);
            }
            _ => {}
        }
        stats.last_rejection_reason = Some(reason);
        stats.last_rejection_at = Some(unix_now_secs());
    }
}

fn bandwidth_limit_bytes_per_second(limit_mbps: u32) -> u64 {
    (limit_mbps as u64).saturating_mul(1_000_000) / 8
}

fn round_percent(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::{NodePolicyRuntime, NodePolicySnapshot};

    #[test]
    fn placement_snapshot_blocks_maintenance_mode() {
        let runtime = NodePolicyRuntime::default();
        runtime.update(NodePolicySnapshot {
            maintenance_mode: true,
            max_sessions: 10,
            bandwidth_limit_mbps: 0,
            ..NodePolicySnapshot::default()
        });

        let snapshot = runtime.placement_snapshot(1);

        assert!(!snapshot.accepting_new_sessions);
        assert_eq!(snapshot.status, "blocked");
        assert_eq!(snapshot.reason, "maintenance_mode");
        assert_eq!(snapshot.session_capacity_remaining, Some(9));
    }

    #[test]
    fn placement_snapshot_blocks_full_max_sessions() {
        let runtime = NodePolicyRuntime::default();
        runtime.update(NodePolicySnapshot {
            max_sessions: 2,
            bandwidth_limit_mbps: 100,
            ..NodePolicySnapshot::default()
        });

        let snapshot = runtime.placement_snapshot(2);

        assert!(!snapshot.accepting_new_sessions);
        assert_eq!(snapshot.status, "blocked");
        assert_eq!(snapshot.reason, "max_sessions");
        assert_eq!(snapshot.session_capacity_remaining, Some(0));
        assert_eq!(snapshot.session_capacity_used_percent, Some(100.0));
    }

    #[test]
    fn placement_snapshot_accepts_unlimited_sessions() {
        let runtime = NodePolicyRuntime::default();

        let snapshot = runtime.placement_snapshot(128);

        assert!(snapshot.accepting_new_sessions);
        assert_eq!(snapshot.status, "accepting");
        assert_eq!(snapshot.reason, "accepting");
        assert_eq!(snapshot.session_capacity_remaining, None);
        assert_eq!(snapshot.session_capacity_used_percent, None);
    }
}
