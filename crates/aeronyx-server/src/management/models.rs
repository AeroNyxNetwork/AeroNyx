// ============================================
// File: crates/aeronyx-server/src/management/models.rs
// Path: aeronyx-server/src/management/models.rs
// ============================================
//! Purpose: Management API data models for CMS communication
//!
//! Key Fix v1.2.0:
//!   - HeartbeatRequest struct is kept for documentation but NOT used for sending
//!   - Actual heartbeat uses json! macro in client.rs to preserve field order
//!   - SessionEventReport uses String for event_type (not enum) for flexibility
//!
//! Modification Reason (v1.3.0):
//!   - Added `Command` struct for CMS command dispatch (Phase 1: Command Pipeline)
//!   - Upgraded `HeartbeatResponse.commands` from `Option<Vec<String>>`
//!     to `Option<Vec<Command>>` to support structured commands with params
//!   - Added `AgentStatus` enum and `AgentStatusInfo` for OpenClaw lifecycle tracking
//!   - Added `CommandStatusReport` for Rust -> CMS command execution feedback
//!   - Added `agent_status` to `SystemStats` for heartbeat-based status reporting
//!
//! Modification Reason (v1.0.0-TrafficAccounting):
//!   - Added `SessionTrafficSnapshot` to `SessionEventType` for periodic
//!     mid-session traffic reporting. Long-lived sessions previously had zero
//!     visibility until disconnect. Now a snapshot is sent every 5 minutes
//!     with cumulative bytes_in / bytes_out since session start.
//!   - Added `is_final` field to `SessionEventReport` so the backend can
//!     distinguish a snapshot (upsert, no billing trigger) from a final
//!     session_ended event (closes billing period).
//!   - All existing fields and behaviour preserved verbatim.
//!
//! Main Data Structures:
//!   - BindNodeRequest/Response: Node registration
//!   - HeartbeatRequest/Response: Periodic status updates
//!   - SessionEventReport/Response: Session event reporting
//!   - HardwareInfo: System hardware information
//!   - SystemStats: Runtime system statistics
//!   - Command: CMS -> Rust structured command with params
//!   - AgentStatus / AgentStatusInfo: OpenClaw agent lifecycle state
//!   - CommandStatusReport: Rust -> CMS command execution result
//!
//! ⚠️ Important Note for Next Developer:
//!   - HeartbeatRequest struct field order is documented but struct is NOT used
//!     for actual HTTP request — client.rs uses json! macro directly
//!   - If you need to modify heartbeat fields, update BOTH models.rs AND client.rs
//!   - `Command.id` is used by CMS to track command lifecycle — always include
//!     it in status reports
//!   - `Command.params` is `serde_json::Value` for maximum flexibility
//!   - `AgentStatus` default is `NotInstalled` — do NOT rename variants,
//!     CMS relies on snake_case serialisation
//!   - `SessionTrafficSnapshot` bytes_in/bytes_out are CUMULATIVE since session
//!     start, NOT deltas. Backend must upsert, not accumulate.
//!   - `is_final=true` only on SessionEnded — backend uses this to close billing.
//!
//! Dependencies:
//!   - serde for serialization/deserialization
//!   - serde_json for `Command.params` (dynamic JSON value)
//!   - System files (/proc/cpuinfo, /proc/meminfo, etc.) for Linux stats
//!
//! Last Modified:
//!   v1.0.0 - Initial models
//!   v1.2.0 - Clarified HeartbeatRequest is documentation only
//!   v1.3.0 - Added Command, AgentStatus, AgentStatusInfo, CommandStatusReport
//!   v1.0.0-TrafficAccounting - Added SessionTrafficSnapshot event type,
//!     is_final field to SessionEventReport

use serde::{Deserialize, Serialize};

// ============================================
// Node Registration
// ============================================

/// Request to bind a node to a user account using a registration code.
#[derive(Debug, Serialize)]
pub struct BindNodeRequest {
    /// Registration code from CMS dashboard
    pub code: String,
    /// Node's public key (hex-encoded)
    pub public_key: String,
    /// Hardware information for node identification
    pub hardware_info: HardwareInfo,
}

/// Hardware information collected from the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model name
    pub cpu: String,
    /// Total memory (e.g., "16GB")
    pub memory: String,
    /// Operating system name
    pub os: String,
}

impl HardwareInfo {
    /// Collects hardware information from the current system.
    pub fn collect() -> Self {
        Self {
            cpu: Self::get_cpu_info(),
            memory: Self::get_memory_info(),
            os: Self::get_os_info(),
        }
    }

    fn get_cpu_info() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|content| {
                    content.lines()
                        .find(|line| line.starts_with("model name"))
                        .and_then(|line| line.split(':').nth(1))
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_else(|| "Unknown CPU".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        { "Unknown CPU".to_string() }
    }

    fn get_memory_info() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content.lines()
                        .find(|line| line.starts_with("MemTotal"))
                        .and_then(|line| line.split(':').nth(1))
                        .map(|s| {
                            let kb: u64 = s.trim().split_whitespace().next()
                                .and_then(|n| n.parse().ok()).unwrap_or(0);
                            format!("{}GB", kb / 1024 / 1024)
                        })
                })
                .unwrap_or_else(|| "Unknown".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        { "Unknown".to_string() }
    }

    fn get_os_info() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/etc/os-release")
                .ok()
                .and_then(|content| {
                    content.lines()
                        .find(|line| line.starts_with("PRETTY_NAME"))
                        .and_then(|line| line.split('=').nth(1))
                        .map(|s| s.trim_matches('"').to_string())
                })
                .unwrap_or_else(|| "Linux".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        { "Unknown OS".to_string() }
    }
}

/// Response from node binding request.
#[derive(Debug, Deserialize)]
pub struct BindNodeResponse {
    /// Whether the binding was successful
    pub success: bool,
    /// Node information (present on success)
    pub node: Option<NodeInfo>,
    /// Human-readable message
    pub message: Option<String>,
    /// Error message (present on failure)
    pub error: Option<String>,
}

/// Information about a registered node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier
    pub id: String,
    /// Owner's wallet address
    pub owner_wallet: String,
    /// Human-readable node name
    pub name: String,
    /// Node's public key (hex-encoded)
    pub public_key: String,
    /// Current status string
    pub status: String,
    /// ISO 8601 creation timestamp
    pub created_at: String,
}

// ============================================
// Heartbeat
// ============================================

/// Heartbeat request structure.
///
/// ⚠️ IMPORTANT: This struct is for DOCUMENTATION ONLY.
/// The actual HTTP request in client.rs uses json! macro to ensure field order.
///
/// CRITICAL: Field order MUST match Python backend expectation:
/// 1. node_id
/// 2. timestamp
/// 3. public_ip
/// 4. version
/// 5. binary_hash
/// 6. system_stats
/// 7. signature
///
/// Do NOT use this struct directly with .json(&request) — it will serialize
/// fields in alphabetical order, breaking signature verification!
#[derive(Debug, Serialize)]
pub struct HeartbeatRequest {
    pub node_id: String,           // 1st - MUST be first
    pub timestamp: u64,            // 2nd
    pub public_ip: String,         // 3rd
    pub version: String,           // 4th
    pub binary_hash: String,       // 5th
    pub system_stats: SystemStats, // 6th
    pub signature: String,         // 7th - MUST be last
}

/// System statistics for heartbeat reporting.
///
/// v1.3.1: Enhanced for cloud/container compatibility.
/// All fields use best-effort collection — missing data defaults to 0/None.
///
/// New optional fields use `skip_serializing_if` so they don't
/// break CMS versions that don't expect them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    /// CPU usage percentage (0.0 - 100.0).
    pub cpu_usage: f32,
    /// Memory usage in megabytes.
    pub memory_mb: u64,
    /// Number of active VPN sessions.
    pub active_sessions: u32,

    /// v1.3.0: OpenClaw agent status for CMS dashboard display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_status: Option<AgentStatusInfo>,

    /// v1.3.1: Total network bytes received since boot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub net_rx_bytes: Option<u64>,

    /// v1.3.1: Total network bytes transmitted since boot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub net_tx_bytes: Option<u64>,

    /// v1.3.1: Total memory available on the system/container in MB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_total_mb: Option<u64>,

    /// v1.3.1: Number of CPU cores (vCPU count).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_count: Option<u32>,

    /// M2: Per-process runtime identifier for server reset detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_id: Option<String>,

    /// M2: Unix timestamp for when this Rust process started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_started_at: Option<u64>,
}

impl SystemStats {
    /// Collects current system statistics.
    pub fn collect(active_sessions: u32) -> Self {
        let (memory_mb, memory_total_mb) = Self::get_memory_info();
        let (net_rx, net_tx) = Self::get_network_stats();
        Self {
            cpu_usage: Self::get_cpu_usage(),
            memory_mb,
            active_sessions,
            agent_status: None,
            net_rx_bytes: net_rx,
            net_tx_bytes: net_tx,
            memory_total_mb,
            cpu_count: std::thread::available_parallelism()
                .map(|p| Some(p.get() as u32))
                .unwrap_or(None),
            runtime_id: None,
            runtime_started_at: None,
        }
    }

    /// Collects current system statistics with agent status.
    pub fn collect_with_agent(active_sessions: u32, agent_status: AgentStatusInfo) -> Self {
        let mut stats = Self::collect(active_sessions);
        stats.agent_status = Some(agent_status);
        stats
    }

    /// Gets CPU usage percentage from /proc/loadavg.
    fn get_cpu_usage() -> f32 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/loadavg")
                .ok()
                .and_then(|c| c.split_whitespace().next()?.parse::<f32>().ok())
                .map(|load| {
                    let cpus = std::thread::available_parallelism()
                        .map(|p| p.get()).unwrap_or(1);
                    (load * 100.0 / cpus as f32).min(100.0)
                })
                .unwrap_or(0.0)
        }
        #[cfg(not(target_os = "linux"))]
        { 0.0 }
    }

    /// Gets memory usage and total memory (cgroup-aware).
    ///
    /// Priority: cgroup v2 → cgroup v1 → /proc/meminfo
    fn get_memory_info() -> (u64, Option<u64>) {
        #[cfg(target_os = "linux")]
        {
            if let Some(r) = Self::get_memory_cgroup_v2() { return r; }
            if let Some(r) = Self::get_memory_cgroup_v1() { return r; }
            Self::get_memory_procinfo()
        }
        #[cfg(not(target_os = "linux"))]
        { (0, None) }
    }

    #[cfg(target_os = "linux")]
    fn get_memory_cgroup_v2() -> Option<(u64, Option<u64>)> {
        let current: u64 = std::fs::read_to_string("/sys/fs/cgroup/memory.current")
            .ok()?.trim().parse().ok()?;
        let max = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok()
            .and_then(|s| {
                let t = s.trim();
                if t == "max" { None } else { t.parse::<u64>().ok() }
            });
        Some((current / 1024 / 1024, max.map(|m| m / 1024 / 1024)))
    }

    #[cfg(target_os = "linux")]
    fn get_memory_cgroup_v1() -> Option<(u64, Option<u64>)> {
        let usage: u64 = std::fs::read_to_string(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        ).ok()?.trim().parse().ok()?;
        let limit = std::fs::read_to_string(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        ).ok().and_then(|s| {
            let v: u64 = s.trim().parse().ok()?;
            if v > 1_000_000_000_000 { None } else { Some(v) }
        });
        Some((usage / 1024 / 1024, limit.map(|l| l / 1024 / 1024)))
    }

    #[cfg(target_os = "linux")]
    fn get_memory_procinfo() -> (u64, Option<u64>) {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                let mut total_kb = 0u64;
                let mut avail_kb = 0u64;
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        total_kb = line.split_whitespace().nth(1)?.parse().ok()?;
                    } else if line.starts_with("MemAvailable:") {
                        avail_kb = line.split_whitespace().nth(1)?.parse().ok()?;
                    }
                }
                Some((
                    total_kb.saturating_sub(avail_kb) / 1024,
                    Some(total_kb / 1024),
                ))
            })
            .unwrap_or((0, None))
    }

    /// Gets network I/O stats from /proc/net/dev.
    /// Sums rx/tx across all non-loopback interfaces.
    fn get_network_stats() -> (Option<u64>, Option<u64>) {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/net/dev")
                .ok()
                .map(|content| {
                    let mut total_rx = 0u64;
                    let mut total_tx = 0u64;
                    for line in content.lines().skip(2) {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        let parts: Vec<&str> = line.splitn(2, ':').collect();
                        if parts.len() != 2 { continue; }
                        if parts[0].trim() == "lo" { continue; }
                        let fields: Vec<&str> = parts[1].split_whitespace().collect();
                        if fields.len() < 10 { continue; }
                        if let (Ok(rx), Ok(tx)) = (
                            fields[0].parse::<u64>(),
                            fields[8].parse::<u64>(),
                        ) {
                            total_rx += rx;
                            total_tx += tx;
                        }
                    }
                    (Some(total_rx), Some(total_tx))
                })
                .unwrap_or((None, None))
        }
        #[cfg(not(target_os = "linux"))]
        { (None, None) }
    }
}

/// Response from heartbeat request.
#[derive(Debug, Deserialize)]
pub struct HeartbeatResponse {
    /// Whether the heartbeat was accepted
    pub success: bool,
    /// Suggested interval for next heartbeat (seconds)
    pub next_heartbeat_in: Option<u64>,
    /// v1.3.0: Structured commands from CMS to execute on this node.
    #[serde(default)]
    pub commands: Option<Vec<Command>>,
    /// Full operator wallet ban policy from CMS.
    #[serde(default)]
    pub operator_bans: Option<Vec<String>>,
    /// Error message (present on failure)
    pub error: Option<String>,
}

// ============================================
// Command Pipeline Models (v1.3.0)
// ============================================

/// A structured command dispatched from CMS to a Rust node.
///
/// ## Known Actions
/// - `"install_openclaw"` — Download and install OpenClaw agent
/// - `"start_openclaw"`   — Start the OpenClaw process
/// - `"stop_openclaw"`    — Stop the OpenClaw process
/// - `"uninstall_openclaw"` — Remove OpenClaw from the system
/// - `"update_openclaw"`  — Update OpenClaw to a new version
///
/// ⚠️ `id` MUST be included in all status reports back to CMS.
/// Unknown actions should be logged and reported as `failed`, never panic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// Unique command identifier (UUID from CMS) for lifecycle tracking.
    pub id: String,
    /// Action to perform.
    pub action: String,
    /// Action-specific parameters. Schema depends on `action`.
    #[serde(default = "default_empty_object")]
    pub params: serde_json::Value,
    /// Execution priority (lower = higher priority). Default: 10.
    #[serde(default = "default_priority")]
    pub priority: u8,
    /// ISO 8601 timestamp when the command was issued by CMS.
    #[serde(default)]
    pub issued_at: Option<String>,
}

fn default_empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

fn default_priority() -> u8 { 10 }

/// Status report sent from Rust -> CMS after executing a command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandStatusReport {
    /// The command ID this report refers to (from `Command.id`).
    pub command_id: String,
    /// Agent type identifier. Default: "openclaw".
    #[serde(default = "default_agent_type")]
    pub agent_type: String,
    /// Execution status.
    pub status: CommandExecutionStatus,
    /// Progress percentage (0–100). Meaningful for `in_progress` status.
    #[serde(default)]
    pub progress: u8,
    /// Human-readable status message for dashboard display.
    #[serde(default)]
    pub message: String,
    /// Unix timestamp of this status update.
    pub timestamp: u64,
}

fn default_agent_type() -> String { "openclaw".to_string() }

/// Execution status for a CMS command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandExecutionStatus {
    Received,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

// ============================================
// Agent Status Models (v1.3.0)
// ============================================

/// OpenClaw agent lifecycle state.
///
/// ⚠️ Do NOT rename variants — CMS relies on snake_case serialisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    NotInstalled,
    Installing,
    Stopped,
    Running,
    Error,
    Updating,
}

impl Default for AgentStatus {
    fn default() -> Self { Self::NotInstalled }
}

/// Detailed agent status information for heartbeat reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatusInfo {
    pub status: AgentStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_usage: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<u64>,
}

impl AgentStatusInfo {
    pub fn not_installed() -> Self {
        Self {
            status: AgentStatus::NotInstalled,
            version: None, message: None, pid: None,
            local_port: None, cpu_usage: None, memory_mb: None,
        }
    }

    pub fn running(version: String, pid: u32) -> Self {
        Self {
            status: AgentStatus::Running,
            version: Some(version),
            message: Some("Healthy".to_string()),
            pid: Some(pid),
            local_port: Some(18789),
            cpu_usage: None,
            memory_mb: None,
        }
    }

    pub fn installing(message: String) -> Self {
        Self {
            status: AgentStatus::Installing,
            version: None, message: Some(message), pid: None,
            local_port: None, cpu_usage: None, memory_mb: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            status: AgentStatus::Error,
            version: None, message: Some(message), pid: None,
            local_port: None, cpu_usage: None, memory_mb: None,
        }
    }
}

// ============================================
// Session Events
// ============================================

/// Session event type enumeration.
///
/// ## v1.0.0-TrafficAccounting
/// Added `SessionTrafficSnapshot` for periodic in-flight traffic reporting.
///
/// Long-lived sessions previously had zero traffic visibility until disconnect.
/// Now every 5 minutes a snapshot is sent with cumulative bytes_in / bytes_out
/// since session start.
///
/// ⚠️ Backend must treat `SessionTrafficSnapshot` as an upsert, NOT an
/// accumulation. Values are cumulative totals, not deltas.
/// Only `SessionEnded` (with `is_final=true`) closes the billing period.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventType {
    /// A new session was created.
    SessionCreated,
    /// An existing session was updated.
    SessionUpdated,
    /// A session has ended — triggers final billing reconciliation.
    /// Sent with `is_final = true`.
    SessionEnded,
    /// Periodic cumulative traffic snapshot for a live session (every 5 min).
    /// bytes_in / bytes_out are totals since session start.
    /// Backend must upsert, NOT accumulate.
    /// Sent with `is_final = false`.
    SessionTrafficSnapshot,
}

/// Session event report sent to CMS.
///
/// ## v1.0.0-TrafficAccounting
/// Added `is_final` to distinguish:
/// - `SessionEnded`            → `is_final = true`  → backend closes billing
/// - `SessionTrafficSnapshot`  → `is_final = false` → backend upserts live totals
///
/// `bytes_in` and `bytes_out` are always CUMULATIVE since session start.
/// Reports are idempotent — safe to re-send after a network error.
#[derive(Debug, Serialize)]
pub struct SessionEventReport {
    /// Event type.
    #[serde(rename = "type")]
    pub event_type: SessionEventType,
    /// Unique session identifier.
    pub session_id: String,
    /// Client's wallet address (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_wallet: Option<String>,
    /// Client's IP address (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<String>,
    /// Cumulative bytes received from client since session start.
    pub bytes_in: u64,
    /// Cumulative bytes sent to client since session start.
    pub bytes_out: u64,
    /// Unix timestamp of the event.
    pub timestamp: u64,
    /// Whether this is the final report for this session.
    /// true  → SessionEnded, backend closes billing period.
    /// false → SessionTrafficSnapshot, backend upserts live totals.
    pub is_final: bool,
    /// Unix timestamp for the last accepted client -> node VPN packet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_rx_at: Option<u64>,
    /// Unix timestamp for the last node -> client VPN packet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_tx_at: Option<u64>,
    /// Keepalive/ACK round-trip time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtt_ms: Option<f64>,
    /// Estimated packet loss percentage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packet_loss: Option<f64>,
    /// Packets rejected by the replay window as duplicates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_rejections: Option<u64>,
    /// Packets rejected by the replay window as too old.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub too_old_rejections: Option<u64>,
    /// Accepted VPN packets received from the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packets_rx: Option<u64>,
    /// VPN packets sent to the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packets_tx: Option<u64>,
}

/// Response from session event report.
#[derive(Debug, Deserialize)]
pub struct SessionReportResponse {
    /// Whether the report was accepted
    pub success: bool,
    /// Error message (present on failure)
    pub error: Option<String>,
}

// ============================================
// Persistent Node Info
// ============================================

/// Locally stored node information for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredNodeInfo {
    /// Unique node identifier
    pub node_id: String,
    /// Owner's wallet address
    pub owner_wallet: String,
    /// Human-readable node name
    pub name: String,
    /// ISO 8601 registration timestamp
    pub registered_at: String,
}

impl StoredNodeInfo {
    /// Loads stored node info from a file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Saves node info to a file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }
}
