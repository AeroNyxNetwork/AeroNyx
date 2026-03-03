//! ============================================
//! File: crates/aeronyx-server/src/management/models.rs
//! Path: aeronyx-server/src/management/models.rs
//! ============================================
//! Purpose: Management API data models for CMS communication
//!
//! Key Fix v1.2.0:
//!   - HeartbeatRequest struct is kept for documentation but NOT used for sending
//!   - Actual heartbeat uses json! macro in client.rs to preserve field order
//!   - SessionEventReport uses String for event_type (not enum) for flexibility
//!
//! Modification Reason (v1.3.0):
//!   - 🌟 Added `Command` struct for CMS command dispatch (Phase 1: Command Pipeline)
//!   - 🌟 Upgraded `HeartbeatResponse.commands` from `Option<Vec<String>>`
//!     to `Option<Vec<Command>>` to support structured commands with params
//!   - 🌟 Added `AgentStatus` enum and `AgentStatusInfo` for OpenClaw lifecycle tracking
//!   - 🌟 Added `CommandStatusReport` for Rust → CMS command execution feedback
//!   - 🌟 Added `agent_status` to `SystemStats` for heartbeat-based status reporting
//!
//! Main Data Structures:
//!   - BindNodeRequest/Response: Node registration
//!   - HeartbeatRequest/Response: Periodic status updates
//!   - SessionEventReport/Response: Session event reporting
//!   - HardwareInfo: System hardware information
//!   - SystemStats: Runtime system statistics
//!   - 🌟 Command: CMS → Rust structured command with params
//!   - 🌟 AgentStatus / AgentStatusInfo: OpenClaw agent lifecycle state
//!   - 🌟 CommandStatusReport: Rust → CMS command execution result
//!
//! ⚠️ Important Note for Next Developer:
//!   - HeartbeatRequest struct field order is documented but struct is NOT used
//!     for actual HTTP request
//!   - The client.rs uses json! macro directly to ensure field order
//!   - If you need to modify heartbeat fields, update BOTH models.rs AND client.rs
//!     json! calls
//!   - 🌟 `Command.id` is used by CMS to track command lifecycle — always include
//!     it in status reports
//!   - 🌟 `Command.params` is `serde_json::Value` for maximum flexibility — each
//!     action handler should define its own expected param schema
//!   - 🌟 `AgentStatus` default is `NotInstalled` — do NOT change the variant names
//!     as CMS relies on snake_case serialisation
//!
//! Dependencies:
//!   - serde for serialization/deserialization
//!   - serde_json for `Command.params` (dynamic JSON value)
//!   - System files (/proc/cpuinfo, /proc/meminfo, etc.) for Linux system information
//!
//! Last Modified:
//!   v1.0.0 - Initial models
//!   v1.2.0 - Clarified that HeartbeatRequest is for documentation only
//!   v1.3.0 - 🌟 Added Command, AgentStatus, AgentStatusInfo, CommandStatusReport
//! ============================================

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
/// Do NOT use this struct directly with .json(&request) - it will serialize
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
/// CRITICAL: Field order MUST match Python backend expectation:
/// 1. cpu_usage
/// 2. memory_mb
/// 3. active_sessions
/// 4. agent_status (🌟 v1.3.0 — optional, omitted if None for backward compat)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    /// CPU usage percentage (0.0 - 100.0)
    pub cpu_usage: f32,        // 1st
    /// Memory usage in megabytes
    pub memory_mb: u64,        // 2nd
    /// Number of active VPN sessions
    pub active_sessions: u32,  // 3rd

    /// 🌟 v1.3.0: OpenClaw agent status for CMS dashboard display.
    /// `None` = field omitted from JSON (backward compatible with old CMS).
    /// `Some(info)` = agent lifecycle state + optional detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_status: Option<AgentStatusInfo>,
}

impl SystemStats {
    /// Collects current system statistics.
    ///
    /// # Arguments
    /// * `active_sessions` - Number of active client sessions
    pub fn collect(active_sessions: u32) -> Self {
        Self {
            cpu_usage: Self::get_cpu_usage(),
            memory_mb: Self::get_memory_usage_mb(),
            active_sessions,
            agent_status: None, // Will be populated by server if AgentManager is active
        }
    }

    /// Collects current system statistics with agent status.
    ///
    /// # Arguments
    /// * `active_sessions` - Number of active client sessions
    /// * `agent_status` - Current OpenClaw agent status
    pub fn collect_with_agent(active_sessions: u32, agent_status: AgentStatusInfo) -> Self {
        Self {
            cpu_usage: Self::get_cpu_usage(),
            memory_mb: Self::get_memory_usage_mb(),
            active_sessions,
            agent_status: Some(agent_status),
        }
    }

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

    fn get_memory_usage_mb() -> u64 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    let mut total = 0u64;
                    let mut avail = 0u64;
                    for line in content.lines() {
                        if line.starts_with("MemTotal:") {
                            total = line.split_whitespace().nth(1)?.parse().ok()?;
                        } else if line.starts_with("MemAvailable:") {
                            avail = line.split_whitespace().nth(1)?.parse().ok()?;
                        }
                    }
                    Some((total - avail) / 1024)
                })
                .unwrap_or(0)
        }
        #[cfg(not(target_os = "linux"))]
        { 0 }
    }
}

/// Response from heartbeat request.
///
/// 🌟 v1.3.0: `commands` upgraded from `Vec<String>` to `Vec<Command>`
/// for structured command dispatch with params and tracking IDs.
///
/// CMS backward compatibility: if CMS sends no `commands` field or sends
/// `null`, serde will deserialise it as `None`. If CMS sends an empty
/// array `[]`, it becomes `Some(vec![])`.
#[derive(Debug, Deserialize)]
pub struct HeartbeatResponse {
    /// Whether the heartbeat was accepted
    pub success: bool,
    /// Suggested interval for next heartbeat (seconds)
    pub next_heartbeat_in: Option<u64>,
    /// 🌟 v1.3.0: Structured commands from CMS to execute on this node.
    /// Each command has a unique `id` for lifecycle tracking.
    #[serde(default)]
    pub commands: Option<Vec<Command>>,
    /// Error message (present on failure)
    pub error: Option<String>,
}

// ============================================
// 🌟 v1.3.0: Command Pipeline Models
// ============================================

/// A structured command dispatched from CMS to a Rust node.
///
/// ## Lifecycle
/// 1. CMS creates a command and places it in the node's queue
/// 2. Rust receives it via `HeartbeatResponse.commands`
/// 3. Rust executes the command and reports status via `CommandStatusReport`
/// 4. CMS updates the command status for the Next.js dashboard
///
/// ## Known Actions
/// - `"install_openclaw"` — Download and install OpenClaw agent
/// - `"start_openclaw"` — Start the OpenClaw process
/// - `"stop_openclaw"` — Stop the OpenClaw process
/// - `"uninstall_openclaw"` — Remove OpenClaw from the system
/// - `"update_openclaw"` — Update OpenClaw to a new version
///
/// ## Example JSON from CMS
/// ```json
/// {
///     "id": "cmd-550e8400-e29b-41d4-a716-446655440000",
///     "action": "install_openclaw",
///     "params": {
///         "version": "1.0.0",
///         "download_url": "https://releases.openclaw.ai/v1.0.0/openclaw-linux-amd64"
///     },
///     "priority": 1,
///     "issued_at": "2026-03-03T15:00:00Z"
/// }
/// ```
///
/// ⚠️ Important:
/// - `id` MUST be included in all status reports back to CMS
/// - `params` schema varies by action — handlers should validate internally
/// - Unknown actions should be logged and reported as `failed` (not panic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// Unique command identifier (UUID from CMS) for lifecycle tracking.
    pub id: String,

    /// Action to perform (e.g., "install_openclaw", "stop_openclaw").
    pub action: String,

    /// Action-specific parameters. Schema depends on `action`.
    /// Use `serde_json::Value` for maximum flexibility.
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

fn default_priority() -> u8 {
    10
}

/// Status report sent from Rust → CMS after executing (or attempting) a command.
///
/// Sent via `POST /node/agent/status` with the same Ed25519 signature auth
/// as heartbeat requests.
///
/// ## Example JSON
/// ```json
/// {
///     "command_id": "cmd-550e8400-e29b-41d4-a716-446655440000",
///     "agent_type": "openclaw",
///     "status": "in_progress",
///     "progress": 45,
///     "message": "Downloading OpenClaw binary...",
///     "timestamp": 1709474400
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandStatusReport {
    /// The command ID this report refers to (from `Command.id`).
    pub command_id: String,

    /// Agent type identifier. CMS uses this to route the status
    /// to the correct agent handler. Default: "openclaw".
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

fn default_agent_type() -> String {
    "openclaw".to_string()
}

/// Execution status for a CMS command.
///
/// Serialised as snake_case strings for CMS compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandExecutionStatus {
    /// Command received and queued for execution.
    Received,
    /// Command is currently executing.
    InProgress,
    /// Command completed successfully.
    Completed,
    /// Command failed (see `message` for details).
    Failed,
    /// Command was cancelled (e.g., superseded by a newer command).
    Cancelled,
}

// ============================================
// 🌟 v1.3.0: Agent Status Models
// ============================================

/// OpenClaw agent lifecycle state.
///
/// Reported in `SystemStats.agent_status` during heartbeat,
/// and used internally by `AgentManager` to track state.
///
/// ⚠️ Do NOT rename variants — CMS relies on snake_case serialisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// Agent is not installed on this node.
    NotInstalled,
    /// Agent is currently being downloaded/installed.
    Installing,
    /// Agent is installed but not running.
    Stopped,
    /// Agent is running and healthy.
    Running,
    /// Agent process has crashed or is unresponsive.
    Error,
    /// Agent is being updated to a new version.
    Updating,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::NotInstalled
    }
}

/// Detailed agent status information for heartbeat reporting.
///
/// Combines the high-level `AgentStatus` enum with optional detail
/// fields for richer dashboard display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatusInfo {
    /// High-level lifecycle state.
    pub status: AgentStatus,

    /// Agent version string (e.g., "1.0.0"), if installed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Human-readable detail message (e.g., "Downloading 45%", "Healthy").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,

    /// PID of the running agent process, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
}

impl AgentStatusInfo {
    /// Creates a status info indicating the agent is not installed.
    pub fn not_installed() -> Self {
        Self {
            status: AgentStatus::NotInstalled,
            version: None,
            message: None,
            pid: None,
        }
    }

    /// Creates a status info indicating the agent is running.
    pub fn running(version: String, pid: u32) -> Self {
        Self {
            status: AgentStatus::Running,
            version: Some(version),
            message: Some("Healthy".to_string()),
            pid: Some(pid),
        }
    }

    /// Creates a status info indicating the agent is installing.
    pub fn installing(message: String) -> Self {
        Self {
            status: AgentStatus::Installing,
            version: None,
            message: Some(message),
            pid: None,
        }
    }

    /// Creates a status info indicating the agent has errored.
    pub fn error(message: String) -> Self {
        Self {
            status: AgentStatus::Error,
            version: None,
            message: Some(message),
            pid: None,
        }
    }
}

// ============================================
// Session Events
// ============================================

/// Session event type enumeration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventType {
    /// A new session was created
    SessionCreated,
    /// An existing session was updated
    SessionUpdated,
    /// A session has ended
    SessionEnded,
}

/// Session event report sent to CMS.
#[derive(Debug, Serialize)]
pub struct SessionEventReport {
    /// Event type: session_created, session_updated, or session_ended
    #[serde(rename = "type")]
    pub event_type: SessionEventType,
    /// Unique session identifier
    pub session_id: String,
    /// Client's wallet address (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_wallet: Option<String>,
    /// Client's IP address (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<String>,
    /// Bytes received from client
    pub bytes_in: u64,
    /// Bytes sent to client
    pub bytes_out: u64,
    /// Unix timestamp of the event
    pub timestamp: u64,
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
    ///
    /// # Arguments
    /// * `path` - Path to the JSON file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Saves node info to a file.
    ///
    /// # Arguments
    /// * `path` - Path to save the JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }
}
