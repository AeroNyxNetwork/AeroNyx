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
//! Main Data Structures:
//!   - BindNodeRequest/Response: Node registration
//!   - HeartbeatRequest/Response: Periodic status updates
//!   - SessionEventReport/Response: Session event reporting
//!   - HardwareInfo: System hardware information
//!   - SystemStats: Runtime system statistics
//!
//! ⚠️ Important Note for Next Developer:
//!   - HeartbeatRequest struct field order is documented but struct is NOT used for actual HTTP request
//!   - The client.rs uses json! macro directly to ensure field order
//!   - If you need to modify heartbeat fields, update BOTH models.rs AND client.rs json! calls
//!
//! Dependencies:
//!   - serde for serialization/deserialization
//!   - System files (/proc/cpuinfo, /proc/meminfo, etc.) for Linux system information
//!
//! Last Modified: v1.2.0 - Clarified that HeartbeatRequest is for documentation only
//! ============================================

use serde::{Deserialize, Serialize};

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
    pub success: bool,
    pub node: Option<NodeInfo>,
    pub message: Option<String>,
    pub error: Option<String>,
}

/// Information about a registered node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub owner_wallet: String,
    pub name: String,
    pub public_key: String,
    pub status: String,
    pub created_at: String,
}

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub cpu_usage: f32,        // 1st
    pub memory_mb: u64,        // 2nd
    pub active_sessions: u32,  // 3rd
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
#[derive(Debug, Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub next_heartbeat_in: Option<u64>,
    pub commands: Option<Vec<String>>,
    pub error: Option<String>,
}

/// Session event type enumeration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventType {
    SessionCreated,
    SessionUpdated,
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
    pub success: bool,
    pub error: Option<String>,
}

/// Locally stored node information for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredNodeInfo {
    pub node_id: String,
    pub owner_wallet: String,
    pub name: String,
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
