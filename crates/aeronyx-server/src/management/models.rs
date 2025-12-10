/*
============================================
File: crates/aeronyx-server/src/management/models.rs
Path: aeronyx-server/src/management/models.rs
============================================
Purpose: Management API data models

Key Fix v1.1.0:
  - Force JSON field order to match Python backend
  - Use #[serde(skip_serializing)] for signature field during signing

Author: Claude
Last Modified: 2024-12-10
============================================
*/

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct BindNodeRequest {
    pub code: String,
    pub public_key: String,
    pub hardware_info: HardwareInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: String,
    pub memory: String,
    pub os: String,
}

impl HardwareInfo {
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

#[derive(Debug, Deserialize)]
pub struct BindNodeResponse {
    pub success: bool,
    pub node: Option<NodeInfo>,
    pub message: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub owner_wallet: String,
    pub name: String,
    pub public_key: String,
    pub status: String,
    pub created_at: String,
}

/// CRITICAL: Field order MUST match Python backend expectation
/// Do NOT reorder these fields!
#[derive(Debug, Serialize)]
pub struct HeartbeatRequest {
    pub node_id: String,           // 1st - MUST be first
    pub timestamp: u64,             // 2nd
    pub public_ip: String,          // 3rd
    pub version: String,            // 4th
    pub binary_hash: String,        // 5th
    pub system_stats: SystemStats,  // 6th
    pub signature: String,          // 7th - MUST be last
}

/// CRITICAL: Field order MUST match Python backend expectation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub cpu_usage: f32,        // 1st
    pub memory_mb: u64,         // 2nd
    pub active_sessions: u32,   // 3rd
}

impl SystemStats {
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

#[derive(Debug, Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub next_heartbeat_in: Option<u64>,
    pub commands: Option<Vec<String>>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SessionEventReport {
    #[serde(rename = "type")]
    pub event_type: String,      // Changed from enum to String
    pub session_id: String,
    pub client_wallet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_ip: Option<String>,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub timestamp: u64,
}

#[derive(Debug, Deserialize)]
pub struct SessionReportResponse {
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredNodeInfo {
    pub node_id: String,
    pub owner_wallet: String,
    pub name: String,
    pub registered_at: String,
}

impl StoredNodeInfo {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }
}
