// ============================================
// FILE 3: crates/aeronyx-server/src/management/models.rs
// ============================================
//! # Management API Data Models

use serde::{Deserialize, Serialize};

// ============================================
// Node Registration
// ============================================

/// Request to bind a node using a registration code.
#[derive(Debug, Serialize)]
pub struct BindNodeRequest {
    pub code: String,
    pub public_key: String,
    pub hardware_info: HardwareInfo,
}

/// Hardware information collected from the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: String,
    pub memory: String,
    pub os: String,
}

impl HardwareInfo {
    /// Collects hardware info from the current system.
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
                    content
                        .lines()
                        .find(|line| line.starts_with("model name"))
                        .and_then(|line| line.split(':').nth(1))
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_else(|| "Unknown CPU".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        {
            "Unknown CPU".to_string()
        }
    }

    fn get_memory_info() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|line| line.starts_with("MemTotal"))
                        .and_then(|line| line.split(':').nth(1))
                        .map(|s| {
                            let kb: u64 = s
                                .trim()
                                .split_whitespace()
                                .next()
                                .and_then(|n| n.parse().ok())
                                .unwrap_or(0);
                            let gb = kb / 1024 / 1024;
                            format!("{}GB", gb)
                        })
                })
                .unwrap_or_else(|| "Unknown".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        {
            "Unknown".to_string()
        }
    }

    fn get_os_info() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/etc/os-release")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|line| line.starts_with("PRETTY_NAME"))
                        .and_then(|line| line.split('=').nth(1))
                        .map(|s| s.trim_matches('"').to_string())
                })
                .unwrap_or_else(|| "Linux".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        {
            "Unknown OS".to_string()
        }
    }
}

/// Response from node binding.
#[derive(Debug, Deserialize)]
pub struct BindNodeResponse {
    pub success: bool,
    pub node: Option<NodeInfo>,
    pub message: Option<String>,
    pub error: Option<String>,
}

/// Node information returned after binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub owner_wallet: String,
    pub name: String,
    pub public_key: String,
    pub status: String,
    pub created_at: String,
}

// ============================================
// Heartbeat
// ============================================

/// Heartbeat request sent periodically.
#[derive(Debug, Serialize)]
pub struct HeartbeatRequest {
    pub node_id: String,
    pub timestamp: u64,
    pub public_ip: String,
    pub version: String,
    pub binary_hash: String,
    pub system_stats: SystemStats,
    pub signature: String,
}

/// System statistics for heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub cpu_usage: f32,
    pub memory_mb: u64,
    pub active_sessions: u32,
}

impl SystemStats {
    /// Collects current system stats.
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
                .and_then(|content| {
                    content
                        .split_whitespace()
                        .next()
                        .and_then(|s| s.parse::<f32>().ok())
                })
                .map(|load| {
                    let cpus = std::thread::available_parallelism()
                        .map(|p| p.get())
                        .unwrap_or(1);
                    (load * 100.0 / cpus as f32).min(100.0)
                })
                .unwrap_or(0.0)
        }
        #[cfg(not(target_os = "linux"))]
        {
            0.0
        }
    }

    fn get_memory_usage_mb() -> u64 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    let mut total_kb = 0u64;
                    let mut available_kb = 0u64;
                    for line in content.lines() {
                        if line.starts_with("MemTotal:") {
                            total_kb = line
                                .split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0);
                        } else if line.starts_with("MemAvailable:") {
                            available_kb = line
                                .split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0);
                        }
                    }
                    if total_kb > 0 {
                        Some((total_kb - available_kb) / 1024)
                    } else {
                        None
                    }
                })
                .unwrap_or(0)
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }
}

/// Heartbeat response.
#[derive(Debug, Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub next_heartbeat_in: Option<u64>,
    pub commands: Option<Vec<String>>,
    pub error: Option<String>,
}

// ============================================
// Session Reporting
// ============================================

/// Session event types.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventType {
    SessionCreated,
    SessionUpdated,
    SessionEnded,
}

/// Session event report.
#[derive(Debug, Serialize)]
pub struct SessionEventReport {
    #[serde(rename = "type")]
    pub event_type: SessionEventType,
    pub session_id: String,
    pub client_wallet: Option<String>,
    pub client_ip: Option<String>,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub timestamp: u64,
}

/// Response from session report.
#[derive(Debug, Deserialize)]
pub struct SessionReportResponse {
    pub success: bool,
    pub error: Option<String>,
}

// ============================================
// Stored Node Info
// ============================================

/// Node info stored locally after registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredNodeInfo {
    pub node_id: String,
    pub owner_wallet: String,
    pub name: String,
    pub registered_at: String,
}

impl StoredNodeInfo {
    /// Loads stored node info from file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Saves node info to file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }
}


// ============================================
// FILE 4: crates/aeronyx-server/src/management/integrity.rs
// ============================================
//! # Binary Integrity Check

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tracing::{debug, warn};

/// Computes SHA256 hash of the current executable.
pub fn compute_binary_hash() -> String {
    match get_current_exe_path() {
        Some(path) => {
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let mut hasher = Sha256::new();
                    hasher.update(&bytes);
                    let result = hasher.finalize();
                    let hash = hex::encode(result);
                    debug!("Binary hash computed: {}", &hash[..16]);
                    hash
                }
                Err(e) => {
                    warn!("Failed to read binary for hash: {}", e);
                    "unable_to_read_binary".to_string()
                }
            }
        }
        None => {
            warn!("Failed to get current executable path");
            "unknown_binary".to_string()
        }
    }
}

fn get_current_exe_path() -> Option<PathBuf> {
    std::env::current_exe().ok()
}

/// Gets the current version from Cargo.toml.
pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}


// ============================================
// FILE 5: crates/aeronyx-server/src/management/client.rs
// ============================================
//! # Management API Client

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;

use super::config::ManagementConfig;
use super::models::*;

/// Management API client with Ed25519 authentication.
pub struct ManagementClient {
    config: ManagementConfig,
    http: Client,
    identity: IdentityKeyPair,
    binary_hash: String,
}

impl ManagementClient {
    /// Creates a new management client.
    pub fn new(config: ManagementConfig, identity: IdentityKeyPair) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        let binary_hash = super::integrity::compute_binary_hash();

        Self {
            config,
            http,
            identity,
            binary_hash,
        }
    }

    /// Returns the node's public key as hex string.
    pub fn node_id(&self) -> String {
        hex::encode(self.identity.public_key_bytes())
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Creates signature: Ed25519_Sign(SHA256(node_id + timestamp + body))
    fn create_signature(&self, timestamp: u64, body: &str) -> String {
        let node_id = self.node_id();
        let message = format!("{}{}{}", node_id, timestamp, body);
        
        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        let hash = hasher.finalize();
        
        let signature = self.identity.sign(&hash);
        hex::encode(signature)
    }

    /// Registers the node with a registration code.
    pub async fn register_node(&self, code: &str) -> Result<NodeInfo, String> {
        let url = format!("{}/node/bind/", self.config.cms_url);
        
        let request = BindNodeRequest {
            code: code.to_string(),
            public_key: self.node_id(),
            hardware_info: HardwareInfo::collect(),
        };

        info!("Registering node with code: {}...", &code[..code.len().min(8)]);

        let response = self.http
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        let status = response.status();
        let body: BindNodeResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        if body.success {
            if let Some(node) = body.node {
                info!("✅ Node registered successfully!");
                return Ok(node);
            }
        }

        let error_msg = body.error
            .or(body.message)
            .unwrap_or_else(|| format!("Registration failed with status {}", status));
        
        error!("❌ Node registration failed: {}", error_msg);
        Err(error_msg)
    }

    /// Sends a heartbeat to the CMS.
    pub async fn send_heartbeat(
        &self,
        public_ip: &str,
        active_sessions: u32,
    ) -> Result<HeartbeatResponse, String> {
        let url = format!("{}/node/heartbeat/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();

        let stats = SystemStats::collect(active_sessions);
        
        let body_for_signing = serde_json::json!({
            "node_id": node_id,
            "timestamp": timestamp,
            "public_ip": public_ip,
            "version": super::integrity::get_version(),
            "binary_hash": self.binary_hash,
            "system_stats": stats,
        });
        let body_str = serde_json::to_string(&body_for_signing)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;

        let signature = self.create_signature(timestamp, &body_str);

        let request = HeartbeatRequest {
            node_id: node_id.clone(),
            timestamp,
            public_ip: public_ip.to_string(),
            version: super::integrity::get_version().to_string(),
            binary_hash: self.binary_hash.clone(),
            system_stats: stats,
            signature: signature.clone(),
        };

        debug!(
            "Sending heartbeat: sessions={}, cpu={:.1}%",
            active_sessions, request.system_stats.cpu_usage
        );

        let response = self.http
            .post(&url)
            .header("X-Node-ID", &node_id)
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Signature", &signature)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Heartbeat request failed: {}", e))?;

        let status = response.status();
        
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            warn!("Heartbeat failed: {} - {}", status, error_text);
            return Err(format!("Heartbeat failed: {}", status));
        }

        response
            .json()
            .await
            .map_err(|e| format!("Failed to parse heartbeat response: {}", e))
    }

    /// Reports a session event to the CMS.
    pub async fn report_session_event(
        &self,
        event: SessionEventReport,
    ) -> Result<SessionReportResponse, String> {
        let url = format!("{}/node/sessions/report/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();

        let body_str = serde_json::to_string(&event)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;

        let signature = self.create_signature(timestamp, &body_str);

        let response = self.http
            .post(&url)
            .header("X-Node-ID", &node_id)
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Signature", &signature)
            .json(&event)
            .send()
            .await
            .map_err(|e| format!("Session report failed: {}", e))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(format!("Session report failed: {} - {}", status, error_text));
        }

        response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ManagementConfig {
        &self.config
    }
}

impl std::fmt::Debug for ManagementClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagementClient")
            .field("cms_url", &self.config.cms_url)
            .field("node_id", &format!("{}...", &self.node_id()[..16]))
            .finish()
    }
}
