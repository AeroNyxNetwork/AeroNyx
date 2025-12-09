// ============================================
// FILE 2: crates/aeronyx-server/src/management/config.rs
// ============================================
//! # Management Configuration

use serde::{Deserialize, Serialize};

/// Management system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementConfig {
    /// CMS API base URL.
    #[serde(default = "default_cms_url")]
    pub cms_url: String,

    /// Heartbeat reporting interval in seconds.
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,

    /// Session report batch interval in seconds.
    #[serde(default = "default_session_report_interval")]
    pub session_report_interval_secs: u64,

    /// HTTP request timeout in seconds.
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Maximum retry attempts for failed requests.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Path to store node binding info.
    #[serde(default = "default_node_info_path")]
    pub node_info_path: String,
}

fn default_cms_url() -> String {
    "https://api.aeronyx.network/api/privacy_network".to_string()
}

fn default_heartbeat_interval() -> u64 {
    30
}

fn default_session_report_interval() -> u64 {
    60
}

fn default_request_timeout() -> u64 {
    10
}

fn default_max_retries() -> u32 {
    3
}

fn default_node_info_path() -> String {
    "/etc/aeronyx/node_info.json".to_string()
}

impl Default for ManagementConfig {
    fn default() -> Self {
        Self {
            cms_url: default_cms_url(),
            heartbeat_interval_secs: default_heartbeat_interval(),
            session_report_interval_secs: default_session_report_interval(),
            request_timeout_secs: default_request_timeout(),
            max_retries: default_max_retries(),
            node_info_path: default_node_info_path(),
        }
    }
}

impl ManagementConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.cms_url.is_empty() {
            return Err("cms_url cannot be empty".to_string());
        }
        if self.heartbeat_interval_secs == 0 {
            return Err("heartbeat_interval_secs must be > 0".to_string());
        }
        if self.request_timeout_secs == 0 {
            return Err("request_timeout_secs must be > 0".to_string());
        }
        Ok(())
    }

    /// Checks if node is registered.
    pub fn is_registered(&self) -> bool {
        std::path::Path::new(&self.node_info_path).exists()
    }
}
