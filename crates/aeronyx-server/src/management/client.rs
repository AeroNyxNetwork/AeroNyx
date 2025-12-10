//! # Management API Client

use std::time::{Duration, SystemTime, UNIX_EPOCH};
use reqwest::Client;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use super::config::ManagementConfig;
use super::models::*;

pub struct ManagementClient {
    config: ManagementConfig,
    http: Client,
    identity: IdentityKeyPair,
    binary_hash: String,
}

impl ManagementClient {
    pub fn new(config: ManagementConfig, identity: IdentityKeyPair) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");
        let binary_hash = super::integrity::compute_binary_hash();
        Self { config, http, identity, binary_hash }
    }

    pub fn node_id(&self) -> String {
        hex::encode(self.identity.public_key_bytes())
    }

    fn current_timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    fn create_signature(&self, timestamp: u64, body: &str) -> String {
        let node_id = self.node_id();
        
     
        let message = format!("{}{}{}", node_id, timestamp, body);
        
        eprintln!("============================================================");
        eprintln!("[Rust Signature] Node ID: {}", node_id);
        eprintln!("[Rust Signature] Timestamp: {}", timestamp);
        eprintln!("[Rust Signature] Body (first 200): {}", &body.chars().take(200).collect::<String>());
        eprintln!("[Rust Signature] Message (first 200): {}", &message.chars().take(200).collect::<String>());
        

        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        let message_hash = hasher.finalize();
        let hash_hex = hex::encode(&message_hash);
        
        eprintln!("[Rust Signature] Message hash: {}", hash_hex);
        

        let signature = self.identity.sign(&message_hash);
        let sig_hex = hex::encode(signature);
        
        eprintln!("[Rust Signature] Signature: {}", sig_hex);
        eprintln!("============================================================");
        
        sig_hex
    }

    pub async fn register_node(&self, code: &str) -> Result<NodeInfo, String> {
        let url = format!("{}/node/bind/", self.config.cms_url);
        let request = BindNodeRequest {
            code: code.to_string(),
            public_key: self.node_id(),
            hardware_info: HardwareInfo::collect(),
        };

        info!("Registering node...");

        let response = self.http.post(&url).json(&request).send().await
            .map_err(|e| format!("Request failed: {}", e))?;

        let body: BindNodeResponse = response.json().await
            .map_err(|e| format!("Parse failed: {}", e))?;

        if body.success {
            if let Some(node) = body.node {
                info!("Node registered!");
                return Ok(node);
            }
        }

        Err(body.error.or(body.message).unwrap_or_else(|| "Unknown error".to_string()))
    }

    pub async fn send_heartbeat(&self, public_ip: &str, active_sessions: u32) -> Result<HeartbeatResponse, String> {
        let url = format!("{}/node/heartbeat/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();
        let stats = SystemStats::collect(active_sessions);
    
        use serde_json::json;
        let body_for_signing = json!({
            "node_id": node_id,
            "timestamp": timestamp,
            "public_ip": public_ip,
            "version": super::integrity::get_version(),
            "binary_hash": self.binary_hash,
            "system_stats": {
                "cpu_usage": stats.cpu_usage,
                "memory_mb": stats.memory_mb,
                "active_sessions": stats.active_sessions
            },
            "signature": ""
        });
        let body_str = serde_json::to_string(&body_for_signing).map_err(|e| e.to_string())?;
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

        debug!("Heartbeat: sessions={}", active_sessions);

        let response = self.http.post(&url)
            .header("X-Node-ID", &node_id)
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Signature", &signature)
            .json(&request)
            .send().await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Status: {}", response.status()));
        }

        response.json().await.map_err(|e| e.to_string())
    }

    pub async fn report_session_event(&self, event: SessionEventReport) -> Result<SessionReportResponse, String> {
        let url = format!("{}/node/sessions/report/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();
        let body_str = serde_json::to_string(&event).map_err(|e| e.to_string())?;
        let signature = self.create_signature(timestamp, &body_str);

        let response = self.http.post(&url)
            .header("X-Node-ID", &node_id)
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Signature", &signature)
            .json(&event)
            .send().await
            .map_err(|e| e.to_string())?;

        response.json().await.map_err(|e| e.to_string())
    }

    pub fn config(&self) -> &ManagementConfig {
        &self.config
    }
}

impl std::fmt::Debug for ManagementClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagementClient")
            .field("cms_url", &self.config.cms_url)
            .finish()
    }
}
