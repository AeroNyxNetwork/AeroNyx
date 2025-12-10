/*
============================================
File: crates/aeronyx-server/src/management/client.rs
Path: aeronyx-server/src/management/client.rs
============================================
Purpose: Management API Client for node registration, heartbeat, and session reporting

Key Fix v1.2.0:
  - Use serde_json::json! macro with preserve_order feature to maintain field order
  - Send the same JSON object used for signing (not a separate struct)
  - This ensures signature verification passes on Python backend

Main Logical Flow:
  1. create_signature() - Creates Ed25519 signature over message hash
  2. send_heartbeat() - Sends node status with signature verification
  3. report_session_event() - Reports session events to CMS

⚠️ Important Note for Next Developer:
  - JSON field order MUST match Python backend expectation
  - The body_json used for signing MUST be the same as what's sent
  - Do NOT use HeartbeatRequest struct for sending - use json! macro directly

Dependencies:
  - super::config::ManagementConfig
  - super::models::*
  - super::integrity (for version and binary hash)
  - aeronyx_core::crypto::IdentityKeyPair

Last Modified: v1.2.0 - Fixed JSON field ordering issue by using json! macro for both signing and sending
============================================
*/

use std::time::{Duration, SystemTime, UNIX_EPOCH};
use reqwest::Client;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};
use serde_json::json;

use aeronyx_core::crypto::IdentityKeyPair;
use super::config::ManagementConfig;
use super::models::*;

/// Management API client for communicating with the CMS backend.
/// 
/// Handles:
/// - Node registration with binding codes
/// - Periodic heartbeat with system stats
/// - Session event reporting
pub struct ManagementClient {
    config: ManagementConfig,
    http: Client,
    identity: IdentityKeyPair,
    binary_hash: String,
}

impl ManagementClient {
    /// Creates a new ManagementClient instance.
    /// 
    /// # Arguments
    /// * `config` - Management configuration containing CMS URL and timeouts
    /// * `identity` - Ed25519 identity keypair for signing requests
    pub fn new(config: ManagementConfig, identity: IdentityKeyPair) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");
        let binary_hash = super::integrity::compute_binary_hash();
        Self { config, http, identity, binary_hash }
    }

    /// Returns the node ID (hex-encoded public key).
    pub fn node_id(&self) -> String {
        hex::encode(self.identity.public_key_bytes())
    }

    /// Returns current Unix timestamp in seconds.
    fn current_timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    /// Creates an Ed25519 signature over the message.
    /// 
    /// Signature process:
    /// 1. Concatenate: node_id + timestamp + body
    /// 2. SHA256 hash the concatenated message
    /// 3. Sign the hash with Ed25519
    /// 
    /// # Arguments
    /// * `timestamp` - Unix timestamp used in the message
    /// * `body` - JSON body string to sign
    /// 
    /// # Returns
    /// Hex-encoded signature string
    fn create_signature(&self, timestamp: u64, body: &str) -> String {
        let node_id = self.node_id();
        
        // Message format: node_id + timestamp + body (no separators)
        let message = format!("{}{}{}", node_id, timestamp, body);
        
        // Debug output for signature verification troubleshooting
        eprintln!("============================================================");
        eprintln!("[Rust Signature] Node ID: {}", node_id);
        eprintln!("[Rust Signature] Timestamp: {}", timestamp);
        eprintln!("[Rust Signature] Body (first 200): {}", &body.chars().take(200).collect::<String>());
        eprintln!("[Rust Signature] Message (first 200): {}", &message.chars().take(200).collect::<String>());
        
        // SHA256 hash of the message
        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        let message_hash = hasher.finalize();
        let hash_hex = hex::encode(&message_hash);
        
        eprintln!("[Rust Signature] Message hash: {}", hash_hex);
        
        // Ed25519 signature of the hash
        let signature = self.identity.sign(&message_hash);
        let sig_hex = hex::encode(signature);
        
        eprintln!("[Rust Signature] Signature: {}", sig_hex);
        eprintln!("============================================================");
        
        sig_hex
    }

    /// Registers the node with the CMS using a binding code.
    /// 
    /// # Arguments
    /// * `code` - The binding code from the CMS dashboard
    /// 
    /// # Returns
    /// NodeInfo on success, error message on failure
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

    /// Sends a heartbeat to the CMS with current node status.
    /// 
    /// CRITICAL: JSON field order MUST match Python backend expectation:
    /// 1. node_id
    /// 2. timestamp
    /// 3. public_ip
    /// 4. version
    /// 5. binary_hash
    /// 6. system_stats (with cpu_usage, memory_mb, active_sessions)
    /// 7. signature
    /// 
    /// # Arguments
    /// * `public_ip` - Node's public IP address
    /// * `active_sessions` - Number of active client sessions
    /// 
    /// # Returns
    /// HeartbeatResponse on success, error message on failure
    pub async fn send_heartbeat(&self, public_ip: &str, active_sessions: u32) -> Result<HeartbeatResponse, String> {
        let url = format!("{}/node/heartbeat/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();
        let stats = SystemStats::collect(active_sessions);
    
        // CRITICAL: Build JSON with exact field order matching Python backend
        // Using json! macro preserves insertion order when serde_json has "preserve_order" feature
        // 
        // Step 1: Create body with empty signature for signing
        let body_for_signing = json!({
            "node_id": node_id,
            "timestamp": timestamp,
            "public_ip": public_ip,
            "version": super::integrity::get_version(),
            "binary_hash": &self.binary_hash,
            "system_stats": {
                "cpu_usage": stats.cpu_usage,
                "memory_mb": stats.memory_mb,
                "active_sessions": stats.active_sessions
            },
            "signature": ""
        });
        
        // Serialize to string for signing
        let body_str = serde_json::to_string(&body_for_signing).map_err(|e| e.to_string())?;
        
        // Create signature over the body
        let signature = self.create_signature(timestamp, &body_str);

        // Step 2: Create final body with actual signature
        // IMPORTANT: Must use json! macro again to preserve field order
        let body_json = json!({
            "node_id": node_id,
            "timestamp": timestamp,
            "public_ip": public_ip,
            "version": super::integrity::get_version(),
            "binary_hash": &self.binary_hash,
            "system_stats": {
                "cpu_usage": stats.cpu_usage,
                "memory_mb": stats.memory_mb,
                "active_sessions": stats.active_sessions
            },
            "signature": &signature
        });

        debug!("Heartbeat: sessions={}", active_sessions);

        // Send the json! constructed value directly - NOT a HeartbeatRequest struct
        // This ensures field order is preserved
        let response = self.http.post(&url)
            .header("X-Node-ID", &node_id)
            .header("X-Timestamp", timestamp.to_string())
            .header("X-Signature", &signature)
            .json(&body_json)  // Use body_json, NOT a struct
            .send().await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            error!("Heartbeat failed: status={}, body={}", status, body_text);
            return Err(format!("Status: {}, Body: {}", status, body_text));
        }

        response.json().await.map_err(|e| e.to_string())
    }

    /// Reports a session event (create/update/end) to the CMS.
    /// 
    /// # Arguments
    /// * `event` - The session event details
    /// 
    /// # Returns
    /// SessionReportResponse on success, error message on failure
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

    /// Returns a reference to the management configuration.
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
