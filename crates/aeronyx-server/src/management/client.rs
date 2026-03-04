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

Modification Reason (v1.3.0):
  - 🌟 Added `report_command_status()` for CommandHandler → CMS status feedback
  - 🐛 Replaced `eprintln!` debug output with `tracing::trace!` to stop polluting
    stderr in production (prints every 30 seconds)
  - 🌟 Refactored duplicate `json!` body construction into helper to reduce
    risk of signing/sending field mismatch

Main Logical Flow:
  1. create_signature() - Creates Ed25519 signature over message hash
  2. send_heartbeat() - Sends node status with signature verification
  3. report_session_event() - Reports session events to CMS
  4. 🌟 report_command_status() - Reports command execution status to CMS

⚠️ Important Note for Next Developer:
  - JSON field order MUST match Python backend expectation
  - The body_json used for signing MUST be the same as what's sent
  - Do NOT use HeartbeatRequest struct for sending - use json! macro directly
  - 🌟 `report_command_status()` uses the same signature scheme as heartbeat

Dependencies:
  - super::config::ManagementConfig
  - super::models::*
  - super::integrity (for version and binary hash)
  - aeronyx_core::crypto::IdentityKeyPair

Last Modified:
  v1.0.0 - Initial implementation
  v1.2.0 - Fixed JSON field ordering issue by using json! macro
  v1.3.0 - 🌟 Added report_command_status(), fixed eprintln!, refactored signing
============================================
*/

use std::time::{Duration, SystemTime, UNIX_EPOCH};
use reqwest::Client;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, trace, warn};
use serde_json::json;

use aeronyx_core::crypto::IdentityKeyPair;
use super::config::ManagementConfig;
use super::models::*;/// Management API client for communicating with the CMS backend.
///
/// Handles:
/// - Node registration with binding codes
/// - Periodic heartbeat with system stats
/// - Session event reporting
/// - 🌟 Command execution status reporting
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

        // 🐛 v1.3.0: Replaced eprintln! with trace! to stop polluting stderr
        trace!(
            node_id = %node_id,
            timestamp = timestamp,
            body_preview = %body.chars().take(200).collect::<String>(),
            "[SIGNATURE] Creating signature"
        );

        // SHA256 hash of the message
        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        let message_hash = hasher.finalize();

        trace!(
            hash = %hex::encode(&message_hash),
            "[SIGNATURE] Message hash computed"
        );

        // Ed25519 signature of the hash
        let signature = self.identity.sign(&message_hash);
        let sig_hex = hex::encode(signature);

        trace!(
            signature = %sig_hex,
            "[SIGNATURE] Signature created"
        );

        sig_hex
    }

    /// Helper: builds signed request headers.
    ///
    /// # Arguments
    /// * `node_id` - Hex-encoded public key
    /// * `timestamp` - Unix timestamp
    /// * `signature` - Hex-encoded Ed25519 signature
    fn signed_headers(node_id: &str, timestamp: u64, signature: &str) -> Vec<(&'static str, String)> {
        vec![
            ("X-Node-ID", node_id.to_string()),
            ("X-Timestamp", timestamp.to_string()),
            ("X-Signature", signature.to_string()),
        ]
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
    /// CRITICAL: Signature is computed over body WITHOUT the signature field.
    /// Python backend will reconstruct the same body (without signature) to verify.
    ///
    /// JSON field order for body (without signature):
    /// 1. node_id
    /// 2. timestamp
    /// 3. public_ip
    /// 4. version
    /// 5. binary_hash
    /// 6. system_stats (with cpu_usage, memory_mb, active_sessions)
    ///
    /// # Arguments
    /// * `public_ip` - Node's public IP address
    /// * `active_sessions` - Number of active client sessions
    ///
    /// # Returns
    /// HeartbeatResponse on success, error message on failure
    pub async fn send_heartbeat(
        &self,
        public_ip: &str,
        active_sessions: u32,
        agent_status: Option<AgentStatusInfo>,
    ) -> Result<HeartbeatResponse, String> {
        let url = format!("{}/node/heartbeat/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();
        let stats = SystemStats::collect(active_sessions);

        // CRITICAL: Build JSON WITHOUT signature field for signing.
        // This is what Python will reconstruct to verify the signature.
        //
        // 🌟 v1.3.1: Optional fields (agent_status, net_rx_bytes, etc.)
        // are conditionally injected so they don't break older CMS versions.
        let mut system_stats_json = serde_json::json!({
            "cpu_usage": stats.cpu_usage,
            "memory_mb": stats.memory_mb,
            "active_sessions": stats.active_sessions
        });

        if let Some(obj) = system_stats_json.as_object_mut() {
            if let Some(ref status) = agent_status {
                obj.insert(
                    "agent_status".to_string(),
                    serde_json::to_value(status).unwrap_or(serde_json::Value::Null),
                );
            }
            if let Some(rx) = stats.net_rx_bytes {
                obj.insert("net_rx_bytes".to_string(), serde_json::json!(rx));
            }
            if let Some(tx) = stats.net_tx_bytes {
                obj.insert("net_tx_bytes".to_string(), serde_json::json!(tx));
            }
            if let Some(total) = stats.memory_total_mb {
                obj.insert("memory_total_mb".to_string(), serde_json::json!(total));
            }
            if let Some(count) = stats.cpu_count {
                obj.insert("cpu_count".to_string(), serde_json::json!(count));
            }
        }

        let body_for_signing = json!({
            "node_id": &node_id,
            "timestamp": timestamp,
            "public_ip": public_ip,
            "version": super::integrity::get_version(),
            "binary_hash": &self.binary_hash,
            "system_stats": system_stats_json
        });

        // Serialize to string for signing (NO signature field)
        let body_str = serde_json::to_string(&body_for_signing).map_err(|e| e.to_string())?;

        // Create signature over the body
        let signature = self.create_signature(timestamp, &body_str);

        // Build final body by inserting signature into existing object
        let body_json = {
            let mut obj = body_for_signing;
            if let Some(map) = obj.as_object_mut() {
                map.insert("signature".to_string(), serde_json::Value::String(signature.clone()));
            }
            obj
        };

        debug!("Heartbeat: sessions={}", active_sessions);

        let mut request = self.http.post(&url);
        for (header, value) in Self::signed_headers(&node_id, timestamp, &signature) {
            request = request.header(header, value);
        }

        let response = request
            .json(&body_json)
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

        let mut request = self.http.post(&url);
        for (header, value) in Self::signed_headers(&node_id, timestamp, &signature) {
            request = request.header(header, value);
        }

        let response = request
            .json(&event)
            .send().await
            .map_err(|e| e.to_string())?;

        response.json().await.map_err(|e| e.to_string())
    }

    /// 🌟 v1.3.0: Reports command execution status to CMS.
    ///
    /// Called by `CommandHandler` after executing (or attempting) a command.
    /// Uses the same Ed25519 signature authentication as heartbeat requests.
    ///
    /// # Arguments
    /// * `report` - Command status report with ID, status, progress, message
    ///
    /// # Returns
    /// Ok(()) on success, Err(message) on failure
    ///
    /// # Endpoint
    /// `POST {cms_url}/node/agent/status/`
    pub async fn report_command_status(&self, report: &CommandStatusReport) -> Result<(), String> {
        let url = format!("{}/node/agent/status/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();
        let body_str = serde_json::to_string(report).map_err(|e| e.to_string())?;
        let signature = self.create_signature(timestamp, &body_str);

        debug!(
            command_id = %report.command_id,
            status = ?report.status,
            "[CMS_CLIENT] 📤 Reporting command status"
        );

        let mut request = self.http.post(&url);
        for (header, value) in Self::signed_headers(&node_id, timestamp, &signature) {
            request = request.header(header, value);
        }

        let response = request
            .json(report)
            .send().await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            warn!(
                command_id = %report.command_id,
                http_status = %status,
                "[CMS_CLIENT] ⚠️ Command status report failed: {}",
                body_text
            );
            return Err(format!("Status: {}, Body: {}", status, body_text));
        }

        debug!(
            command_id = %report.command_id,
            "[CMS_CLIENT] ✅ Command status reported"
        );

        Ok(())
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
