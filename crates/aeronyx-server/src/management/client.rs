// ============================================
// File: crates/aeronyx-server/src/management/client.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Added TrafficDelta and UserPermission types.
//   Extended HeartbeatResponse with node_tier + user_permissions.
//   send_heartbeat() now accepts connected_wallets + traffic_delta
//   and injects them into the signed body.
//
// Main Logical Flow:
//   1. create_signature() - Ed25519 over SHA256(node_id + timestamp + body)
//   2. send_heartbeat()   - signed heartbeat with membership fields
//   3. report_session_event() - session lifecycle events
//   4. report_command_status() - command execution feedback to CMS
//
// ⚠️ Important Notes for Next Developer:
//   - connected_wallets + traffic_delta are injected into the TOP-LEVEL
//     signed body (not inside system_stats).
//   - Both use skip_serializing_if — backward compatible with old CMS.
//   - HeartbeatResponse new fields use #[serde(default)] — old CMS
//     responses without these fields deserialize cleanly.
//   - wallet_hex keys must be lowercase (hex::encode output).
//   - The body used for signing MUST be the same as what is sent.
//     Do NOT add fields after signing.
//
// Last Modified:
//   v1.0.0         - Initial implementation
//   v1.2.0         - Fixed JSON field ordering (json! macro)
//   v1.3.0         - Added report_command_status(), fixed eprintln!
//   v2.3.0         - Added memchain_status to heartbeat system_stats
//   v1.0.0-Membership - TrafficDelta, UserPermission, extended heartbeat
// ============================================

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, trace, warn};
use serde_json::json;

use aeronyx_core::crypto::IdentityKeyPair;
use super::config::ManagementConfig;
use super::models::*;

// ============================================
// MemChainHeartbeatStatus (v2.3.0)
// ============================================

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemChainHeartbeatStatus {
    pub enabled:               bool,
    pub allow_remote_storage:  bool,
    pub max_remote_owners:     usize,
    pub current_remote_owners: usize,
}

// ============================================
// v1.0.0-Membership: TrafficDelta + UserPermission
// ============================================

/// Per-wallet traffic increment for a single heartbeat period.
///
/// bytes_in  = client → server (rx from server perspective)
/// bytes_out = server → client (tx from server perspective)
///
/// Zero-traffic wallets are omitted from heartbeat payloads.
/// CMS must use F() atomic update on UserTrafficQuota.used_bytes.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TrafficDelta {
    pub bytes_in:  u64,
    pub bytes_out: u64,
}

/// Permission snapshot for a single wallet, returned per heartbeat response.
///
/// CMS returns this only for wallets listed in connected_wallets.
/// Rust enforces tier and quota constraints on active sessions using this.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct UserPermission {
    /// Subscription tier: "free" | "premium" | "ultimate"
    pub tier: String,
    /// Whether the subscription is currently active (not expired/cancelled).
    pub is_active: bool,
    /// Whether this wallet may connect to premium-tier nodes.
    pub can_access_premium_nodes: bool,
    /// false = Free tier monthly quota exceeded — disconnect the session.
    pub traffic_allowed: bool,
}

// ============================================
// HeartbeatResponse
// ============================================

#[derive(Debug, serde::Deserialize)]
pub struct HeartbeatResponse {
    pub success:           bool,
    pub next_heartbeat_in: Option<u64>,
    pub commands:          Option<Vec<Command>>,

    // v1.0.0-Membership
    /// Node access tier: "public" | "premium".
    /// None = CMS did not return a tier — keep current cached value.
    #[serde(default)]
    pub node_tier: Option<String>,

    /// Per-wallet permission snapshot for all wallets in connected_wallets.
    /// Empty map = CMS returned no permissions — keep current cached state.
    #[serde(default)]
    pub user_permissions: HashMap<String, UserPermission>,
}

// ============================================
// ManagementClient
// ============================================

pub struct ManagementClient {
    config:      ManagementConfig,
    http:        Client,
    identity:    IdentityKeyPair,
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

    /// Creates an Ed25519 signature over SHA256(node_id + timestamp + body).
    fn create_signature(&self, timestamp: u64, body: &str) -> String {
        let node_id = self.node_id();
        let message = format!("{}{}{}", node_id, timestamp, body);

        trace!(
            node_id       = %node_id,
            timestamp     = timestamp,
            body_preview  = %body.chars().take(200).collect::<String>(),
            "[SIGNATURE] Creating signature"
        );

        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        let message_hash = hasher.finalize();

        trace!(hash = %hex::encode(&message_hash), "[SIGNATURE] Message hash computed");

        let signature = self.identity.sign(&message_hash);
        let sig_hex   = hex::encode(signature);

        trace!(signature = %sig_hex, "[SIGNATURE] Signature created");
        sig_hex
    }

    fn signed_headers(node_id: &str, timestamp: u64, signature: &str)
        -> Vec<(&'static str, String)>
    {
        vec![
            ("X-Node-ID",    node_id.to_string()),
            ("X-Timestamp",  timestamp.to_string()),
            ("X-Signature",  signature.to_string()),
        ]
    }

    /// Registers the node with the CMS using a binding code.
    pub async fn register_node(&self, code: &str) -> Result<NodeInfo, String> {
        let url     = format!("{}/node/bind/", self.config.cms_url);
        let request = BindNodeRequest {
            code:          code.to_string(),
            public_key:    self.node_id(),
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
    /// ## v1.0.0-Membership additions
    /// - connected_wallets: hex list of currently connected wallet pubkeys.
    ///   CMS uses this to know which wallets to include in user_permissions.
    /// - traffic_delta: per-wallet byte increments since last heartbeat.
    ///   CMS atomically adds these to UserTrafficQuota.used_bytes (F() update).
    ///
    /// Both fields are part of the signed body — CMS can verify integrity.
    /// Empty collections are omitted from the serialized body (skip_serializing_if).
    pub async fn send_heartbeat(
        &self,
        public_ip:         &str,
        active_sessions:   u32,
        agent_status:      Option<AgentStatusInfo>,
        memchain_status:   Option<MemChainHeartbeatStatus>,
        // v1.0.0-Membership
        connected_wallets: Vec<String>,
        traffic_delta:     HashMap<String, TrafficDelta>,
    ) -> Result<HeartbeatResponse, String> {
        let url       = format!("{}/node/heartbeat/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id   = self.node_id();
        let stats     = SystemStats::collect(active_sessions);

        // Build system_stats (unchanged from v2.3.0).
        let mut system_stats_json = serde_json::json!({
            "cpu_usage":       stats.cpu_usage,
            "memory_mb":       stats.memory_mb,
            "active_sessions": stats.active_sessions,
        });

        if let Some(obj) = system_stats_json.as_object_mut() {
            if let Some(ref status) = agent_status {
                obj.insert(
                    "agent_status".to_string(),
                    serde_json::to_value(status).unwrap_or(serde_json::Value::Null),
                );
            }
            if let Some(ref mc) = memchain_status {
                obj.insert(
                    "memchain_status".to_string(),
                    serde_json::to_value(mc).unwrap_or(serde_json::Value::Null),
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

        // Build base body WITHOUT signature field (for signing).
        let mut body_for_signing = json!({
            "node_id":      &node_id,
            "timestamp":    timestamp,
            "public_ip":    public_ip,
            "version":      super::integrity::get_version(),
            "binary_hash":  &self.binary_hash,
            "system_stats": system_stats_json,
        });

        // v1.0.0-Membership: inject connected_wallets + traffic_delta
        // into the signed body so CMS can verify their integrity.
        if let Some(obj) = body_for_signing.as_object_mut() {
            if !connected_wallets.is_empty() {
                obj.insert(
                    "connected_wallets".to_string(),
                    serde_json::to_value(&connected_wallets)
                        .unwrap_or(serde_json::Value::Array(vec![])),
                );
            }
            if !traffic_delta.is_empty() {
                obj.insert(
                    "traffic_delta".to_string(),
                    serde_json::to_value(&traffic_delta)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                );
            }
        }

        // Sign the body (no signature field present yet).
        let body_str  = serde_json::to_string(&body_for_signing).map_err(|e| e.to_string())?;
        let signature = self.create_signature(timestamp, &body_str);

        // Insert signature into body.
        let body_json = {
            let mut obj = body_for_signing;
            if let Some(map) = obj.as_object_mut() {
                map.insert(
                    "signature".to_string(),
                    serde_json::Value::String(signature.clone()),
                );
            }
            obj
        };

        debug!(
            sessions  = active_sessions,
            wallets   = connected_wallets.len(),
            tx_deltas = traffic_delta.len(),
            "[HEARTBEAT] Sending"
        );

        let mut request = self.http.post(&url);
        for (header, value) in Self::signed_headers(&node_id, timestamp, &signature) {
            request = request.header(header, value);
        }

        let response = request
            .json(&body_json)
            .send().await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status    = response.status();
            let body_text = response.text().await.unwrap_or_default();
            error!("Heartbeat failed: status={}, body={}", status, body_text);
            return Err(format!("Status: {}, Body: {}", status, body_text));
        }

        response.json().await.map_err(|e| e.to_string())
    }

    /// Reports a session event (create/update/end) to the CMS.
    pub async fn report_session_event(&self, event: SessionEventReport) -> Result<(), String> {
        let url       = format!("{}/node/sessions/report/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id   = self.node_id();
        let body_str  = serde_json::to_string(&event).map_err(|e| e.to_string())?;
        let signature = self.create_signature(timestamp, &body_str);

        let mut request = self.http.post(&url);
        for (header, value) in Self::signed_headers(&node_id, timestamp, &signature) {
            request = request.header(header, value);
        }

        let response = request
            .json(&event)
            .send().await
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            let status    = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(format!("Status: {}, Body: {}", status, body_text));
        }

        Ok(())
    }

    /// Reports command execution status to CMS (v1.3.0).
    pub async fn report_command_status(&self, report: &CommandStatusReport) -> Result<(), String> {
        let url       = format!("{}/node/agent/status/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id   = self.node_id();
        let body_str  = serde_json::to_string(report).map_err(|e| e.to_string())?;
        let signature = self.create_signature(timestamp, &body_str);

        debug!(
            command_id = %report.command_id,
            status     = ?report.status,
            "[CMS_CLIENT] Reporting command status"
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
            let status    = response.status();
            let body_text = response.text().await.unwrap_or_default();
            warn!(
                command_id  = %report.command_id,
                http_status = %status,
                "[CMS_CLIENT] Command status report failed: {}",
                body_text
            );
            return Err(format!("Status: {}, Body: {}", status, body_text));
        }

        debug!(command_id = %report.command_id, "[CMS_CLIENT] Command status reported");
        Ok(())
    }

    pub fn config(&self) -> &ManagementConfig { &self.config }
}

impl std::fmt::Debug for ManagementClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagementClient")
            .field("cms_url", &self.config.cms_url)
            .finish()
    }
}
