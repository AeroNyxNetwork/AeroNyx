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
//   - record_commitment_sync is aggregate runtime health only. Never add
//     coordinator identity, endpoint, block hashes, owners, or payload data.
//   - record_commitment_integrity reports aggregate verification evidence only.
//     Never add hashes, proposer identity, commitment IDs, or user metadata.
//   - checkpoint observation freshness derives from applicable durable signed
//     proof time; deferred evidence above a recovered follower tip, attempts,
//     and inbound served responses must never refresh it.
//   - witness round counts are aggregate operational evidence only. They are
//     not votes, quorum, finality, or a fork-choice input.
//   - pinned witness reporting exposes only scope, count, and strict-policy
//     state/threshold. Never report pinned node identities or resolved endpoints.
//   - commitment durability reports only SQLite's aggregate synchronous mode;
//     never add database paths, host details, block hashes, or user data.
//   - rollback guard reporting is aggregate only. Never send the sidecar path,
//     signer, signature, identity material, or anchored block hash.
//   - The body used for signing MUST be the same as what is sent.
//     Do NOT add fields after signing.
//
// Last Modified:
//   v1.0.0         - Initial implementation
//   v1.2.0         - Fixed JSON field ordering (json! macro)
//   v1.3.0         - Added report_command_status(), fixed eprintln!
//   v2.4.0         - Added chat_relay_status to heartbeat system_stats
//   v2.3.0         - Added memchain_status to heartbeat system_stats
//   v2.7.1         - Added compact privacy-safe commitment sync evidence
//   v2.7.4         - Added compact verified commitment-chain integrity evidence
//   v2.7.11        - Added durable checkpoint observation freshness evidence
//   v2.7.12        - Added privacy-safe witness round coverage evidence
//   v2.7.13        - Added aggregate commitment durability evidence
//   v2.7.14        - Added aggregate signed tip rollback-guard evidence
//   v2.7.15        - Added privacy-safe pinned witness policy evidence
//   v2.7.16        - Added the aggregate strict witness startup threshold
//   v2.7.19        - Added applicable/deferred checkpoint evidence counts
//   v2.7.20        - Added aggregate durable witness-equivocation incidents
//   v1.0.0-Membership - TrafficDelta, UserPermission, extended heartbeat
// ============================================

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde_json::json;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, trace, warn};

use super::config::ManagementConfig;
use super::models::*;
use aeronyx_core::crypto::IdentityKeyPair;

static RUNTIME_STARTED_AT: OnceLock<u64> = OnceLock::new();
static RUNTIME_ID: OnceLock<String> = OnceLock::new();

// ============================================
// MemChainHeartbeatStatus (v2.3.0)
// ============================================

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemChainHeartbeatStatus {
    pub enabled: bool,
    pub allow_remote_storage: bool,
    pub max_remote_owners: usize,
    pub current_remote_owners: usize,
    /// Compact privacy-safe persisted-chain verification evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_commitment_integrity: Option<RecordCommitmentIntegrityHeartbeatStatus>,
    /// Compact privacy-safe Block Sync runtime evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_commitment_sync: Option<RecordCommitmentSyncHeartbeatStatus>,
    /// Compact privacy-safe signed checkpoint reconciliation evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_commitment_checkpoint: Option<RecordCommitmentCheckpointHeartbeatStatus>,
}

/// Compact verified-chain evidence sent to the central health plane.
///
/// This intentionally excludes block hashes, proposer identities, commitment
/// IDs, owners, payloads, peers, endpoints, routes, and client metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecordCommitmentIntegrityHeartbeatStatus {
    /// Stable heartbeat contract name.
    pub contract_version: &'static str,
    /// `verified` or `not_verified`.
    pub state: String,
    /// Time the complete persisted-chain baseline was established.
    pub baseline_verified_at: Option<u64>,
    /// Time of the most recent full audit or verified append.
    pub last_verified_at: Option<u64>,
    /// Wall-clock duration of the complete baseline audit.
    pub verification_duration_ms: Option<u64>,
    /// Blocks covered by the current verified baseline.
    pub verified_block_count: u64,
    /// Opaque commitments covered by the current verified baseline.
    pub verified_commitment_count: u64,
    /// Last verified one-based height, or zero for an empty chain.
    pub verified_tip_height: u64,
    /// Effective SQLite mode: `off`, `normal`, `full`, or `extra`.
    pub durability_mode: &'static str,
    /// Signed local high-water guard state.
    pub rollback_guard_state: &'static str,
    /// Highest height covered by the local guard.
    pub rollback_guard_height: u64,
    /// Last signature/ancestry verification time.
    pub rollback_guard_last_verified_at: Option<u64>,
    /// Last successful atomic sidecar write time.
    pub rollback_guard_last_persisted_at: Option<u64>,
    /// Process-lifetime atomic sidecar write failures.
    pub rollback_guard_write_failures_total: u64,
}

/// Compact commitment follower status sent to the central health plane.
///
/// This intentionally excludes recent event history, coordinator identity,
/// endpoint, block hashes, record commitments, owners, and payload metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecordCommitmentSyncHeartbeatStatus {
    /// Stable heartbeat contract name.
    pub contract_version: &'static str,
    /// `coordinator`, `follower`, or `verifier`.
    pub role: String,
    /// Current runtime state.
    pub state: String,
    /// Whether active follower polling is configured.
    pub enabled: bool,
    /// Most recent follower pull attempt.
    pub last_attempt_at: Option<u64>,
    /// Most recent successful verified response page.
    pub last_success_at: Option<u64>,
    /// Most recent fail-closed pull event.
    pub last_failure_at: Option<u64>,
    /// Most recent recovery after a failure streak.
    pub last_recovered_at: Option<u64>,
    /// Next scheduled poll or backoff retry.
    pub next_poll_at: Option<u64>,
    /// Current consecutive failure count.
    pub consecutive_failures: u32,
    /// Stable allow-listed last error code.
    pub last_error_code: Option<String>,
    /// Last verified remote tip height.
    pub remote_tip_height: Option<u64>,
    /// Verified pages received since process start.
    pub pages_received_total: u64,
    /// Verified blocks received since process start.
    pub blocks_received_total: u64,
    /// Failure events observed since process start.
    pub failure_events_total: u64,
    /// Recovery events observed since process start.
    pub recovery_events_total: u64,
}

/// Compact signed-checkpoint evidence sent to the central health plane.
///
/// This deliberately excludes the evidence digest, peer identities, chain or
/// block hashes, signatures, request IDs, endpoints, and user metadata. The
/// central plane can monitor convergence but cannot reconstruct the ledger.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecordCommitmentCheckpointHeartbeatStatus {
    /// Stable heartbeat contract name.
    pub contract_version: &'static str,
    /// `operator_pinned` or `permissionless_evidence`; never a peer identity.
    pub witness_scope: &'static str,
    /// Number of configured pinned identities without revealing those pins.
    pub pinned_witnesses_configured: usize,
    /// Whether startup enforces the configured signed-evidence threshold.
    pub startup_evidence_required: bool,
    /// Minimum distinct pinned responses required when strict startup is enabled.
    pub startup_minimum_verified: usize,
    /// Latest aggregate relation or proof lifecycle state.
    pub state: String,
    /// Most recent outbound checkpoint verification attempt.
    pub last_checked_at: Option<u64>,
    /// Most recent verified equal-tip comparison.
    pub last_converged_at: Option<u64>,
    /// Most recent verified shared-prefix mismatch.
    pub last_divergence_at: Option<u64>,
    /// Most recent failed proof attempt.
    pub last_failure_at: Option<u64>,
    /// Most recent authenticated checkpoint response served.
    pub last_served_at: Option<u64>,
    /// Local verified tip height from the latest observation.
    pub local_tip_height: Option<u64>,
    /// Remote verified tip height from the latest observation.
    pub remote_tip_height: Option<u64>,
    /// Signature-verified checkpoint responses since process start.
    pub proofs_verified_total: u64,
    /// Proof attempts rejected or unavailable since process start.
    pub proofs_failed_total: u64,
    /// Verified shared-prefix mismatches since process start.
    pub divergences_total: u64,
    /// Authenticated checkpoint requests served since process start.
    pub requests_served_total: u64,
    /// Cryptographic frame audit: `not_audited`, `verified`, or `invalid`.
    /// Applicability to the current chain is reported by the counts below.
    pub evidence_state: String,
    /// Current bounded durable proof count; raw frames never leave the node.
    pub evidence_records: u64,
    /// Proofs applicable to the currently audited local chain.
    pub applicable_evidence_records: u64,
    /// Historical proofs above the current audited local tip.
    pub deferred_evidence_records: u64,
    /// Applicable durable proof count classified as divergence.
    pub divergence_evidence_records: u64,
    /// Durable trusted-witness same-height conflicting-hash incidents.
    pub equivocation_incidents: u64,
    /// Most recent applicable durable evidence observation time.
    pub last_evidence_at: Option<u64>,
    /// `unavailable`, `fresh`, or `stale` for durable signed observations.
    pub observation_freshness: String,
    /// Age of the latest durable observation; absent when unavailable.
    pub observation_age_seconds: Option<u64>,
    /// Maximum age still classified as fresh.
    pub freshness_window_seconds: u64,
    /// Latest bounded witness-round classification.
    pub last_round_state: String,
    /// Completion time of the latest bounded witness round.
    pub last_round_at: Option<u64>,
    /// Eligible witnesses before the per-round cap.
    pub last_round_eligible: usize,
    /// Witnesses contacted after the per-round cap.
    pub last_round_attempted: usize,
    /// Responses that established durable signed evidence.
    pub last_round_verified: usize,
    /// Attempts that established no durable signed evidence.
    pub last_round_failed: usize,
    /// Same-tip signed observations in the latest round.
    pub last_round_converged: usize,
    /// Signed remote-ahead observations in the latest round.
    pub last_round_remote_ahead: usize,
    /// Signed remote-behind observations in the latest round.
    pub last_round_remote_behind: usize,
    /// Signed shared-prefix mismatches in the latest round.
    pub last_round_diverged: usize,
    /// Local persistence failures since process start.
    pub evidence_persistence_failures_total: u64,
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
    pub bytes_in: u64,
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

#[derive(Debug, Clone, serde::Deserialize)]
pub struct NodePolicy {
    #[serde(default)]
    pub node_tier: String,
    #[serde(default)]
    pub maintenance_mode: bool,
    #[serde(default)]
    pub max_sessions: u32,
    #[serde(default)]
    pub bandwidth_limit_mbps: u32,
    #[serde(default)]
    pub heartbeat_interval_seconds: u64,
    #[serde(default)]
    pub updated_at: Option<String>,
}

// ============================================
// HeartbeatResponse
// ============================================

#[derive(Debug, serde::Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub next_heartbeat_in: Option<u64>,
    pub commands: Option<Vec<Command>>,

    // v1.0.0-Membership
    /// Node access tier: "public" | "premium".
    /// None = CMS did not return a tier — keep current cached value.
    #[serde(default)]
    pub node_tier: Option<String>,

    /// Per-wallet permission snapshot for all wallets in connected_wallets.
    /// Empty map = CMS returned no permissions — keep current cached state.
    #[serde(default)]
    pub user_permissions: HashMap<String, UserPermission>,

    /// Full operator-managed wallet deny policy for this node.
    /// None = CMS did not include the field — keep current runtime state.
    /// Some(empty) = CMS explicitly has no active operator bans.
    #[serde(default)]
    pub operator_bans: Option<Vec<String>>,

    /// Operator VPN policy from nodeboard Settings.
    /// Older CMS versions omit this field; Rust must keep local defaults then.
    #[serde(default)]
    pub node_policy: Option<NodePolicy>,
}

// ============================================
// ManagementClient
// ============================================

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
        Self {
            config,
            http,
            identity,
            binary_hash,
        }
    }

    pub fn node_id(&self) -> String {
        hex::encode(self.identity.public_key_bytes())
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn runtime_started_at() -> u64 {
        *RUNTIME_STARTED_AT.get_or_init(Self::current_timestamp)
    }

    fn runtime_id(&self, node_id: &str) -> String {
        RUNTIME_ID
            .get_or_init(|| {
                let mut hasher = Sha256::new();
                hasher.update(node_id.as_bytes());
                hasher.update(self.binary_hash.as_bytes());
                hasher.update(Self::runtime_started_at().to_string().as_bytes());
                hasher.update(std::process::id().to_string().as_bytes());
                hex::encode(hasher.finalize())
            })
            .clone()
    }

    fn build_git_commit() -> &'static str {
        option_env!("AERONYX_GIT_COMMIT")
            .or(option_env!("GIT_COMMIT"))
            .or(option_env!("VERGEN_GIT_SHA"))
            .unwrap_or("unknown")
    }

    fn build_profile() -> &'static str {
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
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
        let sig_hex = hex::encode(signature);

        trace!(signature = %sig_hex, "[SIGNATURE] Signature created");
        sig_hex
    }

    fn signed_headers(
        node_id: &str,
        timestamp: u64,
        signature: &str,
    ) -> Vec<(&'static str, String)> {
        vec![
            ("X-Node-ID", node_id.to_string()),
            ("X-Timestamp", timestamp.to_string()),
            ("X-Signature", signature.to_string()),
        ]
    }

    /// Registers the node with the CMS using a binding code.
    pub async fn register_node(&self, code: &str) -> Result<NodeInfo, String> {
        let url = format!("{}/node/bind/", self.config.cms_url);
        let request = BindNodeRequest {
            code: code.to_string(),
            public_key: self.node_id(),
            hardware_info: HardwareInfo::collect(),
        };

        info!("Registering node...");

        let response = self
            .http
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        let body: BindNodeResponse = response
            .json()
            .await
            .map_err(|e| format!("Parse failed: {}", e))?;

        if body.success {
            if let Some(node) = body.node {
                info!("Node registered!");
                return Ok(node);
            }
        }

        Err(body
            .error
            .or(body.message)
            .unwrap_or_else(|| "Unknown error".to_string()))
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
        public_ip: &str,
        active_sessions: u32,
        memchain_status: Option<MemChainHeartbeatStatus>,
        // v1.0.0-Membership
        connected_wallets: Vec<String>,
        traffic_delta: HashMap<String, TrafficDelta>,
        vpn_health: Option<serde_json::Value>,
        operator_status: Option<serde_json::Value>,
        discovery_status: Option<serde_json::Value>,
        chat_relay_status: Option<serde_json::Value>,
    ) -> Result<HeartbeatResponse, String> {
        let url = format!("{}/node/heartbeat/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();
        let stats = SystemStats::collect(active_sessions);
        let runtime_id = self.runtime_id(&node_id);
        let runtime_started_at = Self::runtime_started_at();
        let now = Self::current_timestamp();

        // Build system_stats (unchanged from v2.3.0).
        let mut system_stats_json = serde_json::json!({
            "cpu_usage":       stats.cpu_usage,
            "memory_mb":       stats.memory_mb,
            "active_sessions": stats.active_sessions,
            "runtime_id":      runtime_id,
            "runtime_started_at": runtime_started_at,
            "runtime_uptime_seconds": now.saturating_sub(runtime_started_at),
            "runtime_version": super::integrity::get_version(),
            "runtime_git_commit": Self::build_git_commit(),
            "runtime_build_profile": Self::build_profile(),
            "runtime_build_target": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            "runtime_process_id": std::process::id(),
        });

        if let Some(obj) = system_stats_json.as_object_mut() {
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
            if let Some(health) = vpn_health {
                obj.insert("vpn_health".to_string(), health);
            }
            if let Some(status) = operator_status {
                obj.insert("operator_status".to_string(), status);
            }
            if let Some(status) = discovery_status {
                obj.insert("discovery_status".to_string(), status);
            }
            if let Some(status) = chat_relay_status {
                obj.insert("chat_relay_status".to_string(), status);
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
        let body_str = serde_json::to_string(&body_for_signing).map_err(|e| e.to_string())?;
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
            sessions = active_sessions,
            wallets = connected_wallets.len(),
            tx_deltas = traffic_delta.len(),
            "[HEARTBEAT] Sending"
        );

        let mut request = self.http.post(&url);
        for (header, value) in Self::signed_headers(&node_id, timestamp, &signature) {
            request = request.header(header, value);
        }

        let response = request
            .json(&body_json)
            .send()
            .await
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
    pub async fn report_session_event(&self, event: SessionEventReport) -> Result<(), String> {
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
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(format!("Status: {}, Body: {}", status, body_text));
        }

        Ok(())
    }

    /// Reports command execution status to CMS (v1.3.0).
    pub async fn report_command_status(&self, report: &CommandStatusReport) -> Result<(), String> {
        let url = format!("{}/node/vpn/status/", self.config.cms_url);
        let timestamp = Self::current_timestamp();
        let node_id = self.node_id();
        let body_str = serde_json::to_string(report).map_err(|e| e.to_string())?;
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
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_heartbeat_contracts_exclude_sensitive_fields() {
        let status = MemChainHeartbeatStatus {
            enabled: true,
            allow_remote_storage: false,
            max_remote_owners: 100,
            current_remote_owners: 0,
            record_commitment_integrity: Some(RecordCommitmentIntegrityHeartbeatStatus {
                contract_version: "record_commitment_integrity.v1",
                state: "verified".to_string(),
                baseline_verified_at: Some(90),
                last_verified_at: Some(100),
                verification_duration_ms: Some(7),
                verified_block_count: 9,
                verified_commitment_count: 27,
                verified_tip_height: 9,
                durability_mode: "full",
                rollback_guard_state: "verified",
                rollback_guard_height: 9,
                rollback_guard_last_verified_at: Some(100),
                rollback_guard_last_persisted_at: Some(100),
                rollback_guard_write_failures_total: 0,
            }),
            record_commitment_sync: Some(RecordCommitmentSyncHeartbeatStatus {
                contract_version: "record_commitment_sync.v1",
                role: "follower".to_string(),
                state: "backoff".to_string(),
                enabled: true,
                last_attempt_at: Some(99),
                last_success_at: Some(100),
                last_failure_at: Some(110),
                last_recovered_at: None,
                next_poll_at: Some(140),
                consecutive_failures: 1,
                last_error_code: Some("request_timeout".to_string()),
                remote_tip_height: Some(9),
                pages_received_total: 3,
                blocks_received_total: 27,
                failure_events_total: 1,
                recovery_events_total: 0,
            }),
            record_commitment_checkpoint: Some(RecordCommitmentCheckpointHeartbeatStatus {
                contract_version: "record_commitment_checkpoint.v1",
                witness_scope: "operator_pinned",
                pinned_witnesses_configured: 3,
                startup_evidence_required: true,
                startup_minimum_verified: 2,
                state: "converged".to_string(),
                last_checked_at: Some(120),
                last_converged_at: Some(120),
                last_divergence_at: None,
                last_failure_at: None,
                last_served_at: Some(118),
                local_tip_height: Some(9),
                remote_tip_height: Some(9),
                proofs_verified_total: 4,
                proofs_failed_total: 1,
                divergences_total: 0,
                requests_served_total: 2,
                evidence_state: "verified".to_string(),
                evidence_records: 3,
                applicable_evidence_records: 2,
                deferred_evidence_records: 1,
                divergence_evidence_records: 0,
                equivocation_incidents: 0,
                last_evidence_at: Some(120),
                observation_freshness: "fresh".to_string(),
                observation_age_seconds: Some(5),
                freshness_window_seconds: 900,
                last_round_state: "partial".to_string(),
                last_round_at: Some(120),
                last_round_eligible: 4,
                last_round_attempted: 3,
                last_round_verified: 2,
                last_round_failed: 1,
                last_round_converged: 1,
                last_round_remote_ahead: 0,
                last_round_remote_behind: 1,
                last_round_diverged: 0,
                evidence_persistence_failures_total: 0,
            }),
        };
        let value = serde_json::to_value(status).unwrap();
        let integrity = &value["record_commitment_integrity"];
        assert_eq!(integrity["state"], "verified");
        assert_eq!(integrity["verified_block_count"], 9);
        assert_eq!(integrity["verified_commitment_count"], 27);
        assert_eq!(integrity["verified_tip_height"], 9);
        assert_eq!(integrity["durability_mode"], "full");
        assert_eq!(integrity["rollback_guard_state"], "verified");
        assert_eq!(integrity["rollback_guard_height"], 9);
        assert_eq!(integrity["rollback_guard_write_failures_total"], 0);
        let sync = &value["record_commitment_sync"];
        assert_eq!(sync["state"], "backoff");
        assert_eq!(sync["enabled"], true);
        assert_eq!(sync["last_attempt_at"], 99);
        assert_eq!(sync["last_error_code"], "request_timeout");
        let checkpoint = &value["record_commitment_checkpoint"];
        assert_eq!(checkpoint["state"], "converged");
        assert_eq!(checkpoint["witness_scope"], "operator_pinned");
        assert_eq!(checkpoint["pinned_witnesses_configured"], 3);
        assert_eq!(checkpoint["startup_evidence_required"], true);
        assert_eq!(checkpoint["startup_minimum_verified"], 2);
        assert_eq!(checkpoint["proofs_verified_total"], 4);
        assert_eq!(checkpoint["requests_served_total"], 2);
        assert_eq!(checkpoint["evidence_state"], "verified");
        assert_eq!(checkpoint["evidence_records"], 3);
        assert_eq!(checkpoint["applicable_evidence_records"], 2);
        assert_eq!(checkpoint["deferred_evidence_records"], 1);
        assert_eq!(checkpoint["equivocation_incidents"], 0);
        assert_eq!(checkpoint["observation_freshness"], "fresh");
        assert_eq!(checkpoint["observation_age_seconds"], 5);
        assert_eq!(checkpoint["freshness_window_seconds"], 900);
        assert_eq!(checkpoint["last_round_state"], "partial");
        assert_eq!(checkpoint["last_round_eligible"], 4);
        assert_eq!(checkpoint["last_round_attempted"], 3);
        assert_eq!(checkpoint["last_round_verified"], 2);
        assert_eq!(checkpoint["last_round_failed"], 1);
        for section in [integrity, sync, checkpoint] {
            for forbidden in [
                "coordinator_node_id",
                "endpoint",
                "tip_hash",
                "block_hash",
                "proposer",
                "commitment_ids",
                "record_ids",
                "recent_events",
                "owner",
                "payload",
                "signed_response",
                "evidence_digest",
                "signature",
                "signer",
                "anchor_path",
                "rollback_guard_path",
                "request_id",
                "peer",
                "route",
                "client",
                "evidence_digest",
                "request_id",
                "signature",
            ] {
                assert!(
                    section.get(forbidden).is_none(),
                    "unexpected field: {forbidden}"
                );
            }
        }
    }
}
