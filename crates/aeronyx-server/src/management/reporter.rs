// ============================================
// File: crates/aeronyx-server/src/management/reporter.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   HeartbeatReporter now collects connected_wallets + traffic_delta
//   before each heartbeat and handles node_tier / user_permissions
//   from the response to enforce membership rules.
//
// Main Functionality:
//   - HeartbeatReporter: periodic heartbeat with membership enforcement
//   - SessionReporter: session lifecycle event reporting
//   - SessionEventSender: thread-safe event sender
//
// ⚠️ Important Notes for Next Developer:
//   - On heartbeat timeout or error: preserve last permissions, disconnect
//     nobody. Fail-open is intentional to avoid mass disconnects on
//     transient CMS outages.
//   - handle_membership_response() sends 0xFF RESET then sessions.remove().
//     remove() handles routing + wallet_index + cooldown cleanup.
//   - user_permissions is only updated when response contains a non-empty
//     map. Empty map = keep last known state.
//   - node_tier defaults to "public" until CMS says otherwise.
//   - SessionTrafficSnapshot bytes are CUMULATIVE totals, never deltas.
//     Backend must upsert, not accumulate. is_final=false for snapshots.
//
// Last Modified:
//   v1.0.0                - Initial implementation
//   v1.2.0                - Fixed SessionEventType import
//   v1.3.0                - Command forwarding, dynamic interval
//   v2.3.0                - memchain_status reporting
//   v1.0.0-TrafficAccounting - SessionTrafficSnapshot event
//   v1.0.0-Membership     - wallet collection, delta drain, permission enforcement
// ============================================

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::client::{ManagementClient, MemChainHeartbeatStatus, TrafficDelta, UserPermission};
use super::models::{Command, SessionEventReport, SessionEventType};
use crate::services::deny_list::{DenyList, DenyReason};
use crate::services::traffic_tracker::TrafficTracker;
use crate::services::{NodePolicyRuntime, NodePolicySnapshot, SessionManager};
use aeronyx_transport::traits::Transport;

// ============================================
// Constants
// ============================================

const MIN_HEARTBEAT_INTERVAL_SECS: u64 = 10;
const MAX_HEARTBEAT_INTERVAL_SECS: u64 = 300;
const COLD_START_GRACE_BEATS: u32 = 5;
const COLD_START_TIMEOUT_SECS: u64 = 30;

// ============================================
// MemChainStatusFn
// ============================================

pub type MemChainStatusFn = Box<dyn Fn() -> Option<MemChainHeartbeatStatus> + Send + Sync>;

/// Async VPN health probe injected by server startup.
///
/// Source path:
///   /root/open/AeroNyx/crates/aeronyx-server/src/management/reporter.rs
///
/// The probe returns privacy-safe node diagnostics for heartbeat
/// `system_stats.vpn_health`.
pub type VpnHealthStatusFn =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = Option<serde_json::Value>> + Send>> + Send + Sync>;

/// Async node-operator service snapshot injected by server startup.
///
/// The probe returns privacy-safe aggregate service readiness for heartbeat
/// `system_stats.operator_status`.
pub type OperatorStatusFn =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = Option<serde_json::Value>> + Send>> + Send + Sync>;

/// Async peer-discovery snapshot injected by server startup.
///
/// The probe returns aggregate node discovery counters for heartbeat
/// `system_stats.discovery_status`. It must not include client IPs,
/// destinations, DNS contents, packet payloads, chat plaintext, voucher
/// secrets, private keys, or wallet-level traffic.
pub type DiscoveryStatusFn =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = Option<serde_json::Value>> + Send>> + Send + Sync>;

/// Async encrypted chat relay snapshot injected by server startup.
///
/// The probe returns aggregate node-to-node relay counters for heartbeat
/// `system_stats.chat_relay_status`. It must not include message IDs, wallet
/// IDs, client IPs, destinations, DNS contents, packet payloads, chat
/// plaintext, ciphertext, private keys, voucher secrets, or per-user traffic.
pub type ChatRelayStatusFn =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = Option<serde_json::Value>> + Send>> + Send + Sync>;

// ============================================
// Session Events
// ============================================

#[derive(Debug, Clone)]
pub struct SessionEvent {
    pub event_type: SessionEventType,
    pub session_id: String,
    pub client_wallet: Option<String>,
    pub virtual_ip: Option<String>,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub timestamp: u64,
    pub last_rx_at: Option<u64>,
    pub last_tx_at: Option<u64>,
    pub rtt_ms: Option<f64>,
    pub packet_loss: Option<f64>,
    pub replay_rejections: Option<u64>,
    pub too_old_rejections: Option<u64>,
    pub packets_rx: Option<u64>,
    pub packets_tx: Option<u64>,
    pub keepalive_probes_sent: Option<u64>,
    pub keepalive_acks: Option<u64>,
    pub keepalive_missed: Option<u64>,
    pub keepalive_pending: Option<u64>,
}

impl SessionEvent {
    pub fn created(
        session_id: String,
        client_wallet: Option<String>,
        virtual_ip: Option<String>,
    ) -> Self {
        Self {
            event_type: SessionEventType::SessionCreated,
            session_id,
            client_wallet,
            virtual_ip,
            bytes_in: 0,
            bytes_out: 0,
            timestamp: now_unix(),
            last_rx_at: None,
            last_tx_at: None,
            rtt_ms: None,
            packet_loss: None,
            replay_rejections: None,
            too_old_rejections: None,
            packets_rx: None,
            packets_tx: None,
            keepalive_probes_sent: None,
            keepalive_acks: None,
            keepalive_missed: None,
            keepalive_pending: None,
        }
    }

    pub fn ended(
        session_id: String,
        client_wallet: Option<String>,
        virtual_ip: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
        quality: SessionQuality,
    ) -> Self {
        Self {
            event_type: SessionEventType::SessionEnded,
            session_id,
            client_wallet,
            virtual_ip,
            bytes_in,
            bytes_out,
            timestamp: now_unix(),
            last_rx_at: quality.last_rx_at,
            last_tx_at: quality.last_tx_at,
            rtt_ms: quality.rtt_ms,
            packet_loss: quality.packet_loss,
            replay_rejections: quality.replay_rejections,
            too_old_rejections: quality.too_old_rejections,
            packets_rx: quality.packets_rx,
            packets_tx: quality.packets_tx,
            keepalive_probes_sent: quality.keepalive_probes_sent,
            keepalive_acks: quality.keepalive_acks,
            keepalive_missed: quality.keepalive_missed,
            keepalive_pending: quality.keepalive_pending,
        }
    }

    pub fn snapshot(
        session_id: String,
        client_wallet: Option<String>,
        virtual_ip: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
        quality: SessionQuality,
    ) -> Self {
        Self {
            event_type: SessionEventType::SessionTrafficSnapshot,
            session_id,
            client_wallet,
            virtual_ip,
            bytes_in,
            bytes_out,
            timestamp: now_unix(),
            last_rx_at: quality.last_rx_at,
            last_tx_at: quality.last_tx_at,
            rtt_ms: quality.rtt_ms,
            packet_loss: quality.packet_loss,
            replay_rejections: quality.replay_rejections,
            too_old_rejections: quality.too_old_rejections,
            packets_rx: quality.packets_rx,
            packets_tx: quality.packets_tx,
            keepalive_probes_sent: quality.keepalive_probes_sent,
            keepalive_acks: quality.keepalive_acks,
            keepalive_missed: quality.keepalive_missed,
            keepalive_pending: quality.keepalive_pending,
        }
    }

    fn to_report(&self) -> SessionEventReport {
        SessionEventReport {
            event_type: self.event_type,
            session_id: self.session_id.clone(),
            client_wallet: self.client_wallet.clone(),
            client_ip: None,
            virtual_ip: self.virtual_ip.clone(),
            bytes_in: self.bytes_in,
            bytes_out: self.bytes_out,
            timestamp: self.timestamp,
            is_final: matches!(self.event_type, SessionEventType::SessionEnded),
            last_rx_at: self.last_rx_at,
            last_tx_at: self.last_tx_at,
            rtt_ms: self.rtt_ms,
            packet_loss: self.packet_loss,
            replay_rejections: self.replay_rejections,
            too_old_rejections: self.too_old_rejections,
            packets_rx: self.packets_rx,
            packets_tx: self.packets_tx,
            keepalive_probes_sent: self.keepalive_probes_sent,
            keepalive_acks: self.keepalive_acks,
            keepalive_missed: self.keepalive_missed,
            keepalive_pending: self.keepalive_pending,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SessionQuality {
    pub last_rx_at: Option<u64>,
    pub last_tx_at: Option<u64>,
    pub rtt_ms: Option<f64>,
    pub packet_loss: Option<f64>,
    pub replay_rejections: Option<u64>,
    pub too_old_rejections: Option<u64>,
    pub packets_rx: Option<u64>,
    pub packets_tx: Option<u64>,
    pub keepalive_probes_sent: Option<u64>,
    pub keepalive_acks: Option<u64>,
    pub keepalive_missed: Option<u64>,
    pub keepalive_pending: Option<u64>,
}

#[inline]
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================
// HeartbeatReporter
// ============================================

pub struct HeartbeatReporter {
    client: Arc<ManagementClient>,
    interval: Duration,
    public_ip: String,
    command_tx: Option<mpsc::Sender<Command>>,
    memchain_status_fn: Option<MemChainStatusFn>,
    vpn_health_status_fn: Option<VpnHealthStatusFn>,
    operator_status_fn: Option<OperatorStatusFn>,
    discovery_status_fn: Option<DiscoveryStatusFn>,
    chat_relay_status_fn: Option<ChatRelayStatusFn>,

    // v1.0.0-Membership
    sessions: Option<Arc<SessionManager>>,
    traffic: Option<Arc<TrafficTracker>>,
    udp: Option<Arc<aeronyx_transport::UdpTransport>>,
    /// v1.0.0-Membership: deny list for writing disconnection decisions.
    deny_list: Option<Arc<DenyList>>,
    /// Runtime operator policy shared with handshake and packet handlers.
    node_policy: Option<Arc<NodePolicyRuntime>>,
    /// Cached node tier from last CMS response. Default = "public".
    node_tier: Arc<RwLock<String>>,
    /// Cached per-wallet permissions from last CMS response.
    user_permissions: Arc<RwLock<HashMap<String, UserPermission>>>,
}

impl HeartbeatReporter {
    pub fn new(client: Arc<ManagementClient>, public_ip: String) -> Self {
        let interval = Duration::from_secs(client.config().heartbeat_interval_secs);
        Self {
            client,
            interval,
            public_ip,
            command_tx: None,
            memchain_status_fn: None,
            vpn_health_status_fn: None,
            operator_status_fn: None,
            discovery_status_fn: None,
            chat_relay_status_fn: None,
            sessions: None,
            traffic: None,
            udp: None,
            deny_list: None,
            node_policy: None,
            node_tier: Arc::new(RwLock::new("public".to_string())),
            user_permissions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_command_sender(mut self, tx: mpsc::Sender<Command>) -> Self {
        self.command_tx = Some(tx);
        self
    }

    pub fn with_memchain_status(mut self, f: MemChainStatusFn) -> Self {
        self.memchain_status_fn = Some(f);
        self
    }

    pub fn with_vpn_health_status(mut self, f: VpnHealthStatusFn) -> Self {
        self.vpn_health_status_fn = Some(f);
        self
    }

    pub fn with_operator_status(mut self, f: OperatorStatusFn) -> Self {
        self.operator_status_fn = Some(f);
        self
    }

    pub fn with_discovery_status(mut self, f: DiscoveryStatusFn) -> Self {
        self.discovery_status_fn = Some(f);
        self
    }

    pub fn with_chat_relay_status(mut self, f: ChatRelayStatusFn) -> Self {
        self.chat_relay_status_fn = Some(f);
        self
    }

    pub fn with_sessions(mut self, sessions: Arc<SessionManager>) -> Self {
        self.sessions = Some(sessions);
        self
    }

    pub fn with_traffic_tracker(mut self, traffic: Arc<TrafficTracker>) -> Self {
        self.traffic = Some(traffic);
        self
    }

    pub fn with_udp(mut self, udp: Arc<aeronyx_transport::UdpTransport>) -> Self {
        self.udp = Some(udp);
        self
    }

    pub fn with_deny_list(mut self, deny_list: Arc<DenyList>) -> Self {
        self.deny_list = Some(deny_list);
        self
    }

    pub fn with_node_policy(mut self, node_policy: Arc<NodePolicyRuntime>) -> Self {
        self.node_policy = Some(node_policy);
        self
    }

    /// Runs the heartbeat loop until shutdown signal received.
    pub async fn run<F>(
        self,
        session_count_fn: F,
        mut shutdown: tokio::sync::broadcast::Receiver<()>,
    ) where
        F: Fn() -> u32 + Send + 'static,
    {
        info!(
            interval_secs = self.interval.as_secs(),
            has_command_channel = self.command_tx.is_some(),
            has_memchain_status = self.memchain_status_fn.is_some(),
            has_membership = self.sessions.is_some(),
            "[HEARTBEAT] Reporter started"
        );

        let mut active_interval_secs = self.interval.as_secs();
        let mut interval = tokio::time::interval(Duration::from_secs(active_interval_secs));
        let mut failures = 0u32;
        let mut total_beats = 0u32;

        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("[HEARTBEAT] Stopping");
                    break;
                }
                _ = interval.tick() => {
                    total_beats += 1;
                    let in_cold_start = total_beats <= COLD_START_GRACE_BEATS;

                    let memchain_status = self.memchain_status_fn.as_ref()
                        .and_then(|f| f());

                    let vpn_health_status = if let Some(ref f) = self.vpn_health_status_fn {
                        f().await
                    } else {
                        None
                    };

                    let operator_status = if let Some(ref f) = self.operator_status_fn {
                        f().await
                    } else {
                        None
                    };

                    let discovery_status = if let Some(ref f) = self.discovery_status_fn {
                        f().await
                    } else {
                        None
                    };

                    let chat_relay_status = if let Some(ref f) = self.chat_relay_status_fn {
                        f().await
                    } else {
                        None
                    };

                    // v1.0.0-Membership: collect connected wallets (deduplicated).
                    let connected_wallets: Vec<String> =
                        if let Some(ref sm) = self.sessions {
                            let mut seen = std::collections::HashSet::new();
                            sm.all_sessions()
                                .into_iter()
                                .filter_map(|s| {
                                    if seen.insert(s.wallet_hex.clone()) {
                                        Some(s.wallet_hex.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };

                    // v1.0.0-Membership: drain per-wallet traffic deltas.
                    let traffic_delta: HashMap<String, TrafficDelta> =
                        if let Some(ref t) = self.traffic {
                            t.drain()
                        } else {
                            HashMap::new()
                        };

                    let timeout_secs = if in_cold_start { COLD_START_TIMEOUT_SECS } else { 60 };

                    let result = tokio::time::timeout(
                        Duration::from_secs(timeout_secs),
                        self.client.send_heartbeat(
                            &self.public_ip,
                            session_count_fn(),
                            memchain_status,
                            connected_wallets,
                            traffic_delta,
                            vpn_health_status,
                            operator_status,
                            discovery_status,
                            chat_relay_status,
                        ),
                    ).await;

                    match result {
                        Err(_elapsed) => {
                            failures += 1;
                            // Fail-open: preserve last permissions, disconnect nobody.
                            if in_cold_start {
                                info!(
                                    beat = total_beats,
                                    "[HEARTBEAT] Timeout during cold-start (CMS may still be loading)"
                                );
                            } else {
                                warn!(failures, "[HEARTBEAT] Heartbeat outer timeout");
                            }
                        }
                        Ok(Ok(response)) => {
                            if failures > 0 {
                                info!(
                                    previous_failures = failures,
                                    "[HEARTBEAT] Recovered after {} failure(s)", failures
                                );
                            }
                            failures = 0;

                            // Forward commands (unchanged).
                            if let Some(ref commands) = response.commands {
                                if !commands.is_empty() {
                                    info!(
                                        count = commands.len(),
                                        "[HEARTBEAT] Received {} command(s) from CMS",
                                        commands.len()
                                    );
                                    self.forward_commands(commands.clone()).await;
                                }
                            }

                            // Adjust interval from nodeboard Settings.
                            if let Some(next_in) = response.next_heartbeat_in {
                                let clamped = next_in
                                    .max(MIN_HEARTBEAT_INTERVAL_SECS)
                                    .min(MAX_HEARTBEAT_INTERVAL_SECS);
                                if clamped != active_interval_secs {
                                    info!(
                                        current   = active_interval_secs,
                                        requested = next_in,
                                        applied   = clamped,
                                        "[HEARTBEAT] CMS requested interval change"
                                    );
                                    active_interval_secs = clamped;
                                    interval = tokio::time::interval(Duration::from_secs(clamped));
                                }
                            }

                            // v1.0.0-Membership: handle tier + permissions.
                            self.handle_membership_response(&response).await;
                        }
                        Ok(Err(e)) => {
                            failures += 1;
                            // Fail-open: preserve last permissions, disconnect nobody.
                            if in_cold_start {
                                info!(
                                    beat = total_beats,
                                    failures,
                                    "[HEARTBEAT] Failed during cold-start (expected): {}", e
                                );
                            } else if failures >= 5 {
                                error!(failures, "[HEARTBEAT] Persistent failure: {}", e);
                            } else if failures >= 3 {
                                warn!(failures, "[HEARTBEAT] Failed: {}", e);
                            } else {
                                debug!(failures, "[HEARTBEAT] Failed (transient): {}", e);
                            }
                        }
                    }
                }
            }
        }
    }

    async fn forward_commands(&self, commands: Vec<Command>) {
        let Some(ref tx) = self.command_tx else {
            warn!(
                count = commands.len(),
                "[HEARTBEAT] Received commands but no command channel — discarding"
            );
            return;
        };

        for cmd in commands {
            debug!(
                command_id = %cmd.id,
                action     = %cmd.action,
                priority   = cmd.priority,
                "[HEARTBEAT] Forwarding command to handler"
            );
            if let Err(e) = tx.try_send(cmd) {
                match e {
                    mpsc::error::TrySendError::Full(cmd) => {
                        warn!(
                            command_id = %cmd.id,
                            "[HEARTBEAT] Command channel full — dropping"
                        );
                    }
                    mpsc::error::TrySendError::Closed(cmd) => {
                        error!(
                            command_id = %cmd.id,
                            "[HEARTBEAT] Command channel closed — handler may have crashed"
                        );
                    }
                }
            }
        }
    }

    /// Handles node_tier and user_permissions from a heartbeat response.
    ///
    /// Updates caches, then disconnects any session that violates:
    ///   - node_tier == "premium" && user.tier == "free"
    ///   - !traffic_allowed  (Free quota exceeded)
    ///   - !can_access_premium_nodes && node_tier == "premium"
    ///
    /// Fail-open: no-op if sessions or udp are not injected.
    async fn handle_membership_response(&self, response: &super::client::HeartbeatResponse) {
        // Update node_tier cache.
        if let Some(ref tier) = response.node_tier {
            *self.node_tier.write() = tier.clone();
            info!(tier = %tier, "[MEMBERSHIP] node_tier updated");
        }

        if let Some(ref policy) = response.node_policy {
            if let Some(ref runtime_policy) = self.node_policy {
                runtime_policy.update(NodePolicySnapshot {
                    node_tier: policy.node_tier.clone(),
                    maintenance_mode: policy.maintenance_mode,
                    max_sessions: policy.max_sessions,
                    bandwidth_limit_mbps: policy.bandwidth_limit_mbps,
                    heartbeat_interval_seconds: policy.heartbeat_interval_seconds,
                    updated_at: policy.updated_at.clone(),
                });
            }
            info!(
                node_tier = %policy.node_tier,
                maintenance_mode = policy.maintenance_mode,
                max_sessions = policy.max_sessions,
                bandwidth_limit_mbps = policy.bandwidth_limit_mbps,
                heartbeat_interval_seconds = policy.heartbeat_interval_seconds,
                updated_at = ?policy.updated_at,
                "[NODE_POLICY] CMS operator policy updated"
            );
            if !policy.node_tier.is_empty() {
                *self.node_tier.write() = policy.node_tier.clone();
            }
        }

        // Update user_permissions cache (only when non-empty).
        if !response.user_permissions.is_empty() {
            *self.user_permissions.write() = response.user_permissions.clone();
        }

        self.sync_operator_bans(response);

        // Voucher rollout note:
        // Billing identity is now represented by blind-signed vouchers issued by
        // the CMS. `Session::wallet_hex` is the VPN transport identity from the
        // ClientHello; it is not the user's wallet and must not be used for
        // membership/quota enforcement. Doing so disconnects valid voucher
        // sessions a few seconds after handshake and leaves clients with an
        // established tunnel whose packets hit "Session NOT FOUND".
        //
        // Keep heartbeat cache updates above, but skip legacy per-wallet session
        // enforcement on the node. Voucher issuance/verifier is the authority.
        let enforce_legacy_wallet_sessions =
            std::env::var("AERONYX_ENFORCE_LEGACY_WALLET_SESSIONS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        if !enforce_legacy_wallet_sessions {
            debug!("[MEMBERSHIP] legacy wallet/session enforcement skipped; voucher auth is authoritative");
            return;
        }

        let Some(ref sessions) = self.sessions else {
            return;
        };
        let Some(ref udp) = self.udp else {
            return;
        };

        let node_tier = self.node_tier.read().clone();
        let permissions = self.user_permissions.read().clone();

        if permissions.is_empty() {
            return;
        }

        // ── Restore access for wallets whose permissions improved ─────────
        // Check deny reason independently:
        //   NoPremiumAccess → remove if can_access_premium_nodes is now true
        //   QuotaExceeded   → remove if traffic_allowed is now true
        // Do NOT combine into a single now_ok flag — a wallet can have
        // NoPremiumAccess cleared (tier upgrade) while still over quota,
        // or vice versa. Each reason is independent.
        if let Some(ref dl) = self.deny_list {
            for (wallet, perm) in &permissions {
                if let Some(reason) = dl.deny_reason(wallet) {
                    let should_remove = match reason {
                        DenyReason::NoPremiumAccess => perm.can_access_premium_nodes,
                        DenyReason::QuotaExceeded => perm.traffic_allowed,
                        DenyReason::OperatorBan => false,
                    };
                    if should_remove {
                        dl.remove(wallet);
                    }
                }
            }
        }

        // ── Collect sessions that violate membership rules ────────────────
        //
        // Rule 1: node is premium AND wallet cannot access premium nodes.
        //   Covers both "free tier on premium node" and any future tier that
        //   lacks premium access. can_access_premium_nodes is the single
        //   authoritative field — no need to check tier separately.
        //
        // Rule 2: traffic quota exhausted (Free tier monthly limit).
        let to_disconnect: Vec<_> = sessions
            .all_sessions()
            .into_iter()
            .filter_map(|sess| {
                let perm = permissions.get(&sess.wallet_hex)?;

                let rule1 = node_tier == "premium" && !perm.can_access_premium_nodes;
                let rule2 = !perm.traffic_allowed;

                if rule1 || rule2 {
                    let reason = if rule2 {
                        DenyReason::QuotaExceeded
                    } else {
                        DenyReason::NoPremiumAccess
                    };
                    Some((sess, reason))
                } else {
                    None
                }
            })
            .collect();

        for (sess, reason) in &to_disconnect {
            info!(
                wallet     = %&sess.wallet_hex[..8],
                session_id = %sess.id,
                reason     = %reason,
                "[MEMBERSHIP] Disconnecting session"
            );

            // Add to deny list BEFORE sending RESET so that if the client
            // reconnects before we process the next heartbeat, the
            // handshake rejects it immediately.
            if let Some(ref dl) = self.deny_list {
                dl.add(&sess.wallet_hex, reason.clone());
            }

            // Send RESET (0xFF) — client re-handshakes immediately,
            // which will be rejected by the deny list check.
            let _ = udp.send(&[0xFFu8], &sess.client_endpoint).await;

            // Remove session — cleans up routing + wallet_index + cooldown.
            sessions.remove(&sess.id);
        }

        if !to_disconnect.is_empty() {
            info!(
                count     = to_disconnect.len(),
                node_tier = %node_tier,
                "[MEMBERSHIP] Disconnected {} non-compliant session(s)",
                to_disconnect.len()
            );
        }
    }

    fn sync_operator_bans(&self, response: &super::client::HeartbeatResponse) {
        let Some(ref operator_bans) = response.operator_bans else {
            return;
        };
        let Some(ref deny_list) = self.deny_list else {
            return;
        };

        let desired: HashSet<String> = operator_bans
            .iter()
            .filter_map(|wallet| normalize_wallet_hex(wallet))
            .collect();

        for wallet in &desired {
            if deny_list.deny_reason(wallet) != Some(DenyReason::OperatorBan) {
                deny_list.add(wallet, DenyReason::OperatorBan);
            }
        }

        for wallet in deny_list.wallets_for_reason(DenyReason::OperatorBan) {
            if !desired.contains(&wallet) {
                deny_list.remove(&wallet);
            }
        }

        debug!(
            count = desired.len(),
            "[OPERATOR_BAN] Synced operator wallet bans from CMS"
        );
    }
}

fn normalize_wallet_hex(value: &str) -> Option<String> {
    let wallet = value.trim().to_ascii_lowercase();
    if wallet.len() == 64 && wallet.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Some(wallet)
    } else {
        None
    }
}

// ============================================
// SessionReporter
// ============================================

pub struct SessionReporter {
    client: Arc<ManagementClient>,
    event_rx: mpsc::Receiver<SessionEvent>,
}

impl SessionReporter {
    pub fn new(client: Arc<ManagementClient>) -> (Self, mpsc::Sender<SessionEvent>) {
        let (tx, rx) = mpsc::channel(1000);
        (
            Self {
                client,
                event_rx: rx,
            },
            tx,
        )
    }

    pub async fn run(mut self, mut shutdown: tokio::sync::broadcast::Receiver<()>) {
        info!("[SESSION_REPORTER] Started");
        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("[SESSION_REPORTER] Stopping");
                    break;
                }
                Some(event) = self.event_rx.recv() => {
                    let event_type = format!("{:?}", event.event_type);
                    if let Err(e) = self.client.report_session_event(event.to_report()).await {
                        warn!(
                            event_type = %event_type,
                            "[SESSION_REPORTER] Report failed: {}", e
                        );
                    } else {
                        debug!(
                            event_type = %event_type,
                            "[SESSION_REPORTER] Report sent"
                        );
                    }
                }
            }
        }
    }
}

// ============================================
// SessionEventSender
// ============================================

#[derive(Clone)]
pub struct SessionEventSender {
    tx: Option<mpsc::Sender<SessionEvent>>,
}

impl SessionEventSender {
    pub fn new(tx: mpsc::Sender<SessionEvent>) -> Self {
        Self { tx: Some(tx) }
    }

    pub fn disabled() -> Self {
        Self { tx: None }
    }

    pub fn session_created(
        &self,
        session_id: &str,
        client_wallet: Option<String>,
        virtual_ip: Option<String>,
    ) {
        self.try_send(SessionEvent::created(
            session_id.to_string(),
            client_wallet,
            virtual_ip,
        ));
    }

    pub fn session_ended(
        &self,
        session_id: &str,
        client_wallet: Option<String>,
        virtual_ip: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
        quality: SessionQuality,
    ) {
        self.try_send(SessionEvent::ended(
            session_id.to_string(),
            client_wallet,
            virtual_ip,
            bytes_in,
            bytes_out,
            quality,
        ));
    }

    pub fn session_traffic_snapshot(
        &self,
        session_id: &str,
        client_wallet: Option<String>,
        virtual_ip: Option<String>,
        bytes_in: u64,
        bytes_out: u64,
        quality: SessionQuality,
    ) {
        self.try_send(SessionEvent::snapshot(
            session_id.to_string(),
            client_wallet,
            virtual_ip,
            bytes_in,
            bytes_out,
            quality,
        ));
    }

    fn try_send(&self, event: SessionEvent) {
        if let Some(ref tx) = self.tx {
            if tx.try_send(event).is_err() {
                debug!("[SESSION_REPORTER] Event channel full or closed — event dropped");
            }
        }
    }
}
