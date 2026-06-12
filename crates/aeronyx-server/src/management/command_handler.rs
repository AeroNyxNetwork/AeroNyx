//! ============================================
//! File: crates/aeronyx-server/src/management/command_handler.rs
//! Path: aeronyx-server/src/management/command_handler.rs
//! ============================================
//!
//! ## Creation Reason
//! VPN operations command pipeline. Receives structured `Command` objects from
//! the heartbeat response and dispatches safe node operations by
//! `command.action`.
//!
//! ## Modification Reason
//! - v1.5.0: removed legacy non-VPN agent lifecycle command dispatch. The node
//!   now accepts only VPN operations commands from the centralized control
//!   plane.
//!
//! ## Main Functionality
//! - `CommandHandler`: Background async task that consumes commands
//!   from a channel and dispatches them.
//! - Maintains an internal log of received command IDs to avoid
//!   re-executing duplicate commands (CMS may resend until acknowledged).
//! - Reports execution status back to CMS via `ManagementClient`.
//!
//! ## Main Logical Flow
//! 1. `HeartbeatReporter` receives `HeartbeatResponse` with `commands`
//! 2. Commands are sent to `CommandHandler` via `mpsc::Sender<Command>`
//! 3. `CommandHandler::run()` loop receives each `Command`
//! 4. Deduplication check (skip if `command.id` already processed)
//! 5. Match on `command.action` → dispatch to bounded VPN operation handlers
//! 6. Report `CommandStatusReport` back to CMS
//!
//! ## Dependencies
//! - `super::client::ManagementClient` — for status reporting to CMS
//! - `super::models::*` — Command, CommandStatusReport, CommandExecutionStatus
//! - `crate::services::*` — VPN sessions, deny list, and retained runtime state
//!
//! ## ⚠️ Important Note for Next Developer
//! - The deduplication set (`processed_ids`) is in-memory only.
//!   On server restart it will be empty, so CMS should handle
//!   idempotency on its side as well.
//! - The set is capped at `MAX_PROCESSED_IDS` (1000) to prevent
//!   unbounded memory growth. When full, the oldest half is evicted.
//! - Unknown actions are logged and reported as `failed` — never panic.
//! - Legacy non-VPN lifecycle commands are not dispatched. They fall through to
//!   the unknown-action failure path so the CMS audit trail shows rejection.
//!
//! ## Last Modified
//! v1.3.0 - Initial command pipeline
//! v1.5.0 - VPN-only operations dispatch
//! ============================================

use std::collections::HashSet;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use super::client::ManagementClient;
use super::reporter::{SessionEventSender, SessionQuality};
use super::models::{Command, CommandExecutionStatus, CommandStatusReport};
use aeronyx_common::types::SessionId;
use crate::services::{DenyList, DenyReason, NodePolicyRuntime, SessionManager};

// ============================================
// Constants
// ============================================

/// Maximum number of command IDs to retain for deduplication.
/// When this limit is reached, the oldest half is evicted.
const MAX_PROCESSED_IDS: usize = 1000;
const VPN_COMMAND_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_DIAGNOSTIC_MESSAGE: usize = 3800;

fn session_quality_from_stats(snap: &crate::services::session::StatsSnapshot) -> SessionQuality {
    let rejects = snap.replays_rejected + snap.too_old_rejected;
    let accepted = snap.packets_rx + snap.packets_tx;
    let packet_loss = if accepted + rejects > 0 {
        Some((rejects as f64 / (accepted + rejects) as f64) * 100.0)
    } else {
        None
    };

    SessionQuality {
        last_rx_at: (snap.last_rx_at > 0).then_some(snap.last_rx_at),
        last_tx_at: (snap.last_tx_at > 0).then_some(snap.last_tx_at),
        rtt_ms: (snap.rtt_us > 0).then_some(snap.rtt_us as f64 / 1000.0),
        packet_loss,
        replay_rejections: Some(snap.replays_rejected),
        too_old_rejections: Some(snap.too_old_rejected),
        packets_rx: Some(snap.packets_rx),
        packets_tx: Some(snap.packets_tx),
        keepalive_probes_sent: Some(snap.keepalive_probes_sent),
        keepalive_acks: Some(snap.keepalive_acks),
        keepalive_missed: Some(snap.keepalive_missed),
        keepalive_pending: Some(snap.keepalive_pending),
    }
}

// ============================================
// CommandHandler
// ============================================

/// Background task that receives CMS commands and dispatches them.
///
/// ## Architecture
/// ```text
/// HeartbeatReporter
///       │  (mpsc::Sender<Command>)
///       ▼
/// CommandHandler::run()
///       │
///       ├── deduplicate (skip seen command.id)
///       ├── match command.action
///       │     ├── "system_info"     → collect read-only diagnostics
///       │     ├── "collect_logs"    → collect bounded service logs
///       │     ├── "refresh_config"  → validate management config
///       │     ├── "kick_session"    → disconnect one active tunnel
///       │     ├── "ban_wallet"      → block a wallet on this node
///       │     ├── "unban_wallet"    → remove a wallet block
///       │     ├── "restart_service" → restart the fixed VPN service
///       │     ├── "apply_policy"    → report current runtime policy snapshot
///       │     └── unknown           → report failed
///       │
///       └── report_command_status() → CMS
/// ```
pub struct CommandHandler {
    /// Channel receiver for incoming commands from HeartbeatReporter.
    command_rx: mpsc::Receiver<Command>,

    /// Shared management client for reporting status to CMS.
    client: Arc<ManagementClient>,

    /// VPN session manager used by nodeboard operations commands.
    sessions: Option<Arc<SessionManager>>,

    /// Session event sender used to notify CMS after control-plane kicks.
    session_events: SessionEventSender,

    /// Shared VPN deny list used by handshake and nodeboard wallet bans.
    deny_list: Option<Arc<DenyList>>,

    /// Runtime operator policy used by handshake and bandwidth hot paths.
    node_policy: Option<Arc<NodePolicyRuntime>>,

    /// Set of already-processed command IDs for deduplication.
    /// CMS may resend commands in consecutive heartbeats until
    /// it receives an acknowledgement.
    processed_ids: HashSet<String>,

    /// Ordered list of processed IDs for eviction when set is full.
    /// We maintain insertion order so we can evict the oldest half.
    processed_ids_order: Vec<String>,
}

impl CommandHandler {
    /// Creates a new CommandHandler.
    ///
    /// # Arguments
    /// * `command_rx` - Receiver end of the command channel
    /// * `client` - Shared ManagementClient for CMS communication
    /// # Returns
    /// A new `CommandHandler` ready to be spawned with `.run()`.
    pub fn new(
        command_rx: mpsc::Receiver<Command>,
        client: Arc<ManagementClient>,
    ) -> Self {
        Self {
            command_rx,
            client,
            sessions: None,
            session_events: SessionEventSender::disabled(),
            deny_list: None,
            node_policy: None,
            processed_ids: HashSet::with_capacity(128),
            processed_ids_order: Vec::with_capacity(128),
        }
    }

    pub fn with_session_control(
        mut self,
        sessions: Arc<SessionManager>,
        session_events: SessionEventSender,
    ) -> Self {
        self.sessions = Some(sessions);
        self.session_events = session_events;
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

    /// Runs the command handler loop until shutdown signal received.
    ///
    /// This should be spawned as a `tokio::spawn` task from `server.rs`.
    ///
    /// # Arguments
    /// * `shutdown` - Broadcast receiver for graceful shutdown
    pub async fn run(
        mut self,
        mut shutdown: tokio::sync::broadcast::Receiver<()>,
    ) {
        info!("[CMD_HANDLER] Command handler started");

        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    info!("[CMD_HANDLER] Shutting down");
                    break;
                }
                Some(command) = self.command_rx.recv() => {
                    self.handle_command(command).await;
                }
            }
        }

        info!(
            processed_commands = self.processed_ids.len(),
            "[CMD_HANDLER] Command handler stopped"
        );
    }

    /// Processes a single command: deduplicate, dispatch, report.
    async fn handle_command(&mut self, command: Command) {
        let cmd_id = &command.id;
        let action = &command.action;

        // ===== Deduplication =====
        if self.processed_ids.contains(cmd_id) {
            debug!(
                command_id = %cmd_id,
                action = %action,
                "[CMD_HANDLER] ⏭️ Duplicate command — skipping"
            );
            return;
        }

        info!(
            command_id = %cmd_id,
            action = %action,
            priority = command.priority,
            "[CMD_HANDLER] 📥 Received command"
        );

        // Mark as processed (before execution to prevent re-entry)
        self.mark_processed(cmd_id.clone());

        // Report "received" status to CMS
        self.report_status_for_agent_type(
            "vpn",
            cmd_id,
            CommandExecutionStatus::Received,
            0,
            "Command received by node",
        ).await;

        // ===== Dispatch by action =====
        match action.as_str() {
            "system_info" => {
                self.handle_system_info(&command).await;
            }
            "collect_logs" => {
                self.handle_collect_logs(&command).await;
            }
            "kick_session" => {
                self.handle_kick_session(&command).await;
            }
            "refresh_config" => {
                self.handle_refresh_config(&command).await;
            }
            "ban_wallet" => {
                self.handle_ban_wallet(&command).await;
            }
            "unban_wallet" => {
                self.handle_unban_wallet(&command).await;
            }
            "restart_service" => {
                self.handle_restart_service(&command).await;
            }
            "apply_policy" => {
                self.handle_apply_policy(&command).await;
            }
            unknown => {
                warn!(
                    command_id = %cmd_id,
                    action = %unknown,
                    "[CMD_HANDLER] ❓ Unknown action — reporting failed"
                );
                self.report_status_for_agent_type(
                    "vpn",
                    cmd_id,
                    CommandExecutionStatus::Failed,
                    0,
                    &format!("Unknown action: {}", unknown),
                ).await;
            }
        }
    }

    /// Handles `system_info` command.
    ///
    /// This is a read-only VPN operations command used by nodeboard to collect
    /// enough node context for remote diagnosis without SSH access.
    async fn handle_system_info(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 🩺 system_info"
        );

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::InProgress,
            20,
            "Collecting VPN node diagnostics",
        ).await;

        let service_name = command.params
            .get("service_name")
            .and_then(|value| value.as_str())
            .unwrap_or("aeronyx-server");
        let tun_device = command.params
            .get("tun_device")
            .and_then(|value| value.as_str())
            .unwrap_or("aeronyx0");

        let mut parts = Vec::new();
        parts.push(format!("uptime: {}", run_readonly_command("uptime", &[]).await));
        parts.push(format!("kernel: {}", run_readonly_command("uname", &["-a"]).await));
        parts.push(format!(
            "service({}): {}",
            service_name,
            run_readonly_command("systemctl", &["is-active", service_name]).await
        ));
        parts.push(format!(
            "tun({}): {}",
            tun_device,
            run_readonly_command("ip", &["addr", "show", "dev", tun_device]).await
        ));
        parts.push(format!(
            "udp_listeners: {}",
            run_readonly_command("ss", &["-lun"]).await
        ));
        parts.push(format!(
            "ip_forward: {}",
            read_trimmed_file("/proc/sys/net/ipv4/ip_forward").await
        ));

        let message = sanitize_and_truncate(&parts.join("\n"));
        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::Completed,
            100,
            &message,
        ).await;
    }

    /// Handles `collect_logs` command.
    ///
    /// The command intentionally returns a short, redacted journal tail only.
    /// It is for operational diagnosis, not unrestricted remote shell access.
    async fn handle_collect_logs(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 📄 collect_logs"
        );

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::InProgress,
            20,
            "Collecting recent VPN service logs",
        ).await;

        let service_name = command.params
            .get("service_name")
            .and_then(|value| value.as_str())
            .unwrap_or("aeronyx-server");
        let lines = command.params
            .get("lines")
            .and_then(|value| value.as_u64())
            .unwrap_or(60)
            .clamp(10, 120)
            .to_string();

        let output = run_readonly_command(
            "journalctl",
            &[
                "-u",
                service_name,
                "--no-pager",
                "--since",
                "30 minutes ago",
                "-n",
                &lines,
            ],
        ).await;

        let message = sanitize_and_truncate(&format!(
            "recent_logs({}; last {} lines):\n{}",
            service_name, lines, output
        ));

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::Completed,
            100,
            &message,
        ).await;
    }

    /// Handles `kick_session` command.
    ///
    /// This is a bounded control-plane operation: the CMS passes exactly one
    /// base64 `session_id`; the node removes that in-memory VPN session and
    /// reports a final cumulative `session_ended` event.
    async fn handle_kick_session(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 🧹 kick_session"
        );

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::InProgress,
            30,
            "Kicking VPN session",
        ).await;

        let Some(ref sessions) = self.sessions else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "VPN session manager is not available",
            ).await;
            return;
        };

        let Some(session_id_raw) = command.params
            .get("session_id")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "session_id is required",
            ).await;
            return;
        };

        let session_id = match SessionId::from_str(session_id_raw) {
            Ok(value) => value,
            Err(error) => {
                self.report_status_for_agent_type(
                    "vpn",
                    &command.id,
                    CommandExecutionStatus::Failed,
                    0,
                    &format!("invalid session_id: {}", error),
                ).await;
                return;
            }
        };

        let Some(session) = sessions.remove(&session_id) else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "session not found on node",
            ).await;
            return;
        };

        let stats = session.stats_snapshot();
        let quality = session_quality_from_stats(&stats);
        self.session_events.session_ended(
            session_id_raw,
            Some(session.wallet_hex.clone()),
            Some(session.virtual_ip.to_string()),
            stats.bytes_rx,
            stats.bytes_tx,
            quality,
        );

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::Completed,
            100,
            &format!(
                "VPN session kicked: session_id={}, virtual_ip={}, bytes_in={}, bytes_out={}",
                session_id_raw,
                session.virtual_ip,
                stats.bytes_rx,
                stats.bytes_tx
            ),
        ).await;
    }

    /// Handles `refresh_config` command.
    ///
    /// This is the safe first step toward centralized VPN policy refresh. The
    /// node validates and summarizes the fixed management configuration it is
    /// already running with, plus its node binding file, without accepting any
    /// caller-provided path or shell parameter.
    async fn handle_refresh_config(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 🔁 refresh_config"
        );

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::InProgress,
            25,
            "Refreshing VPN management configuration",
        ).await;

        let config = self.client.config();
        let validation = match config.validate() {
            Ok(()) => "ok".to_string(),
            Err(error) => {
                self.report_status_for_agent_type(
                    "vpn",
                    &command.id,
                    CommandExecutionStatus::Failed,
                    0,
                    &format!("management config validation failed: {}", error),
                ).await;
                return;
            }
        };

        let node_info = read_node_info_summary(&config.node_info_path).await;
        let message = sanitize_and_truncate(&format!(
            "VPN management configuration refreshed\n\
             config_validation: {}\n\
             cms_url: {}\n\
             heartbeat_interval_secs: {}\n\
             session_report_interval_secs: {}\n\
             request_timeout_secs: {}\n\
             max_retries: {}\n\
             node_info_path: {}\n\
             node_info: {}",
            validation,
            config.cms_url,
            config.heartbeat_interval_secs,
            config.session_report_interval_secs,
            config.request_timeout_secs,
            config.max_retries,
            config.node_info_path,
            node_info
        ));

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::Completed,
            100,
            &message,
        ).await;
    }

    /// Handles `apply_policy` command.
    ///
    /// The CMS sends policy values in the heartbeat response; the runtime
    /// policy cache is updated before commands from that heartbeat are handled.
    /// This command is therefore a safe acknowledgement point: it reports the
    /// node's current runtime policy snapshot back to nodeboard without
    /// accepting caller-provided policy values or shell parameters.
    async fn handle_apply_policy(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 📋 apply_policy"
        );

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::InProgress,
            40,
            "Confirming runtime VPN policy snapshot",
        ).await;

        let Some(ref node_policy) = self.node_policy else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "runtime node policy is not available",
            ).await;
            return;
        };

        let snapshot = node_policy.snapshot();
        let snapshot_json = serde_json::to_string(&snapshot)
            .unwrap_or_else(|_| "{}".to_string());
        let message = sanitize_and_truncate(&format!(
            "Runtime VPN policy acknowledged\npolicy: {}\nsource: heartbeat_node_policy",
            snapshot_json
        ));

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::Completed,
            100,
            &message,
        ).await;
    }

    /// Handles `ban_wallet` command.
    ///
    /// Adds the wallet to the shared deny list used by handshake processing,
    /// then disconnects all currently active sessions for that wallet. The CMS
    /// passes the wallet as lowercase hex only; this handler validates again
    /// before touching the runtime control plane.
    async fn handle_ban_wallet(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 🚫 ban_wallet"
        );

        let Some(wallet_hex) = normalize_wallet_hex(command.params.get("wallet_hex")) else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "wallet_hex must be 64 lowercase hex characters",
            ).await;
            return;
        };
        let Some(wallet_bytes) = wallet_hex_to_bytes(&wallet_hex) else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "wallet_hex could not be decoded",
            ).await;
            return;
        };
        let reason = command.params
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or("operator_ban");

        let Some(ref deny_list) = self.deny_list else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "VPN deny list is not available",
            ).await;
            return;
        };

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::InProgress,
            35,
            "Adding wallet to VPN deny list",
        ).await;

        deny_list.add(&wallet_hex, DenyReason::OperatorBan);

        let mut disconnected = 0usize;
        if let Some(ref sessions) = self.sessions {
            let active = sessions.get_all_by_wallet(&wallet_bytes);
            for session in active {
                if let Some(removed) = sessions.remove(&session.id) {
                    let stats = removed.stats_snapshot();
                    let quality = session_quality_from_stats(&stats);
                    self.session_events.session_ended(
                        &removed.id.to_string(),
                        Some(removed.wallet_hex.clone()),
                        Some(removed.virtual_ip.to_string()),
                        stats.bytes_rx,
                        stats.bytes_tx,
                        quality,
                    );
                    disconnected += 1;
                }
            }
        }

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::Completed,
            100,
            &format!(
                "Wallet banned: wallet_prefix={}, reason={}, disconnected_sessions={}",
                &wallet_hex[..12],
                sanitize_inline(reason, 80),
                disconnected
            ),
        ).await;
    }

    /// Handles `unban_wallet` command.
    async fn handle_unban_wallet(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] ✅ unban_wallet"
        );

        let Some(wallet_hex) = normalize_wallet_hex(command.params.get("wallet_hex")) else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "wallet_hex must be 64 lowercase hex characters",
            ).await;
            return;
        };

        let Some(ref deny_list) = self.deny_list else {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "VPN deny list is not available",
            ).await;
            return;
        };

        deny_list.remove(&wallet_hex);
        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::Completed,
            100,
            &format!("Wallet unbanned: wallet_prefix={}", &wallet_hex[..12]),
        ).await;
    }

    /// Handles `restart_service` command.
    ///
    /// The node first verifies that the fixed systemd service is loaded, then
    /// reports that the restart has been scheduled and asks systemd to restart
    /// it a few seconds later. This keeps command audit from being stranded
    /// when the current process exits and avoids false success on foreground
    /// deployments without an installed service unit.
    async fn handle_restart_service(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 🔄 restart_service"
        );

        let service_name = command.params
            .get("service_name")
            .and_then(|value| value.as_str())
            .unwrap_or("aeronyx-server");
        if service_name != "aeronyx-server" {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "restart_service only supports aeronyx-server",
            ).await;
            return;
        }

        let confirm = command.params
            .get("confirm")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if confirm != "restart" {
            self.report_status_for_agent_type(
                "vpn",
                &command.id,
                CommandExecutionStatus::Failed,
                0,
                "restart_service requires confirm=restart",
            ).await;
            return;
        }

        self.report_status_for_agent_type(
            "vpn",
            &command.id,
            CommandExecutionStatus::InProgress,
            50,
            "Scheduling VPN service restart",
        ).await;

        match schedule_service_restart(&command.id, service_name).await {
            Ok(message) => {
                self.report_status_for_agent_type(
                    "vpn",
                    &command.id,
                    CommandExecutionStatus::Completed,
                    100,
                    &message,
                ).await;
            }
            Err(message) => {
                self.report_status_for_agent_type(
                    "vpn",
                    &command.id,
                    CommandExecutionStatus::Failed,
                    0,
                    &message,
                ).await;
            }
        }
    }

    // ============================================
    // Status Reporting
    // ============================================

    /// Reports command execution status to CMS.
    ///
    /// Uses `ManagementClient::report_command_status()` which sends
    /// a signed `POST /node/vpn/status` request.
    ///
    /// Failures are logged but do not block command processing —
    /// CMS can always re-query via the next heartbeat.
    async fn report_status(
        &self,
        command_id: &str,
        status: CommandExecutionStatus,
        progress: u8,
        message: &str,
    ) {
        self.report_status_for_agent_type(
            "vpn",
            command_id,
            status,
            progress,
            message,
        ).await;
    }

    async fn report_status_for_agent_type(
        &self,
        agent_type: &str,
        command_id: &str,
        status: CommandExecutionStatus,
        progress: u8,
        message: &str,
    ) {
        let report = CommandStatusReport {
            command_id: command_id.to_string(),
            agent_type: agent_type.to_string(),
            status,
            progress,
            message: message.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        debug!(
            command_id = %command_id,
            status = ?status,
            progress = progress,
            "[CMD_HANDLER] 📤 Reporting status to CMS"
        );

        if let Err(e) = self.client.report_command_status(&report).await {
            warn!(
                command_id = %command_id,
                error = %e,
                "[CMD_HANDLER] ⚠️ Failed to report command status to CMS"
            );
        }
    }

    // ============================================
    // Deduplication
    // ============================================

    /// Marks a command ID as processed and handles eviction if the
    /// set exceeds `MAX_PROCESSED_IDS`.
    fn mark_processed(&mut self, id: String) {
        // Evict oldest half if at capacity
        if self.processed_ids.len() >= MAX_PROCESSED_IDS {
            let evict_count = MAX_PROCESSED_IDS / 2;
            let to_evict: Vec<String> = self.processed_ids_order
                .drain(..evict_count)
                .collect();
            for old_id in &to_evict {
                self.processed_ids.remove(old_id);
            }
            debug!(
                evicted = evict_count,
                remaining = self.processed_ids.len(),
                "[CMD_HANDLER] 🧹 Evicted old command IDs"
            );
        }

        self.processed_ids.insert(id.clone());
        self.processed_ids_order.push(id);
    }
}

async fn run_readonly_command(program: &str, args: &[&str]) -> String {
    let output = timeout(
        VPN_COMMAND_TIMEOUT,
        TokioCommand::new(program).args(args).output(),
    ).await;

    match output {
        Ok(Ok(result)) => {
            let mut text = String::new();
            if !result.stdout.is_empty() {
                text.push_str(&String::from_utf8_lossy(&result.stdout));
            }
            if !result.stderr.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&String::from_utf8_lossy(&result.stderr));
            }
            if text.trim().is_empty() {
                format!("exit={}", result.status)
            } else {
                collapse_lines(text.trim(), 18)
            }
        }
        Ok(Err(e)) => format!("{} failed: {}", program, e),
        Err(_) => format!("{} timed out", program),
    }
}

async fn schedule_service_restart(command_id: &str, service_name: &str) -> Result<String, String> {
    ensure_systemd_service_loaded(service_name).await?;

    let suffix: String = command_id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(12)
        .collect();
    let unit = format!("aeronyx-restart-{}", suffix);

    let systemd_run = timeout(
        VPN_COMMAND_TIMEOUT,
        TokioCommand::new("systemd-run")
            .args([
                "--unit",
                &unit,
                "--on-active=3s",
                "/bin/systemctl",
                "restart",
                service_name,
            ])
            .output(),
    ).await;

    match systemd_run {
        Ok(Ok(output)) if output.status.success() => {
            return Ok(format!(
                "VPN service restart scheduled via systemd-run unit={} delay=3s",
                unit
            ));
        }
        Ok(Ok(output)) => {
            warn!(
                unit = %unit,
                status = %output.status,
                stderr = %String::from_utf8_lossy(&output.stderr),
                "[CMD_HANDLER] systemd-run restart scheduling failed; falling back"
            );
        }
        Ok(Err(error)) => {
            warn!(
                unit = %unit,
                error = %error,
                "[CMD_HANDLER] systemd-run unavailable; falling back"
            );
        }
        Err(_) => {
            warn!(
                unit = %unit,
                "[CMD_HANDLER] systemd-run restart scheduling timed out; falling back"
            );
        }
    }

    let fallback = timeout(
        VPN_COMMAND_TIMEOUT,
        TokioCommand::new("sh")
            .arg("-c")
            .arg("nohup sh -c 'sleep 3; /bin/systemctl restart aeronyx-server' >/dev/null 2>&1 &")
            .status(),
    ).await;

    match fallback {
        Ok(Ok(status)) if status.success() => Ok(
            "VPN service restart scheduled via detached fallback delay=3s".to_string()
        ),
        Ok(Ok(status)) => Err(format!("failed to schedule restart fallback: exit={}", status)),
        Ok(Err(error)) => Err(format!("failed to schedule restart fallback: {}", error)),
        Err(_) => Err("restart fallback scheduling timed out".to_string()),
    }
}

async fn ensure_systemd_service_loaded(service_name: &str) -> Result<(), String> {
    let show = timeout(
        VPN_COMMAND_TIMEOUT,
        TokioCommand::new("systemctl")
            .args(["show", service_name, "--property=LoadState", "--value"])
            .output(),
    ).await;

    match show {
        Ok(Ok(output)) if output.status.success() => {
            let load_state = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if load_state == "loaded" {
                Ok(())
            } else {
                Err(format!(
                    "systemd service {} is not loaded (LoadState={}); install an aeronyx-server service unit or restart this foreground deployment outside nodeboard",
                    service_name,
                    if load_state.is_empty() { "unknown" } else { &load_state }
                ))
            }
        }
        Ok(Ok(output)) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "systemctl show {} failed: {}",
                service_name,
                collapse_lines(stderr.trim(), 4)
            ))
        }
        Ok(Err(error)) => Err(format!("systemctl unavailable for restart_service: {}", error)),
        Err(_) => Err("systemctl service preflight timed out".to_string()),
    }
}

async fn read_node_info_summary(path: &str) -> String {
    let content = match tokio::fs::read_to_string(path).await {
        Ok(value) => value,
        Err(error) => return format!("missing_or_unreadable path={} error={}", path, error),
    };

    let value: serde_json::Value = match serde_json::from_str(&content) {
        Ok(value) => value,
        Err(error) => return format!("invalid_json path={} error={}", path, error),
    };

    let field = |keys: &[&str]| -> String {
        for key in keys {
            if let Some(value) = value.get(key).and_then(|item| item.as_str()) {
                return value.to_string();
            }
        }
        "unknown".to_string()
    };
    let public_key = field(&["public_key"]);
    let public_key_prefix: String = public_key.chars().take(12).collect();

    format!(
        "registered=true id={} name={} status={} key_prefix={} created_at={}",
        field(&["id", "node_id"]),
        field(&["name"]),
        field(&["status"]),
        public_key_prefix,
        field(&["created_at", "registered_at"])
    )
}

fn normalize_wallet_hex(value: Option<&serde_json::Value>) -> Option<String> {
    let wallet = value?.as_str()?.trim().to_ascii_lowercase();
    if wallet.len() == 64 && wallet.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Some(wallet)
    } else {
        None
    }
}

fn wallet_hex_to_bytes(wallet_hex: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(wallet_hex).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Some(out)
}

async fn read_trimmed_file(path: &str) -> String {
    match tokio::fs::read_to_string(path).await {
        Ok(value) => value.trim().to_string(),
        Err(e) => format!("read failed: {}", e),
    }
}

fn collapse_lines(text: &str, max_lines: usize) -> String {
    let mut lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        format!("{}\n...truncated", lines.join("\n"))
    } else {
        lines.join("\n")
    }
}

fn sanitize_inline(input: &str, max_chars: usize) -> String {
    input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':'))
        .take(max_chars)
        .collect::<String>()
}

fn sanitize_and_truncate(input: &str) -> String {
    let mut output = Vec::new();

    for line in input.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("authorization")
            || lower.contains("signature")
            || lower.contains("private_key")
            || lower.contains("secret")
            || lower.contains("voucher")
            || lower.contains("token")
            || lower.contains("api_key")
        {
            output.push("[redacted sensitive log line]".to_string());
        } else {
            output.push(line.to_string());
        }
    }

    let mut text = output.join("\n");
    if text.len() > MAX_DIAGNOSTIC_MESSAGE {
        text.truncate(MAX_DIAGNOSTIC_MESSAGE);
        text.push_str("\n...truncated");
    }
    text
}
