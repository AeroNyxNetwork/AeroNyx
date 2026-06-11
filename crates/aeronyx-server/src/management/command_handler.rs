//! ============================================
//! File: crates/aeronyx-server/src/management/command_handler.rs
//! Path: aeronyx-server/src/management/command_handler.rs
//! ============================================
//!
//! ## Creation Reason
//! Phase 1 of the OpenClaw integration: Command Pipeline.
//! Receives structured `Command` objects from the heartbeat response
//! (via an internal `mpsc` channel) and dispatches them to the
//! appropriate handler based on `command.action`.
//!
//! ## Modification Reason (v1.3.0 Phase 2)
//! - 🌟 Replaced all stub handlers with real `AgentManager` delegation.
//! - 🌟 Added `agent_manager: Arc<AgentManager>` field, injected from server.rs.
//! - 🌟 Each action handler now calls the corresponding AgentManager method
//!   and reports success/failure based on the result.
//!
//! ## Main Functionality
//! - `CommandHandler`: Background async task that consumes commands
//!   from a channel and dispatches them.
//! - Maintains an internal log of received command IDs to avoid
//!   re-executing duplicate commands (CMS may resend until acknowledged).
//! - Reports execution status back to CMS via `ManagementClient`.
//! - 🌟 Delegates agent lifecycle ops to `AgentManager`.
//!
//! ## Main Logical Flow
//! 1. `HeartbeatReporter` receives `HeartbeatResponse` with `commands`
//! 2. Commands are sent to `CommandHandler` via `mpsc::Sender<Command>`
//! 3. `CommandHandler::run()` loop receives each `Command`
//! 4. Deduplication check (skip if `command.id` already processed)
//! 5. Match on `command.action` → dispatch to AgentManager
//! 6. Report `CommandStatusReport` back to CMS
//!
//! ## Dependencies
//! - `super::client::ManagementClient` — for status reporting to CMS
//! - `super::models::*` — Command, CommandStatusReport, CommandExecutionStatus
//! - `crate::services::AgentManager` — OpenClaw lifecycle management
//!
//! ## ⚠️ Important Note for Next Developer
//! - The deduplication set (`processed_ids`) is in-memory only.
//!   On server restart it will be empty, so CMS should handle
//!   idempotency on its side as well.
//! - The set is capped at `MAX_PROCESSED_IDS` (1000) to prevent
//!   unbounded memory growth. When full, the oldest half is evicted.
//! - Unknown actions are logged and reported as `failed` — never panic.
//! - AgentManager methods handle their own progress reporting to CMS,
//!   so CommandHandler only needs to report "received" and handle errors.
//!
//! ## Last Modified
//! v1.3.0 - 🌟 Initial creation (Phase 1: stubs)
//! v1.3.0 - 🌟 Phase 2: Replaced stubs with AgentManager delegation
//! ============================================

use std::collections::HashSet;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use super::client::ManagementClient;
use super::reporter::{SessionEventSender, SessionQuality};
use super::models::{Command, CommandExecutionStatus, CommandStatusReport};
use aeronyx_common::types::SessionId;
use crate::services::{AgentManager, SessionManager};

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
        packet_loss,
        replay_rejections: Some(snap.replays_rejected),
        too_old_rejections: Some(snap.too_old_rejected),
        packets_rx: Some(snap.packets_rx),
        packets_tx: Some(snap.packets_tx),
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
///       │     ├── "install_openclaw"  → AgentManager::install_agent()
///       │     ├── "start_openclaw"    → AgentManager::start_agent()
///       │     ├── "stop_openclaw"     → AgentManager::stop_agent()
///       │     ├── "uninstall_openclaw"→ AgentManager::uninstall_agent()
///       │     ├── "update_openclaw"   → AgentManager::update_agent()
///       │     └── unknown            → report failed
///       │
///       └── report_command_status() → CMS
/// ```
pub struct CommandHandler {
    /// Channel receiver for incoming commands from HeartbeatReporter.
    command_rx: mpsc::Receiver<Command>,

    /// Shared management client for reporting status to CMS.
    client: Arc<ManagementClient>,

    /// 🌟 v1.3.0 Phase 2: OpenClaw agent lifecycle manager.
    agent_manager: Arc<AgentManager>,

    /// VPN session manager used by nodeboard operations commands.
    sessions: Option<Arc<SessionManager>>,

    /// Session event sender used to notify CMS after control-plane kicks.
    session_events: SessionEventSender,

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
    /// * `agent_manager` - Shared AgentManager for OpenClaw lifecycle ops
    ///
    /// # Returns
    /// A new `CommandHandler` ready to be spawned with `.run()`.
    pub fn new(
        command_rx: mpsc::Receiver<Command>,
        client: Arc<ManagementClient>,
        agent_manager: Arc<AgentManager>,
    ) -> Self {
        Self {
            command_rx,
            client,
            agent_manager,
            sessions: None,
            session_events: SessionEventSender::disabled(),
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

        let report_agent_type = command_agent_type(action);

        // Report "received" status to CMS
        self.report_status_for_agent_type(
            report_agent_type,
            cmd_id,
            CommandExecutionStatus::Received,
            0,
            "Command received by node",
        ).await;

        // ===== Dispatch by action =====
        match action.as_str() {
            "install_openclaw" => {
                self.handle_install_openclaw(&command).await;
            }
            "start_openclaw" => {
                self.handle_start_openclaw(&command).await;
            }
            "stop_openclaw" => {
                self.handle_stop_openclaw(&command).await;
            }
            "uninstall_openclaw" => {
                self.handle_uninstall_openclaw(&command).await;
            }
            "update_openclaw" => {
                self.handle_update_openclaw(&command).await;
            }
            "system_info" => {
                self.handle_system_info(&command).await;
            }
            "collect_logs" => {
                self.handle_collect_logs(&command).await;
            }
            "kick_session" => {
                self.handle_kick_session(&command).await;
            }
            "restart_service" => {
                self.handle_restart_service(&command).await;
            }
            unknown => {
                warn!(
                    command_id = %cmd_id,
                    action = %unknown,
                    "[CMD_HANDLER] ❓ Unknown action — reporting failed"
                );
                self.report_status_for_agent_type(
                    report_agent_type,
                    cmd_id,
                    CommandExecutionStatus::Failed,
                    0,
                    &format!("Unknown action: {}", unknown),
                ).await;
            }
        }
    }

    // ============================================
    // Action Handlers (delegating to AgentManager)
    // ============================================

    /// Handles `install_openclaw` command.
    ///
    /// Delegates to `AgentManager::install_agent()` which handles:
    /// 1. Create system user
    /// 2. Install Node.js 22+
    /// 3. Install OpenClaw via npm
    /// 4. Run onboarding with --install-daemon
    /// 5. Start gateway
    /// 6. Verify health
    ///
    /// AgentManager handles its own progress reporting.
    async fn handle_install_openclaw(&self, command: &Command) {
        info!(
            command_id = %command.id,
            params = %command.params,
            "[CMD_HANDLER] 🔧 install_openclaw"
        );

        if let Err(e) = self.agent_manager.install_agent(&command.id, &command.params).await {
            error!(
                command_id = %command.id,
                error = %e,
                "[CMD_HANDLER] ❌ install_openclaw failed"
            );
            // AgentManager already reported failure, but log it here too
        }
    }

    /// Handles `start_openclaw` command.
    async fn handle_start_openclaw(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] ▶️ start_openclaw"
        );

        if let Err(e) = self.agent_manager.start_agent(&command.id).await {
            error!(
                command_id = %command.id,
                error = %e,
                "[CMD_HANDLER] ❌ start_openclaw failed"
            );
        }
    }

    /// Handles `stop_openclaw` command.
    async fn handle_stop_openclaw(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] ⏹️ stop_openclaw"
        );

        if let Err(e) = self.agent_manager.stop_agent(&command.id).await {
            error!(
                command_id = %command.id,
                error = %e,
                "[CMD_HANDLER] ❌ stop_openclaw failed"
            );
        }
    }

    /// Handles `uninstall_openclaw` command.
    async fn handle_uninstall_openclaw(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 🗑️ uninstall_openclaw"
        );

        if let Err(e) = self.agent_manager.uninstall_agent(&command.id).await {
            error!(
                command_id = %command.id,
                error = %e,
                "[CMD_HANDLER] ❌ uninstall_openclaw failed"
            );
        }
    }

    /// Handles `update_openclaw` command.
    async fn handle_update_openclaw(&self, command: &Command) {
        info!(
            command_id = %command.id,
            "[CMD_HANDLER] 🔄 update_openclaw"
        );

        if let Err(e) = self.agent_manager.update_agent(&command.id, &command.params).await {
            error!(
                command_id = %command.id,
                error = %e,
                "[CMD_HANDLER] ❌ update_openclaw failed"
            );
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

        let stats = session.stats.snapshot();
        let quality = session_quality_from_stats(&stats);
        self.session_events.session_ended(
            session_id_raw,
            Some(session.wallet_hex.clone()),
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

    /// Handles `restart_service` command.
    ///
    /// The node first reports that the restart has been scheduled, then asks
    /// systemd to restart the fixed VPN service a few seconds later. This keeps
    /// command audit from being stranded when the current process exits.
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
    /// a signed `POST /node/agent/status` request.
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
            "openclaw",
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

fn command_agent_type(action: &str) -> &'static str {
    match action {
        "install_openclaw"
        | "start_openclaw"
        | "stop_openclaw"
        | "uninstall_openclaw"
        | "update_openclaw" => "openclaw",
        _ => "vpn",
    }
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
