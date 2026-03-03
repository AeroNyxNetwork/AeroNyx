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
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::client::ManagementClient;
use super::models::{Command, CommandExecutionStatus, CommandStatusReport};
use crate::services::AgentManager;

// ============================================
// Constants
// ============================================

/// Maximum number of command IDs to retain for deduplication.
/// When this limit is reached, the oldest half is evicted.
const MAX_PROCESSED_IDS: usize = 1000;

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
            processed_ids: HashSet::with_capacity(128),
            processed_ids_order: Vec::with_capacity(128),
        }
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
        self.report_status(
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
            unknown => {
                warn!(
                    command_id = %cmd_id,
                    action = %unknown,
                    "[CMD_HANDLER] ❓ Unknown action — reporting failed"
                );
                self.report_status(
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
        let report = CommandStatusReport {
            command_id: command_id.to_string(),
            agent_type: "openclaw".to_string(),
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
