//! ============================================
//! File: crates/aeronyx-server/src/services/agent_manager.rs
//! Path: aeronyx-server/src/services/agent_manager.rs
//! ============================================
//!
//! ## Creation Reason
//! Phase 2 of the OpenClaw integration: Agent Process Management.
//!
//! ## Modification Reason (v1.4.0)
//! Major refactor: extracted implementation into sub-modules under `agent/`.
//! This file is now a thin orchestrator (~300 lines) that:
//! - Owns the `AgentState` (status, version, PID, token)
//! - Exposes the public API consumed by `server.rs` and `command_handler.rs`
//! - Delegates to sub-modules for actual work
//!
//! ## Module Delegation
//! ```text
//! agent_manager.rs (this file)
//!   ├── agent::preflight     → PreflightChecker (12 system checks)
//!   ├── agent::installer     → AgentInstaller   (user, Node.js, npm, onboard)
//!   ├── agent::gateway       → GatewayController (start, stop, health, PID)
//!   ├── agent::process_stats → ProcessStats      (Linux /proc CPU & memory)
//!   └── agent::env_config    → EnvConfig         (.env + openclaw config)
//! ```
//!
//! ## Public API (unchanged from v1.3.x — no breaking changes)
//! - `new(client)` → AgentManager
//! - `detect_existing()` → detects installed OpenClaw on startup
//! - `status()` → AgentStatusInfo (for heartbeat)
//! - `gateway_token()` → Option<String>
//! - `install_agent(cmd_id, params)` → Result
//! - `start_agent(cmd_id)` → Result
//! - `stop_agent(cmd_id)` → Result
//! - `uninstall_agent(cmd_id)` → Result
//! - `update_agent(cmd_id, params)` → Result
//! - `periodic_health_check()` → updates state if unhealthy
//!
//! ## ⚠️ Important Note for Next Developer
//! - `server.rs` imports `crate::services::AgentManager` — do NOT rename
//! - The public API signatures must NOT change without updating server.rs
//! - Internal state is behind `RwLock<AgentState>` — hold locks briefly
//! - All CMS progress reporting goes through `self.report_progress()`
//! - Sub-modules are stateless — all state lives here in AgentState
//!
//! ## Last Modified
//! v1.3.0 - 🌟 Initial creation
//! v1.3.2 - 🔄 Added HTTP API enablement + .env loading
//! v1.4.0 - 🔄 Major refactor: extracted into agent/ sub-modules
//! ============================================

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::management::client::ManagementClient;
use crate::management::models::{
    AgentStatus, AgentStatusInfo, CommandExecutionStatus, CommandStatusReport,
};

// Sub-modules (implementation details)
pub(crate) mod agent;

use agent::preflight::PreflightChecker;
use agent::installer::AgentInstaller;
use agent::gateway::GatewayController;
use agent::process_stats::ProcessStats;
use agent::env_config::EnvConfig;
use agent::OPENCLAW_DEFAULT_PORT;

// ============================================
// AgentState
// ============================================

#[derive(Debug)]
struct AgentState {
    status: AgentStatus,
    version: Option<String>,
    message: String,
    pid: Option<u32>,
    gateway_token: Option<String>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            status: AgentStatus::NotInstalled,
            version: None,
            message: "Not installed".to_string(),
            pid: None,
            gateway_token: None,
        }
    }
}

// ============================================
// AgentManager
// ============================================

/// Manages the OpenClaw AI agent lifecycle on this node.
///
/// Thread-safe: wrap in `Arc` and share between CommandHandler
/// (for mutations) and HeartbeatReporter (for status reads).
pub struct AgentManager {
    client: Arc<ManagementClient>,
    state: RwLock<AgentState>,
    gateway_port: u16,
}

impl AgentManager {
    pub fn new(client: Arc<ManagementClient>) -> Self {
        Self {
            client,
            state: RwLock::new(AgentState::default()),
            gateway_port: OPENCLAW_DEFAULT_PORT,
        }
    }

    // ============================================
    // Status Query
    // ============================================

    pub async fn status(&self) -> AgentStatusInfo {
        let state = self.state.read().await;

        let (cpu_usage, memory_mb) = if state.status == AgentStatus::Running {
            if let Some(pid) = state.pid {
                ProcessStats::collect(pid).await
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let local_port = if state.status == AgentStatus::Running {
            Some(self.gateway_port)
        } else {
            None
        };

        AgentStatusInfo {
            status: state.status,
            version: state.version.clone(),
            message: Some(state.message.clone()),
            pid: state.pid,
            local_port,
            cpu_usage,
            memory_mb,
        }
    }

    pub async fn gateway_token(&self) -> Option<String> {
        self.state.read().await.gateway_token.clone()
    }

    // ============================================
    // Startup Detection
    // ============================================

    pub async fn detect_existing(&self) {
        info!("[AGENT] Detecting existing OpenClaw installation...");

        let bin_exists = std::path::Path::new(agent::OPENCLAW_BIN).exists();
        if !bin_exists {
            // Also check system PATH
            let in_path = agent::CommandRunner::run_as_root("which", &["openclaw"]).await.is_ok();
            if !in_path {
                info!("[AGENT] OpenClaw not found — status: NotInstalled");
                return;
            }
        }

        let version = AgentInstaller::get_installed_version().await;
        let gateway_running = GatewayController::check_health(self.gateway_port).await;
        let token = GatewayController::read_token().await;

        // Ensure HTTP API is enabled
        if let Err(e) = EnvConfig::ensure_http_api_enabled().await {
            warn!(error = %e, "[AGENT] ⚠️ Failed to enable HTTP API endpoint");
        }

        let mut state = self.state.write().await;
        state.version = version.clone();
        state.gateway_token = token;

        if gateway_running {
            state.pid = GatewayController::find_pid().await;
            state.status = AgentStatus::Running;
            state.message = format!("Running (v{})", version.as_deref().unwrap_or("unknown"));
            info!(version = ?state.version, pid = ?state.pid, "[AGENT] ✅ OpenClaw detected and running");
        } else {
            state.status = AgentStatus::Stopped;
            state.message = format!("Installed but stopped (v{})", version.as_deref().unwrap_or("unknown"));
            info!(version = ?state.version, "[AGENT] OpenClaw installed but not running");
        }
    }

    // ============================================
    // Install
    // ============================================

    pub async fn install_agent(
        &self,
        command_id: &str,
        params: &serde_json::Value,
    ) -> Result<(), String> {
        // Atomic state check + transition
        {
            let mut state = self.state.write().await;

            if state.status == AgentStatus::Installing {
                return Err("Installation already in progress".to_string());
            }

            if state.status == AgentStatus::Running {
                drop(state);
                self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "OpenClaw is already installed and running").await;
                return Ok(());
            }

            if state.status == AgentStatus::Stopped {
                drop(state);
                return self.start_existing(command_id).await;
            }

            state.status = AgentStatus::Installing;
            state.message = "Starting installation...".to_string();
        }

        // --- Preflight checks ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 5, "Running system checks...").await;
        let report = PreflightChecker::run_all().await;

        if report.has_failures() {
            let summary = report.summary();
            self.set_state(AgentStatus::Error, &summary).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 5, &summary).await;
            return Err(summary);
        }

        if report.warn_count() > 0 {
            info!("[AGENT] Preflight has {} warnings — proceeding with install", report.warn_count());
        }

        // --- Step 1: System user ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 10, "Creating openclaw user...").await;
        if let Err(e) = AgentInstaller::ensure_system_user().await {
            return self.fail_install(command_id, 10, &format!("Failed to create system user: {}", e)).await;
        }

        // --- Step 2: Node.js ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 20, "Setting up Node.js...").await;
        if let Err(e) = AgentInstaller::ensure_nodejs().await {
            return self.fail_install(command_id, 20, &format!("Failed to install Node.js: {}", e)).await;
        }

        // --- Step 3: Build tools ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 30, "Checking build tools...").await;
        let _ = AgentInstaller::ensure_build_tools().await; // Non-fatal

        // --- Step 4: OpenClaw package ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 40, "Installing OpenClaw...").await;
        let version = params.get("version").and_then(|v| v.as_str()).unwrap_or("latest");
        if let Err(e) = AgentInstaller::install_openclaw_package(version).await {
            return self.fail_install(command_id, 40, &format!("Failed to install OpenClaw: {}", e)).await;
        }

        // --- Step 5: Onboarding ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 55, "Configuring OpenClaw...").await;
        if let Err(e) = AgentInstaller::run_onboarding(params).await {
            return self.fail_install(command_id, 55, &format!("Onboarding failed: {}", e)).await;
        }

        // --- Step 6: Post-install doctor ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 65, "Running post-install checks...").await;
        let _ = AgentInstaller::post_install_doctor().await; // Non-fatal

        // --- Step 7: Enable HTTP API ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 70, "Enabling HTTP API...").await;
        if let Err(e) = EnvConfig::ensure_http_api_enabled().await {
            warn!(error = %e, "[AGENT] ⚠️ Failed to enable HTTP API");
        }

        // --- Step 8: Extract token ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 75, "Extracting gateway token...").await;
        {
            let mut state = self.state.write().await;
            state.gateway_token = GatewayController::read_token().await;
        }

        // --- Step 9: Start gateway ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 80, "Starting gateway...").await;
        match GatewayController::start().await {
            Ok(pid) => {
                let mut state = self.state.write().await;
                state.pid = pid;
            }
            Err(e) => {
                return self.fail_install(command_id, 80, &format!("Failed to start gateway: {}", e)).await;
            }
        }

        // --- Step 10: Verify health ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 90, "Verifying gateway health...").await;
        if !GatewayController::wait_for_healthy(self.gateway_port).await {
            return self.fail_install(command_id, 90, "Gateway started but health check timed out").await;
        }

        // --- Finalize ---
        let installed_version = AgentInstaller::get_installed_version().await;
        {
            let mut state = self.state.write().await;
            state.status = AgentStatus::Running;
            state.version = installed_version.clone();
            state.pid = GatewayController::find_pid().await;
            state.message = format!("Running (v{})", installed_version.as_deref().unwrap_or("unknown"));
        }

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "OpenClaw installed and running").await;
        info!(version = ?installed_version, "[AGENT] ✅ Installation complete");
        Ok(())
    }

    /// Handles the "installed but stopped" case during install_agent.
    async fn start_existing(&self, command_id: &str) -> Result<(), String> {
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 80, "Starting existing installation...").await;

        match GatewayController::start().await {
            Ok(pid) => {
                if GatewayController::wait_for_healthy(self.gateway_port).await {
                    let version = AgentInstaller::get_installed_version().await;
                    let mut state = self.state.write().await;
                    state.status = AgentStatus::Running;
                    state.version = version;
                    state.pid = pid.or(GatewayController::find_pid().await);
                    state.message = "Running".to_string();
                    drop(state);
                    self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "OpenClaw started").await;
                    Ok(())
                } else {
                    self.fail_install(command_id, 90, "Gateway health check failed").await
                }
            }
            Err(e) => {
                self.fail_install(command_id, 80, &format!("Failed to start gateway: {}", e)).await
            }
        }
    }

    /// Helper to set error state and report failure.
    async fn fail_install(&self, command_id: &str, progress: u8, message: &str) -> Result<(), String> {
        self.set_state(AgentStatus::Error, message).await;
        self.report_progress(command_id, CommandExecutionStatus::Failed, progress, message).await;
        Err(message.to_string())
    }

    // ============================================
    // Start / Stop
    // ============================================

    pub async fn start_agent(&self, command_id: &str) -> Result<(), String> {
        let current = self.state.read().await.status;

        match current {
            AgentStatus::Running => {
                self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Already running").await;
                return Ok(());
            }
            AgentStatus::NotInstalled => {
                self.report_progress(command_id, CommandExecutionStatus::Failed, 0, "Not installed").await;
                return Err("OpenClaw is not installed".to_string());
            }
            _ => {}
        }

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 30, "Starting gateway...").await;

        match GatewayController::start().await {
            Ok(pid) => {
                if GatewayController::wait_for_healthy(self.gateway_port).await {
                    let mut state = self.state.write().await;
                    state.status = AgentStatus::Running;
                    state.pid = pid.or(GatewayController::find_pid().await);
                    state.message = "Running".to_string();
                    drop(state);
                    self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Gateway started").await;
                    Ok(())
                } else {
                    self.set_state(AgentStatus::Error, "Health check failed").await;
                    self.report_progress(command_id, CommandExecutionStatus::Failed, 50, "Health check failed").await;
                    Err("Gateway failed to start".to_string())
                }
            }
            Err(e) => {
                self.set_state(AgentStatus::Error, &e).await;
                self.report_progress(command_id, CommandExecutionStatus::Failed, 30, &e).await;
                Err(e)
            }
        }
    }

    pub async fn stop_agent(&self, command_id: &str) -> Result<(), String> {
        let (current, pid) = {
            let state = self.state.read().await;
            (state.status, state.pid)
        };

        match current {
            AgentStatus::Stopped | AgentStatus::NotInstalled => {
                self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Not running").await;
                return Ok(());
            }
            _ => {}
        }

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 30, "Stopping gateway...").await;
        let _ = GatewayController::stop(pid).await;

        let mut state = self.state.write().await;
        state.status = AgentStatus::Stopped;
        state.pid = None;
        state.message = "Stopped".to_string();
        drop(state);

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Gateway stopped").await;
        Ok(())
    }

    // ============================================
    // Uninstall
    // ============================================

    pub async fn uninstall_agent(&self, command_id: &str) -> Result<(), String> {
        let current = self.state.read().await.status;

        if current == AgentStatus::NotInstalled {
            self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Not installed").await;
            return Ok(());
        }

        if current == AgentStatus::Running {
            self.report_progress(command_id, CommandExecutionStatus::InProgress, 10, "Stopping before uninstall...").await;
            let _ = self.stop_agent(command_id).await;
        }

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 50, "Uninstalling...").await;
        AgentInstaller::uninstall(true).await?;

        {
            let mut state = self.state.write().await;
            *state = AgentState::default();
        }

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Uninstalled").await;
        Ok(())
    }

    // ============================================
    // Update
    // ============================================

    pub async fn update_agent(
        &self,
        command_id: &str,
        params: &serde_json::Value,
    ) -> Result<(), String> {
        let current = self.state.read().await.status;

        if current == AgentStatus::NotInstalled {
            self.report_progress(command_id, CommandExecutionStatus::Failed, 0, "Not installed").await;
            return Err("Not installed".to_string());
        }

        let was_running = current == AgentStatus::Running;
        self.set_state(AgentStatus::Updating, "Updating...").await;

        if was_running {
            self.report_progress(command_id, CommandExecutionStatus::InProgress, 10, "Stopping for update...").await;
            let _ = self.stop_agent(command_id).await;
        }

        let version = params.get("version").and_then(|v| v.as_str()).unwrap_or("latest");
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 40, "Updating package...").await;

        if let Err(e) = AgentInstaller::install_openclaw_package(version).await {
            let msg = format!("Update failed: {}", e);
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 40, &msg).await;
            return Err(msg);
        }

        // Re-ensure HTTP API after update
        let _ = EnvConfig::ensure_http_api_enabled().await;
        let _ = AgentInstaller::post_install_doctor().await;

        if was_running {
            self.report_progress(command_id, CommandExecutionStatus::InProgress, 80, "Restarting...").await;
            if let Err(e) = GatewayController::start().await {
                let msg = format!("Restart failed: {}", e);
                self.set_state(AgentStatus::Error, &msg).await;
                self.report_progress(command_id, CommandExecutionStatus::Failed, 80, &msg).await;
                return Err(msg);
            }
            GatewayController::wait_for_healthy(self.gateway_port).await;
        }

        let new_version = AgentInstaller::get_installed_version().await;
        {
            let mut state = self.state.write().await;
            state.version = new_version.clone();
            state.status = if was_running { AgentStatus::Running } else { AgentStatus::Stopped };
            state.pid = if was_running { GatewayController::find_pid().await } else { None };
            state.message = format!("Updated to v{}", new_version.as_deref().unwrap_or("unknown"));
        }

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Updated").await;
        Ok(())
    }

    // ============================================
    // Health Check
    // ============================================

    pub async fn periodic_health_check(&self) {
        let current = self.state.read().await.status;
        if current != AgentStatus::Running {
            return;
        }

        if !GatewayController::check_health(self.gateway_port).await {
            warn!("[AGENT] ⚠️ Gateway health check failed");
            let mut state = self.state.write().await;
            state.status = AgentStatus::Error;
            state.pid = None;
            state.message = "Gateway is unresponsive".to_string();
        }
    }

    // ============================================
    // Internal Helpers
    // ============================================

    async fn set_state(&self, status: AgentStatus, message: &str) {
        let mut state = self.state.write().await;
        state.status = status;
        state.message = message.to_string();
    }

    async fn report_progress(
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

        if let Err(e) = self.client.report_command_status(&report).await {
            warn!(command_id = %command_id, error = %e, "[AGENT] ⚠️ Failed to report progress");
        }
    }
}
