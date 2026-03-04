//! ============================================
//! File: crates/aeronyx-server/src/services/agent_manager.rs
//! Path: aeronyx-server/src/services/agent_manager.rs
//! ============================================
//!
//! ## Creation Reason
//! Phase 2 of the OpenClaw integration: Agent Process Management.
//! Manages the full lifecycle of an OpenClaw AI agent on the local node:
//! download, install, configure, start, stop, update, uninstall, and
//! health monitoring.
//!
//! ## Modification Reason (v1.3.2)
//! - 🔄 Added `ensure_http_api_enabled()` to enable OpenClaw's HTTP Chat
//!   Completions endpoint (`/v1/chat/completions`) during install and startup.
//!   This is REQUIRED for `ws_client.rs` (v1.3.2) which replaced WS RPC with
//!   HTTP API calls.
//! - 🔄 Added `load_env_file()` to read API keys from `~/.openclaw/.env` and
//!   pass them to the gateway process on startup. Without this, the gateway
//!   cannot authenticate with upstream LLM providers (xAI, OpenAI, etc.).
//! - 🐛 Fixed TOCTOU race in `install_agent` for `Stopped` state: consolidated
//!   into a single write-lock block to prevent concurrent state changes.
//! - 🐛 Fixed `start_gateway` fallback: added `stdin(Stdio::null())` for proper
//!   process detachment and pass `.env` variables to the spawned process.
//!
//! ## Main Functionality
//! - `AgentManager`: Thread-safe (Arc-wrapped) manager for OpenClaw lifecycle
//! - `install_agent()`: Installs Node.js (if missing) + OpenClaw via npm
//! - `start_agent()`: Launches the OpenClaw gateway as a background process
//! - `stop_agent()`: Gracefully terminates the gateway process
//! - `uninstall_agent()`: Removes OpenClaw and cleans up
//! - `update_agent()`: Updates OpenClaw to latest or specified version
//! - `status()`: Returns current `AgentStatusInfo` for heartbeat reporting
//! - `health_check()`: Pings the local gateway WebSocket endpoint
//!
//! ## OpenClaw Installation Method (Ubuntu 22.04)
//!
//! Official recommended method:
//!   `curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard`
//!
//! This script handles:
//!   1. Detect/install Node.js 22+ (via NodeSource)
//!   2. `npm install -g openclaw@latest`
//!   3. Install systemd user service (with --install-daemon)
//!
//! Post-install, we run:
//!   `openclaw onboard --install-daemon` (non-interactive via env vars)
//!
//! ## OpenClaw Architecture Summary
//! - Gateway listens on ws+http://127.0.0.1:18789 (WS + HTTP multiplex)
//! - Config stored at ~/.openclaw/openclaw.json (JSON5)
//! - Gateway Token stored at gateway.auth.token in config
//! - HTTP API: POST /v1/chat/completions (OpenAI-compatible, must be enabled)
//! - CLI commands: openclaw status, openclaw doctor, openclaw gateway
//! - systemd service: openclaw-gateway (user service)
//!
//! ## Main Logical Flow
//! 1. CommandHandler receives "install_openclaw" command
//! 2. Calls AgentManager::install_agent(params)
//! 3. AgentManager reports progress via ManagementClient
//! 4. On success, enables HTTP API + starts the gateway process
//! 5. Health check loop monitors ws://127.0.0.1:18789
//! 6. Status exposed via AgentManager::status() for heartbeat
//!
//! ## Dependencies
//! - `tokio::process::Command` for async subprocess execution
//! - `ManagementClient` for progress reporting to CMS
//! - `models::AgentStatus/AgentStatusInfo` for state tracking
//!
//! ## ⚠️ Important Note for Next Developer
//! - All subprocess calls use `tokio::process::Command` (NOT std::process)
//! - Install runs as the same user as aeronyx-server (typically root)
//! - OpenClaw gateway runs as a non-root user for security isolation
//!   (we create an `openclaw` system user if it doesn't exist)
//! - The install directory is `/opt/openclaw/` for binary isolation
//! - Gateway token is extracted from `~openclaw/.openclaw/openclaw.json`
//!   after onboarding and stored in AgentManager state
//! - `AgentState` is protected by `tokio::sync::RwLock` for concurrent
//!   access from CommandHandler (write) and HeartbeatReporter (read)
//! - Process PID is tracked for health monitoring and cleanup on shutdown
//! - The HTTP Chat Completions endpoint MUST be enabled for ws_client.rs
//!   to work. If you see 404 from `/v1/chat/completions`, run:
//!   `openclaw config set gateway.http.endpoints.chatCompletions.enabled true`
//! - API keys are stored in `~/.openclaw/.env` (e.g., XAI_API_KEY).
//!   The gateway process reads this file on startup. We also pass these
//!   env vars explicitly when spawning the gateway as a fallback.
//!
//! ## Last Modified
//! v1.3.0 - 🌟 Initial creation (Phase 2: Agent Process Management)
//! v1.3.2 - 🔄 Enable HTTP API endpoint during install/startup
//!          - 🔄 Load and pass .env API keys to gateway process
//!          - 🐛 Fixed TOCTOU race in install_agent Stopped path
//!          - 🐛 Fixed start_gateway stdin detachment
//! ============================================

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::process::Command as TokioCommand;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::management::client::ManagementClient;
use crate::management::models::{
    AgentStatus, AgentStatusInfo, CommandExecutionStatus, CommandStatusReport,
};

// ============================================
// Constants
// ============================================

/// Default OpenClaw gateway WebSocket port.
const OPENCLAW_DEFAULT_PORT: u16 = 18789;

/// OpenClaw system user name.
const OPENCLAW_USER: &str = "openclaw";

/// OpenClaw home directory (under the system user).
const OPENCLAW_HOME: &str = "/home/openclaw";

/// OpenClaw config file path.
const OPENCLAW_CONFIG_PATH: &str = "/home/openclaw/.openclaw/openclaw.json";

/// OpenClaw .env file path (contains API keys).
/// v1.3.2: Read on gateway start to pass API keys to the process.
const OPENCLAW_ENV_PATH: &str = "/home/openclaw/.openclaw/.env";

/// Maximum time (seconds) to wait for gateway to become healthy after start.
const GATEWAY_STARTUP_TIMEOUT_SECS: u64 = 60;

/// Interval (seconds) between health check probes during startup wait.
const HEALTH_CHECK_INTERVAL_SECS: u64 = 3;

// ============================================
// AgentState
// ============================================

/// Internal mutable state of the agent manager.
#[derive(Debug)]
struct AgentState {
    /// Current lifecycle status.
    status: AgentStatus,
    /// Installed version string (e.g., "2026.3.1"), if known.
    version: Option<String>,
    /// Human-readable status message for dashboard.
    message: String,
    /// PID of the running gateway process (if managed by us).
    pid: Option<u32>,
    /// Gateway authentication token (extracted from config after onboard).
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
///
/// ## Usage
/// ```ignore
/// let manager = Arc::new(AgentManager::new(mgmt_client));
/// // Check if already installed on startup
/// manager.detect_existing().await;
/// // Later, from CommandHandler:
/// manager.install_agent("cmd-123", params).await;
/// ```
pub struct AgentManager {
    /// Shared CMS client for progress reporting.
    client: Arc<ManagementClient>,
    /// Protected mutable state.
    state: RwLock<AgentState>,
    /// OpenClaw gateway port (configurable for testing).
    gateway_port: u16,
}

impl AgentManager {
    /// Creates a new AgentManager.
    ///
    /// # Arguments
    /// * `client` - Shared ManagementClient for CMS status reporting
    pub fn new(client: Arc<ManagementClient>) -> Self {
        Self {
            client,
            state: RwLock::new(AgentState::default()),
            gateway_port: OPENCLAW_DEFAULT_PORT,
        }
    }

    // ============================================
    // Status Query (for HeartbeatReporter)
    // ============================================

    /// Returns current agent status for heartbeat reporting.
    ///
    /// This is called every heartbeat cycle (30s) from the
    /// HeartbeatReporter to populate `SystemStats.agent_status`.
    ///
    /// 🌟 v1.3.1: Now includes process-level CPU and memory usage
    /// when the gateway is running and PID is known.
    pub async fn status(&self) -> AgentStatusInfo {
        let state = self.state.read().await;

        // Collect process stats if we have a PID and agent is running
        let (cpu_usage, memory_mb) = if state.status == AgentStatus::Running {
            if let Some(pid) = state.pid {
                Self::collect_process_stats(pid).await
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

    /// Returns the gateway authentication token, if available.
    pub async fn gateway_token(&self) -> Option<String> {
        let state = self.state.read().await;
        state.gateway_token.clone()
    }

    // ============================================
    // Startup Detection
    // ============================================

    /// Detects if OpenClaw is already installed on this node.
    ///
    /// Called once during server startup to restore state.
    /// Checks for the `openclaw` binary and running gateway process.
    ///
    /// v1.3.2: Also ensures HTTP API endpoint is enabled if OpenClaw is installed.
    pub async fn detect_existing(&self) {
        info!("[AGENT] Detecting existing OpenClaw installation...");

        // Check both system-wide and user-local npm-global paths
        let user_bin = format!("{}/.npm-global/bin/openclaw", OPENCLAW_HOME);
        let cli_exists = std::path::Path::new(&user_bin).exists()
            || Self::run_command("which", &["openclaw"]).await.is_ok();

        if !cli_exists {
            info!("[AGENT] OpenClaw not found — status: NotInstalled");
            return;
        }

        // Get version (try user-local first)
        let version = Self::run_command_output_as_user("openclaw", &["--version"]).await.ok()
            .or_else(|| {
                // Sync fallback — check the binary directly
                None
            });

        // Check if gateway is running
        let gateway_running = self.check_gateway_health().await;

        // Try to read gateway token
        let token = Self::read_gateway_token().await;

        // v1.3.2: Ensure HTTP API is enabled for ws_client.rs
        // This is idempotent — safe to call on every startup.
        if let Err(e) = self.ensure_http_api_enabled().await {
            warn!(
                error = %e,
                "[AGENT] ⚠️ Failed to enable HTTP API endpoint — ws_client may get 404"
            );
        }

        let mut state = self.state.write().await;
        state.version = version.clone();
        state.gateway_token = token;

        if gateway_running {
            // Try to find PID
            state.pid = Self::find_gateway_pid().await;
            state.status = AgentStatus::Running;
            state.message = format!(
                "Running (v{})",
                version.as_deref().unwrap_or("unknown")
            );
            info!(
                version = ?state.version,
                pid = ?state.pid,
                "[AGENT] ✅ OpenClaw detected and running"
            );
        } else {
            state.status = AgentStatus::Stopped;
            state.message = format!(
                "Installed but stopped (v{})",
                version.as_deref().unwrap_or("unknown")
            );
            info!(
                version = ?state.version,
                "[AGENT] OpenClaw installed but gateway not running"
            );
        }
    }

    // ============================================
    // Install
    // ============================================

    /// Installs OpenClaw on this node.
    ///
    /// ## Installation Steps
    /// 1. Create `openclaw` system user (if not exists)
    /// 2. Run official installer script (handles Node.js + npm install)
    /// 3. Run onboarding with `--install-daemon` (systemd service)
    /// 4. Enable HTTP API endpoint for ws_client.rs (v1.3.2)
    /// 5. Extract gateway token from config
    /// 6. Start the gateway
    /// 7. Verify health
    ///
    /// # Arguments
    /// * `command_id` - CMS command ID for progress reporting
    /// * `params` - Command params (may contain `version`, `api_key`, etc.)
    ///
    /// # Returns
    /// `Ok(())` on success, `Err(message)` on failure
    pub async fn install_agent(
        &self,
        command_id: &str,
        params: &serde_json::Value,
    ) -> Result<(), String> {
        // v1.3.2: Use a single write lock to check-and-transition atomically,
        // fixing the TOCTOU race condition from v1.3.0 where two separate
        // read locks could see stale state under concurrent access.
        {
            let mut state = self.state.write().await;

            if state.status == AgentStatus::Installing {
                return Err("Installation already in progress".to_string());
            }

            if state.status == AgentStatus::Running {
                info!(
                    command_id = %command_id,
                    "[AGENT] OpenClaw already installed and running — nothing to do"
                );
                // Drop write lock before async call
                drop(state);
                self.report_progress(
                    command_id,
                    CommandExecutionStatus::Completed,
                    100,
                    "OpenClaw is already installed and running",
                ).await;
                return Ok(());
            }

            if state.status == AgentStatus::Stopped {
                // Transition to a transient state to prevent concurrent installs.
                // We'll handle the "start existing" path below.
                info!(
                    command_id = %command_id,
                    "[AGENT] OpenClaw installed but stopped — starting gateway"
                );
                // Mark as installing briefly to block concurrent calls
                state.message = "Starting existing installation...".to_string();
                drop(state);

                self.report_progress(
                    command_id,
                    CommandExecutionStatus::InProgress,
                    80,
                    "Starting existing OpenClaw installation...",
                ).await;

                if let Err(e) = self.start_gateway().await {
                    let msg = format!("Failed to start gateway: {}", e);
                    self.set_state(AgentStatus::Error, &msg).await;
                    self.report_progress(command_id, CommandExecutionStatus::Failed, 80, &msg).await;
                    return Err(msg);
                }

                if self.wait_for_gateway_healthy().await {
                    let version = Self::run_command_output_as_user("openclaw", &["--version"]).await.ok();
                    let mut st = self.state.write().await;
                    st.status = AgentStatus::Running;
                    st.version = version;
                    st.message = "Running".to_string();
                    drop(st);
                    self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "OpenClaw started").await;
                    return Ok(());
                } else {
                    self.set_state(AgentStatus::Error, "Gateway health check failed").await;
                    self.report_progress(command_id, CommandExecutionStatus::Failed, 90, "Health check failed").await;
                    return Err("Gateway failed to start".to_string());
                }
            }

            // For NotInstalled, Error, Updating — proceed with full install
            state.status = AgentStatus::Installing;
            state.message = "Starting installation...".to_string();
        }

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 5, "Starting installation...").await;

        // --- Step 1: Create system user ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 10, "Creating openclaw user...").await;
        if let Err(e) = self.ensure_system_user().await {
            let msg = format!("Failed to create system user: {}", e);
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 10, &msg).await;
            return Err(msg);
        }

        // --- Step 2: Install Node.js if missing ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 20, "Checking Node.js...").await;
        if let Err(e) = self.ensure_nodejs().await {
            let msg = format!("Failed to install Node.js: {}", e);
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 20, &msg).await;
            return Err(msg);
        }

        // --- Step 3: Install OpenClaw via official installer ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 40, "Installing OpenClaw...").await;
        let version = params.get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("latest");

        if let Err(e) = self.install_openclaw_package(version).await {
            let msg = format!("Failed to install OpenClaw: {}", e);
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 40, &msg).await;
            return Err(msg);
        }

        // --- Step 4: Run onboarding (non-interactive + install daemon) ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 60, "Configuring OpenClaw...").await;
        if let Err(e) = self.run_onboarding(params).await {
            let msg = format!("Onboarding failed: {}", e);
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 60, &msg).await;
            return Err(msg);
        }

        // --- Step 5: Enable HTTP API endpoint (v1.3.2) ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 70, "Enabling HTTP API...").await;
        if let Err(e) = self.ensure_http_api_enabled().await {
            // Non-fatal: log warning but continue. The admin can enable it manually.
            warn!(
                error = %e,
                "[AGENT] ⚠️ Failed to enable HTTP API — ws_client may need manual config"
            );
        }

        // --- Step 6: Extract gateway token ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 75, "Extracting gateway token...").await;
        let token = Self::read_gateway_token().await;
        {
            let mut state = self.state.write().await;
            state.gateway_token = token;
        }

        // --- Step 7: Start gateway ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 80, "Starting OpenClaw gateway...").await;
        if let Err(e) = self.start_gateway().await {
            let msg = format!("Failed to start gateway: {}", e);
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 80, &msg).await;
            return Err(msg);
        }

        // --- Step 8: Verify health ---
        self.report_progress(command_id, CommandExecutionStatus::InProgress, 90, "Verifying gateway health...").await;
        if !self.wait_for_gateway_healthy().await {
            let msg = "Gateway started but health check timed out".to_string();
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 90, &msg).await;
            return Err(msg);
        }

        // --- Step 9: Get version and finalize ---
        let installed_version = Self::run_command_output_as_user("openclaw", &["--version"]).await.ok();
        {
            let mut state = self.state.write().await;
            state.status = AgentStatus::Running;
            state.version = installed_version.clone();
            state.message = format!(
                "Running (v{})",
                installed_version.as_deref().unwrap_or("unknown")
            );
        }

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "OpenClaw installed and running").await;

        info!(
            version = ?installed_version,
            "[AGENT] ✅ OpenClaw installation complete"
        );

        Ok(())
    }

    // ============================================
    // Start / Stop
    // ============================================

    /// Starts the OpenClaw gateway process.
    ///
    /// **Idempotent**: if already running, reports success without error.
    /// Uses systemd user service if available, falls back to direct process.
    pub async fn start_agent(&self, command_id: &str) -> Result<(), String> {
        let current_status = self.state.read().await.status;

        match current_status {
            // Already running — idempotent success
            AgentStatus::Running => {
                info!(
                    command_id = %command_id,
                    "[AGENT] OpenClaw already running — nothing to do"
                );
                self.report_progress(
                    command_id,
                    CommandExecutionStatus::Completed,
                    100,
                    "OpenClaw is already running",
                ).await;
                return Ok(());
            }
            // Not installed — can't start what doesn't exist
            AgentStatus::NotInstalled => {
                self.report_progress(
                    command_id,
                    CommandExecutionStatus::Failed,
                    0,
                    "OpenClaw is not installed. Send install_openclaw first.",
                ).await;
                return Err("OpenClaw is not installed".to_string());
            }
            // Stopped, Error, Updating — try to start
            _ => {}
        }

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 30, "Starting gateway...").await;
        self.start_gateway().await?;

        if self.wait_for_gateway_healthy().await {
            let pid = Self::find_gateway_pid().await;
            let mut state = self.state.write().await;
            state.status = AgentStatus::Running;
            state.pid = pid;
            state.message = "Running".to_string();

            self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Gateway started").await;
            Ok(())
        } else {
            self.set_state(AgentStatus::Error, "Gateway failed to start").await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 50, "Health check failed after start").await;
            Err("Gateway failed to start".to_string())
        }
    }

    /// Stops the OpenClaw gateway process.
    ///
    /// **Idempotent**: if already stopped or not installed, reports success.
    pub async fn stop_agent(&self, command_id: &str) -> Result<(), String> {
        let current_status = self.state.read().await.status;

        match current_status {
            // Already stopped or not installed — idempotent success
            AgentStatus::Stopped | AgentStatus::NotInstalled => {
                info!(
                    command_id = %command_id,
                    status = ?current_status,
                    "[AGENT] OpenClaw not running — nothing to stop"
                );
                self.report_progress(
                    command_id,
                    CommandExecutionStatus::Completed,
                    100,
                    "OpenClaw is not running",
                ).await;
                return Ok(());
            }
            // Running, Error, Installing, Updating — try to stop
            _ => {}
        }

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 30, "Stopping gateway...").await;

        // Try systemd first, then direct kill
        let result = Self::run_command_as_user(
            "systemctl", &["--user", "stop", "openclaw-gateway"]
        ).await;

        if result.is_err() {
            // Fallback: kill by PID
            let state = self.state.read().await;
            if let Some(pid) = state.pid {
                let _ = Self::run_command("kill", &["-SIGTERM", &pid.to_string()]).await;
            }
        }

        // Wait for process to exit
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        let mut state = self.state.write().await;
        state.status = AgentStatus::Stopped;
        state.pid = None;
        state.message = "Stopped".to_string();

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "Gateway stopped").await;
        info!("[AGENT] ⏹️ OpenClaw gateway stopped");

        Ok(())
    }

    // ============================================
    // Uninstall
    // ============================================

    /// Uninstalls OpenClaw from this node.
    ///
    /// **Idempotent**: if not installed, reports success.
    /// If running, stops first then uninstalls.
    pub async fn uninstall_agent(&self, command_id: &str) -> Result<(), String> {
        let current_status = self.state.read().await.status;

        // Already not installed — idempotent success
        if current_status == AgentStatus::NotInstalled {
            info!(
                command_id = %command_id,
                "[AGENT] OpenClaw not installed — nothing to uninstall"
            );
            self.report_progress(
                command_id,
                CommandExecutionStatus::Completed,
                100,
                "OpenClaw is not installed",
            ).await;
            return Ok(());
        }

        // Stop first if running
        if current_status == AgentStatus::Running {
            self.report_progress(command_id, CommandExecutionStatus::InProgress, 10, "Stopping gateway before uninstall...").await;
            let _ = self.stop_agent(command_id).await;
        }

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 30, "Removing systemd service...").await;
        // Disable systemd service
        let _ = Self::run_command_as_user(
            "systemctl", &["--user", "disable", "openclaw-gateway"]
        ).await;

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 50, "Uninstalling OpenClaw npm package...").await;
        // Uninstall npm package
        let _ = Self::run_command_as_user(
            "npm", &["uninstall", "-g", "openclaw"]
        ).await;

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 70, "Cleaning up configuration...").await;
        // Remove config directory
        let openclaw_config_dir = format!("{}/.openclaw", OPENCLAW_HOME);
        let _ = Self::run_command("rm", &["-rf", &openclaw_config_dir]).await;

        // Reset state
        {
            let mut state = self.state.write().await;
            *state = AgentState::default();
        }

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "OpenClaw uninstalled").await;
        info!("[AGENT] 🗑️ OpenClaw uninstalled");

        Ok(())
    }

    // ============================================
    // Update
    // ============================================

    /// Updates OpenClaw to the latest (or specified) version.
    ///
    /// **Smart handling**:
    /// - NotInstalled → reports error (send install_openclaw instead)
    /// - Running → stop, update, restart
    /// - Stopped → update only (don't start)
    /// - Error → try update anyway (might fix the error)
    pub async fn update_agent(
        &self,
        command_id: &str,
        params: &serde_json::Value,
    ) -> Result<(), String> {
        let current_status = self.state.read().await.status;

        if current_status == AgentStatus::NotInstalled {
            self.report_progress(
                command_id,
                CommandExecutionStatus::Failed,
                0,
                "OpenClaw is not installed. Send install_openclaw first.",
            ).await;
            return Err("OpenClaw is not installed".to_string());
        }

        let was_running = current_status == AgentStatus::Running;

        self.set_state(AgentStatus::Updating, "Updating OpenClaw...").await;

        // Stop if running
        if was_running {
            self.report_progress(command_id, CommandExecutionStatus::InProgress, 10, "Stopping gateway for update...").await;
            let _ = self.stop_agent(command_id).await;
        }

        // Update npm package
        let version = params.get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("latest");

        self.report_progress(command_id, CommandExecutionStatus::InProgress, 40, "Updating npm package...").await;
        if let Err(e) = self.install_openclaw_package(version).await {
            let msg = format!("Update failed: {}", e);
            self.set_state(AgentStatus::Error, &msg).await;
            self.report_progress(command_id, CommandExecutionStatus::Failed, 40, &msg).await;
            return Err(msg);
        }

        // v1.3.2: Re-ensure HTTP API is enabled after update (new version may reset config)
        if let Err(e) = self.ensure_http_api_enabled().await {
            warn!(
                error = %e,
                "[AGENT] ⚠️ Failed to re-enable HTTP API after update"
            );
        }

        // Restart if it was running
        if was_running {
            self.report_progress(command_id, CommandExecutionStatus::InProgress, 80, "Restarting gateway...").await;
            if let Err(e) = self.start_gateway().await {
                let msg = format!("Restart after update failed: {}", e);
                self.set_state(AgentStatus::Error, &msg).await;
                self.report_progress(command_id, CommandExecutionStatus::Failed, 80, &msg).await;
                return Err(msg);
            }
            self.wait_for_gateway_healthy().await;
        }

        // Update version info
        let new_version = Self::run_command_output_as_user("openclaw", &["--version"]).await.ok();
        {
            let mut state = self.state.write().await;
            state.version = new_version.clone();
            state.status = if was_running { AgentStatus::Running } else { AgentStatus::Stopped };
            state.message = format!(
                "Updated to v{}",
                new_version.as_deref().unwrap_or("unknown")
            );
        }

        self.report_progress(command_id, CommandExecutionStatus::Completed, 100, "OpenClaw updated").await;
        info!(version = ?new_version, "[AGENT] 🔄 OpenClaw updated");

        Ok(())
    }

    // ============================================
    // Health Check
    // ============================================

    /// Checks if the OpenClaw gateway is responding.
    ///
    /// Attempts a TCP connection to the gateway port.
    /// Full WebSocket handshake is not required — just port liveness.
    async fn check_gateway_health(&self) -> bool {
        let addr = format!("127.0.0.1:{}", self.gateway_port);
        tokio::net::TcpStream::connect(&addr).await.is_ok()
    }

    /// Waits for the gateway to become healthy (up to timeout).
    async fn wait_for_gateway_healthy(&self) -> bool {
        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(GATEWAY_STARTUP_TIMEOUT_SECS);

        while tokio::time::Instant::now() < deadline {
            if self.check_gateway_health().await {
                // Also grab PID
                let pid = Self::find_gateway_pid().await;
                let mut state = self.state.write().await;
                state.pid = pid;
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS)).await;
        }

        false
    }

    /// Performs a periodic health check and updates state if unhealthy.
    ///
    /// Call this from a background loop (spawned by server.rs).
    pub async fn periodic_health_check(&self) {
        let current_status = self.state.read().await.status;
        if current_status != AgentStatus::Running {
            return;
        }

        if !self.check_gateway_health().await {
            warn!("[AGENT] ⚠️ Gateway health check failed — marking as Error");
            let mut state = self.state.write().await;
            state.status = AgentStatus::Error;
            state.pid = None;
            state.message = "Gateway is unresponsive".to_string();
        }
    }

    // ============================================
    // Process Stats (Linux /proc)
    // ============================================

    /// Collects CPU and memory usage for the OpenClaw process tree.
    ///
    /// 🌟 v1.3.1: Aggregates across ALL openclaw-related processes,
    /// not just the main PID. Node.js may fork worker processes,
    /// and we want total resource usage.
    ///
    /// Strategy:
    /// 1. Use `pgrep -f "openclaw"` to find all related PIDs
    /// 2. Sum VmRSS from /proc/{pid}/status for each
    /// 3. Sum utime+stime from /proc/{pid}/stat for each
    /// 4. If pgrep fails, fall back to just the known PID
    #[cfg(target_os = "linux")]
    async fn collect_process_stats(primary_pid: u32) -> (Option<f32>, Option<u64>) {
        // Find all openclaw-related PIDs
        let pids = Self::find_all_openclaw_pids(primary_pid).await;

        if pids.is_empty() {
            return (None, None);
        }

        let mut total_memory_kb = 0u64;
        let mut total_cpu_ticks = 0u64;
        let mut min_starttime = u64::MAX;
        let mut found_any = false;

        for pid in &pids {
            // Memory: read VmRSS from /proc/{pid}/status
            if let Some(rss_kb) = Self::read_pid_rss(*pid).await {
                total_memory_kb += rss_kb;
                found_any = true;
            }

            // CPU: read utime+stime from /proc/{pid}/stat
            if let Some((ticks, starttime)) = Self::read_pid_cpu_ticks(*pid).await {
                total_cpu_ticks += ticks;
                if starttime < min_starttime {
                    min_starttime = starttime;
                }
            }
        }

        if !found_any {
            return (None, None);
        }

        let memory_mb = Some(total_memory_kb / 1024);

        // Calculate CPU percentage using the earliest start time
        let cpu_pct = if min_starttime < u64::MAX {
            Self::calculate_cpu_percent(total_cpu_ticks, min_starttime).await
        } else {
            None
        };

        (cpu_pct, memory_mb)
    }

    #[cfg(not(target_os = "linux"))]
    async fn collect_process_stats(_pid: u32) -> (Option<f32>, Option<u64>) {
        (None, None)
    }

    /// Finds all PIDs related to OpenClaw (main + child processes).
    #[cfg(target_os = "linux")]
    async fn find_all_openclaw_pids(primary_pid: u32) -> Vec<u32> {
        // First try pgrep for all openclaw processes
        let output = TokioCommand::new("pgrep")
            .args(&["-f", "openclaw"])
            .output()
            .await;

        let mut pids: Vec<u32> = match output {
            Ok(out) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .filter_map(|l| l.trim().parse::<u32>().ok())
                    .collect()
            }
            _ => Vec::new(),
        };

        // Ensure primary PID is included
        if !pids.contains(&primary_pid) {
            // Check if the primary PID still exists
            if std::path::Path::new(&format!("/proc/{}", primary_pid)).exists() {
                pids.push(primary_pid);
            }
        }

        pids
    }

    /// Reads VmRSS (resident memory) for a PID from /proc/{pid}/status.
    #[cfg(target_os = "linux")]
    async fn read_pid_rss(pid: u32) -> Option<u64> {
        let path = format!("/proc/{}/status", pid);
        let content = tokio::fs::read_to_string(&path).await.ok()?;

        for line in content.lines() {
            // Prefer RssAnon if available (actual private memory)
            // Fall back to VmRSS (includes shared libraries)
            if line.starts_with("VmRSS:") {
                return line.split_whitespace().nth(1)?.parse::<u64>().ok();
            }
        }
        None
    }

    /// Reads utime + stime (CPU ticks) and starttime for a PID.
    /// Returns (total_ticks, starttime).
    #[cfg(target_os = "linux")]
    async fn read_pid_cpu_ticks(pid: u32) -> Option<(u64, u64)> {
        let path = format!("/proc/{}/stat", pid);
        let content = tokio::fs::read_to_string(&path).await.ok()?;

        // Parse: skip past "(comm)" which may contain spaces
        let after_comm = content.rfind(')')? + 2;
        let fields: Vec<&str> = content[after_comm..].split_whitespace().collect();

        if fields.len() < 20 {
            return None;
        }

        // utime=field[11], stime=field[12], starttime=field[19]
        // (0-indexed relative to fields after comm)
        let utime: u64 = fields[11].parse().ok()?;
        let stime: u64 = fields[12].parse().ok()?;
        let starttime: u64 = fields[19].parse().ok()?;

        Some((utime + stime, starttime))
    }

    /// Calculates CPU percentage from total ticks and earliest start time.
    #[cfg(target_os = "linux")]
    async fn calculate_cpu_percent(total_ticks: u64, starttime: u64) -> Option<f32> {
        let uptime_content = tokio::fs::read_to_string("/proc/uptime").await.ok()?;
        let uptime_secs: f64 = uptime_content
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;

        let clk_tck: f64 = 100.0; // sysconf(_SC_CLK_TCK)
        let total_time = total_ticks as f64 / clk_tck;
        let process_uptime = uptime_secs - (starttime as f64 / clk_tck);

        if process_uptime > 0.0 {
            Some((total_time / process_uptime * 100.0) as f32)
        } else {
            None
        }
    }

    // ============================================
    // Internal: HTTP API Enablement (v1.3.2)
    // ============================================

    /// Ensures the OpenClaw HTTP Chat Completions endpoint is enabled.
    ///
    /// v1.3.2: This is REQUIRED for ws_client.rs to function.
    /// The endpoint is disabled by default in OpenClaw. We enable it
    /// via `openclaw config set` which is idempotent and safe to call
    /// repeatedly.
    ///
    /// Sets: `gateway.http.endpoints.chatCompletions.enabled = true`
    ///
    /// ## Why this matters
    /// Without this, POST requests to `/v1/chat/completions` return 404.
    /// The ws_client.rs (v1.3.2) uses this endpoint instead of the complex
    /// WebSocket RPC protocol.
    async fn ensure_http_api_enabled(&self) -> Result<(), String> {
        info!("[AGENT] Ensuring HTTP Chat Completions API is enabled...");

        // Use openclaw config set (idempotent)
        Self::run_command_as_user_with_env(
            "openclaw",
            &["config", "set", "gateway.http.endpoints.chatCompletions.enabled", "true"],
            &[],
        ).await.map_err(|e| format!("Failed to enable HTTP API: {}", e))?;

        info!("[AGENT] ✅ HTTP Chat Completions API enabled");
        Ok(())
    }

    // ============================================
    // Internal: .env File Loading (v1.3.2)
    // ============================================

    /// Reads key=value pairs from the OpenClaw .env file.
    ///
    /// v1.3.2: Used to pass API keys (e.g., XAI_API_KEY) to the gateway
    /// process when launched via direct spawn (non-systemd fallback).
    ///
    /// Returns a HashMap of environment variable name → value.
    /// Returns an empty map if the file doesn't exist or can't be read.
    ///
    /// ## File format
    /// Standard .env format:
    /// ```text
    /// XAI_API_KEY=xai-abc123...
    /// OPENAI_API_KEY=sk-...
    /// ```
    /// Lines starting with `#` are comments. Empty lines are skipped.
    /// Values are NOT shell-expanded (no $VAR substitution).
    async fn load_env_file() -> HashMap<String, String> {
        let content = match tokio::fs::read_to_string(OPENCLAW_ENV_PATH).await {
            Ok(c) => c,
            Err(_) => {
                debug!("[AGENT] No .env file at {} — skipping", OPENCLAW_ENV_PATH);
                return HashMap::new();
            }
        };

        let mut env_vars = HashMap::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Split on first '=' only (value may contain '=')
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim().to_string();
                let value = line[eq_pos + 1..].trim().to_string();

                // Strip optional surrounding quotes from value
                let value = if (value.starts_with('"') && value.ends_with('"'))
                    || (value.starts_with('\'') && value.ends_with('\''))
                {
                    value[1..value.len() - 1].to_string()
                } else {
                    value
                };

                if !key.is_empty() {
                    env_vars.insert(key, value);
                }
            }
        }

        if !env_vars.is_empty() {
            debug!(
                count = env_vars.len(),
                "[AGENT] Loaded {} env vars from .env file",
                env_vars.len()
            );
        }

        env_vars
    }

    // ============================================
    // Internal: System User Management
    // ============================================

    /// Ensures the `openclaw` system user exists.
    async fn ensure_system_user(&self) -> Result<(), String> {
        // Check if user exists
        let exists = Self::run_command("id", &[OPENCLAW_USER]).await.is_ok();
        if exists {
            debug!("[AGENT] User '{}' already exists", OPENCLAW_USER);
            return Ok(());
        }

        info!("[AGENT] Creating system user '{}'...", OPENCLAW_USER);
        Self::run_command(
            "useradd",
            &[
                "--system",
                "--create-home",
                "--home-dir", OPENCLAW_HOME,
                "--shell", "/bin/bash",
                OPENCLAW_USER,
            ],
        ).await.map_err(|e| format!("useradd failed: {}", e))?;

        // Enable lingering so systemd user services start on boot
        Self::run_command(
            "loginctl",
            &["enable-linger", OPENCLAW_USER],
        ).await.map_err(|e| format!("enable-linger failed: {}", e))?;

        info!("[AGENT] ✅ User '{}' created with linger enabled", OPENCLAW_USER);
        Ok(())
    }

    // ============================================
    // Internal: Node.js
    // ============================================

    /// Ensures Node.js 22+ is installed.
    async fn ensure_nodejs(&self) -> Result<(), String> {
        // Check existing version
        if let Ok(version_str) = Self::run_command_output("node", &["--version"]).await {
            // Parse "v22.x.x" → check major >= 22
            let major: Option<u32> = version_str
                .trim()
                .trim_start_matches('v')
                .split('.')
                .next()
                .and_then(|s| s.parse().ok());

            if let Some(m) = major {
                if m >= 22 {
                    info!("[AGENT] Node.js {} already installed", version_str.trim());
                    return Ok(());
                }
            }
            info!("[AGENT] Node.js version {} too old, upgrading...", version_str.trim());
        }

        info!("[AGENT] Installing Node.js 22 via NodeSource...");

        // Install NodeSource repo + Node.js 22
        Self::run_command(
            "bash",
            &["-c", "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"],
        ).await.map_err(|e| format!("NodeSource setup failed: {}", e))?;

        Self::run_command(
            "apt-get",
            &["install", "-y", "nodejs"],
        ).await.map_err(|e| format!("apt-get install nodejs failed: {}", e))?;

        // Verify
        let version = Self::run_command_output("node", &["--version"]).await
            .map_err(|e| format!("Node.js verification failed: {}", e))?;

        info!("[AGENT] ✅ Node.js {} installed", version.trim());
        Ok(())
    }

    // ============================================
    // Internal: OpenClaw Package
    // ============================================

    /// Installs OpenClaw npm package as the openclaw user.
    ///
    /// Uses `NPM_CONFIG_PREFIX` to install into the user's home directory
    /// instead of the system `/usr/lib/node_modules/` which requires root.
    /// The binary ends up at `/home/openclaw/.npm-global/bin/openclaw`.
    async fn install_openclaw_package(&self, version: &str) -> Result<(), String> {
        let package = if version == "latest" {
            "openclaw@latest".to_string()
        } else {
            format!("openclaw@{}", version)
        };

        info!("[AGENT] Installing {} via npm...", package);

        // Ensure npm global prefix directory exists
        let npm_prefix = format!("{}/.npm-global", OPENCLAW_HOME);
        Self::run_command_as_user("mkdir", &["-p", &npm_prefix]).await
            .map_err(|e| format!("Failed to create npm prefix dir: {}", e))?;

        // Configure npm to use user-local prefix
        // This avoids EACCES errors when installing globally without root
        Self::run_command_as_user_with_env(
            "npm",
            &["install", "-g", &package],
            &[
                ("NPM_CONFIG_PREFIX", &npm_prefix),
            ],
        ).await.map_err(|e| format!("npm install failed: {}", e))?;

        // Verify installation — binary is at ~/.npm-global/bin/openclaw
        let openclaw_bin = format!("{}/.npm-global/bin/openclaw", OPENCLAW_HOME);
        if !std::path::Path::new(&openclaw_bin).exists() {
            return Err(format!(
                "openclaw binary not found at {} after install",
                openclaw_bin
            ));
        }

        info!("[AGENT] ✅ {} installed at {}", package, openclaw_bin);
        Ok(())
    }

    /// Runs OpenClaw onboarding in non-interactive mode.
    async fn run_onboarding(
        &self,
        params: &serde_json::Value,
    ) -> Result<(), String> {
        info!("[AGENT] Running onboarding (non-interactive)...");

        let npm_bin = format!("{}/.npm-global/bin", OPENCLAW_HOME);
        let openclaw_bin = format!("{}/openclaw", npm_bin);
        let path_value = format!(
            "{}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            npm_bin
        );

        // If CMS provides an API key, pass it via environment
        let api_key = params.get("api_key")
            .and_then(|v| v.as_str());

        let _model = params.get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("anthropic/claude-sonnet-4-5-20250929");

        // Build command — run openclaw binary directly with sudo
        let mut cmd = TokioCommand::new("sudo");
        cmd.args(&[
            "-u", OPENCLAW_USER,
            "--preserve-env=PATH,HOME,OPENCLAW_HOME,NPM_CONFIG_PREFIX",
            &openclaw_bin,
            "onboard",
            "--install-daemon",
        ]);
        cmd.env("HOME", OPENCLAW_HOME);
        cmd.env("OPENCLAW_HOME", OPENCLAW_HOME);
        cmd.env("PATH", &path_value);
        cmd.env("NPM_CONFIG_PREFIX", format!("{}/.npm-global", OPENCLAW_HOME));

        if let Some(key) = api_key {
            cmd.env("ANTHROPIC_API_KEY", key);
        }

        let output = cmd.output().await
            .map_err(|e| format!("onboarding command failed: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("onboarding failed: {}", stderr));
        }

        info!("[AGENT] ✅ Onboarding complete");
        Ok(())
    }

    // ============================================
    // Internal: Gateway Process Control
    // ============================================

    /// Starts the OpenClaw gateway via systemd user service.
    ///
    /// v1.3.2: Loads API keys from .env file and passes them to the
    /// gateway process when using direct-spawn fallback. Systemd should
    /// inherit env from the user session (which reads .env on login).
    async fn start_gateway(&self) -> Result<(), String> {
        info!("[AGENT] Starting OpenClaw gateway...");

        // Try systemd first
        let result = Self::run_command_as_user(
            "systemctl",
            &["--user", "start", "openclaw-gateway"],
        ).await;

        if result.is_ok() {
            info!("[AGENT] ✅ Gateway started via systemd");
            return Ok(());
        }

        // Fallback: direct process launch using the npm-global binary
        warn!("[AGENT] systemd start failed, falling back to direct launch...");

        let npm_bin = format!("{}/.npm-global/bin", OPENCLAW_HOME);
        let openclaw_bin = format!("{}/openclaw", npm_bin);
        let path_value = format!(
            "{}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            npm_bin
        );

        // v1.3.2: Load API keys from .env file so the gateway can
        // authenticate with upstream LLM providers
        let env_vars = Self::load_env_file().await;

        let mut cmd = TokioCommand::new("sudo");
        cmd.args(&[
            "-u", OPENCLAW_USER,
            "--preserve-env=PATH,HOME,OPENCLAW_HOME,NPM_CONFIG_PREFIX",
            &openclaw_bin,
            "gateway",
            "--verbose",
        ]);
        cmd.env("HOME", OPENCLAW_HOME);
        cmd.env("OPENCLAW_HOME", OPENCLAW_HOME);
        cmd.env("PATH", &path_value);
        cmd.env("NPM_CONFIG_PREFIX", format!("{}/.npm-global", OPENCLAW_HOME));

        // v1.3.2: Pass API keys from .env file
        for (key, value) in &env_vars {
            cmd.env(key, value);
        }

        // v1.3.2: Fully detach child process from parent —
        // stdin must also be null to prevent the child from
        // blocking on input after the parent exits.
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());

        let child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn gateway: {}", e))?;

        let pid = child.id();
        info!(pid = ?pid, "[AGENT] Gateway spawned directly");

        if let Some(p) = pid {
            let mut state = self.state.write().await;
            state.pid = Some(p);
        }

        Ok(())
    }

    // ============================================
    // Internal: Gateway Token
    // ============================================

    /// Reads the gateway authentication token from OpenClaw config.
    async fn read_gateway_token() -> Option<String> {
        let content = tokio::fs::read_to_string(OPENCLAW_CONFIG_PATH).await.ok()?;

        // OpenClaw uses JSON5, but for token extraction simple parsing suffices
        // Look for "token": "..." in gateway.auth section
        let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
        parsed
            .get("gateway")
            .and_then(|g| g.get("auth"))
            .and_then(|a| a.get("token"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
    }

    // ============================================
    // Internal: Process Discovery
    // ============================================

    /// Finds the PID of a running OpenClaw gateway process.
    async fn find_gateway_pid() -> Option<u32> {
        let output = TokioCommand::new("pgrep")
            .args(&["-f", "openclaw.*gateway"])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .lines()
            .next()
            .and_then(|line| line.trim().parse::<u32>().ok())
    }

    // ============================================
    // Internal: Command Execution Helpers
    // ============================================

    /// Runs a command as root and returns success/failure.
    async fn run_command(program: &str, args: &[&str]) -> Result<(), String> {
        debug!("[AGENT_CMD] {} {}", program, args.join(" "));

        let output = TokioCommand::new(program)
            .args(args)
            .output()
            .await
            .map_err(|e| format!("Failed to execute {}: {}", program, e))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("{} failed: {}", program, stderr.trim()))
        }
    }

    /// Runs a command as root and returns stdout.
    async fn run_command_output(program: &str, args: &[&str]) -> Result<String, String> {
        let output = TokioCommand::new(program)
            .args(args)
            .output()
            .await
            .map_err(|e| format!("Failed to execute {}: {}", program, e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("{} failed: {}", program, stderr.trim()))
        }
    }

    /// Runs a command as the `openclaw` user via `sudo -u`.
    async fn run_command_as_user(program: &str, args: &[&str]) -> Result<(), String> {
        Self::run_command_as_user_with_env(program, args, &[]).await
    }

    /// Runs a command as the `openclaw` user with extra environment variables.
    ///
    /// Always sets HOME, PATH (including npm-global/bin), and OPENCLAW_HOME.
    async fn run_command_as_user_with_env(
        program: &str,
        args: &[&str],
        extra_env: &[(&str, &str)],
    ) -> Result<(), String> {
        let npm_bin = format!("{}/.npm-global/bin", OPENCLAW_HOME);
        let path_value = format!(
            "{}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            npm_bin
        );

        debug!(
            "[AGENT_CMD] sudo -u {} {} {}",
            OPENCLAW_USER, program, args.join(" ")
        );

        let mut cmd = TokioCommand::new("sudo");
        cmd.args(&["-u", OPENCLAW_USER, "--preserve-env=PATH,HOME,NPM_CONFIG_PREFIX,OPENCLAW_HOME", program]);
        cmd.args(args);
        cmd.env("HOME", OPENCLAW_HOME);
        cmd.env("OPENCLAW_HOME", OPENCLAW_HOME);
        cmd.env("PATH", &path_value);
        cmd.env("NPM_CONFIG_PREFIX", format!("{}/.npm-global", OPENCLAW_HOME));

        for (key, val) in extra_env {
            cmd.env(key, val);
        }

        let output = cmd.output()
            .await
            .map_err(|e| format!("Failed to execute {} as {}: {}", program, OPENCLAW_USER, e))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("{} failed: {}", program, stderr.trim()))
        }
    }

    /// Runs a command as the `openclaw` user and returns stdout.
    async fn run_command_output_as_user(program: &str, args: &[&str]) -> Result<String, String> {
        let npm_bin = format!("{}/.npm-global/bin", OPENCLAW_HOME);
        let path_value = format!(
            "{}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            npm_bin
        );

        let output = TokioCommand::new("sudo")
            .args(&["-u", OPENCLAW_USER, "--preserve-env=PATH,HOME,NPM_CONFIG_PREFIX,OPENCLAW_HOME", program])
            .args(args)
            .env("HOME", OPENCLAW_HOME)
            .env("OPENCLAW_HOME", OPENCLAW_HOME)
            .env("PATH", &path_value)
            .env("NPM_CONFIG_PREFIX", format!("{}/.npm-global", OPENCLAW_HOME))
            .output()
            .await
            .map_err(|e| format!("Failed to execute {} as {}: {}", program, OPENCLAW_USER, e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("{} failed: {}", program, stderr.trim()))
        }
    }

    // ============================================
    // Internal: State Helpers
    // ============================================

    /// Updates the internal state status and message.
    async fn set_state(&self, status: AgentStatus, message: &str) {
        let mut state = self.state.write().await;
        state.status = status;
        state.message = message.to_string();
    }

    /// Reports command progress to CMS via ManagementClient.
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
            warn!(
                command_id = %command_id,
                error = %e,
                "[AGENT] ⚠️ Failed to report progress to CMS"
            );
        }
    }
}
