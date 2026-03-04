//! ============================================
//! File: crates/aeronyx-server/src/services/agent/gateway.rs
//! Path: aeronyx-server/src/services/agent/gateway.rs
//! ============================================
//!
//! ## Creation Reason
//! Extracted from `agent_manager.rs` to isolate gateway process lifecycle
//! management: start, stop, restart, health check, and PID tracking.
//!
//! ## Main Functionality
//! - `GatewayController`: Gateway process lifecycle operations
//! - `start()`: Starts gateway via systemd or direct spawn fallback
//! - `stop()`: Gracefully stops the gateway process
//! - `restart()`: Stop + start with health verification
//! - `check_health()`: TCP liveness check on gateway port
//! - `wait_for_healthy()`: Polls health check until ready or timeout
//! - `find_pid()`: Discovers the running gateway PID via pgrep
//! - `read_token()`: Extracts gateway auth token from config
//!
//! ## Gateway Startup Strategy
//! 1. **Try systemd first**: `systemctl --user start openclaw-gateway`
//!    - Pros: auto-restart on crash, proper logging, clean shutdown
//!    - Fails if: systemd user services unavailable (containers)
//! 2. **Fallback to direct spawn**: `openclaw gateway --verbose`
//!    - Stdin/stdout/stderr redirected to null for full detachment
//!    - API keys loaded from .env file and passed as env vars
//!    - Less robust: no auto-restart on crash
//!
//! ## ⚠️ Important Note for Next Developer
//! - The gateway process runs as the `openclaw` user, NOT root
//! - Direct spawn uses `sudo -u openclaw` with explicit env vars
//! - stdin/stdout/stderr MUST all be null for proper process detachment
//!   (missing stdin caused hangs in v1.3.0)
//! - The health check is TCP-only (port liveness), not a full WS handshake
//! - Gateway startup can take 5-15 seconds depending on system load
//! - `GATEWAY_STARTUP_TIMEOUT_SECS` (60s) should NOT be reduced — slow
//!   VMs (1 vCPU) legitimately need this long
//!
//! ## Last Modified
//! v1.4.0 - 🌟 Initial creation (extracted from agent_manager.rs)
//! ============================================

use std::collections::HashMap;
use tokio::process::Command as TokioCommand;
use tracing::{debug, info, warn};

use super::{
    CommandRunner, OPENCLAW_HOME, OPENCLAW_USER, OPENCLAW_BIN, OPENCLAW_CONFIG_PATH,
    OPENCLAW_DEFAULT_PORT, NPM_PREFIX, openclaw_path,
    env_config::EnvConfig,
};

// ============================================
// Constants
// ============================================

/// Maximum time (seconds) to wait for gateway to become healthy after start.
const GATEWAY_STARTUP_TIMEOUT_SECS: u64 = 60;

/// Interval (seconds) between health check probes during startup wait.
const HEALTH_CHECK_INTERVAL_SECS: u64 = 3;

// ============================================
// GatewayController
// ============================================

/// Controls the OpenClaw gateway process lifecycle.
///
/// All methods are stateless — state is tracked by `AgentManager`.
/// The gateway port is configurable for testing (default: 18789).
pub struct GatewayController;

impl GatewayController {
    // ============================================
    // Start
    // ============================================

    /// Starts the OpenClaw gateway process.
    ///
    /// Tries systemd first, falls back to direct spawn.
    /// Returns the PID of the started process (if available from direct spawn).
    ///
    /// ## Error Conditions
    /// - Both systemd and direct spawn fail
    /// - OpenClaw binary not found at OPENCLAW_BIN
    pub async fn start() -> Result<Option<u32>, String> {
        info!("[GATEWAY] Starting OpenClaw gateway...");

        // Pre-check: binary must exist
        if !std::path::Path::new(OPENCLAW_BIN).exists() {
            return Err(format!(
                "OpenClaw binary not found at {}. Is OpenClaw installed?",
                OPENCLAW_BIN
            ));
        }

        // Strategy 1: systemd user service
        let systemd_result = CommandRunner::run_as_user(
            "systemctl",
            &["--user", "start", "openclaw-gateway"],
        ).await;

        if systemd_result.is_ok() {
            info!("[GATEWAY] ✅ Started via systemd");
            // PID will be discovered via find_pid() later
            return Ok(None);
        }

        // Strategy 2: direct process spawn
        warn!("[GATEWAY] systemd start failed, falling back to direct spawn...");
        Self::spawn_direct().await
    }

    /// Spawns the gateway as a direct background process.
    ///
    /// Loads API keys from .env and passes them as environment variables.
    /// Fully detaches the child process (stdin/stdout/stderr = null).
    async fn spawn_direct() -> Result<Option<u32>, String> {
        let path_value = openclaw_path();

        // Load API keys from .env file
        let env_vars = EnvConfig::load_env_file().await;

        let mut cmd = TokioCommand::new("sudo");
        cmd.args(&[
            "-u", OPENCLAW_USER,
            "--preserve-env=PATH,HOME,OPENCLAW_HOME,NPM_CONFIG_PREFIX",
            OPENCLAW_BIN,
            "gateway",
            "--verbose",
        ]);
        cmd.env("HOME", OPENCLAW_HOME);
        cmd.env("OPENCLAW_HOME", OPENCLAW_HOME);
        cmd.env("PATH", &path_value);
        cmd.env("NPM_CONFIG_PREFIX", NPM_PREFIX);

        // Pass API keys from .env file
        for (key, value) in &env_vars {
            cmd.env(key, value);
        }

        // Fully detach child process
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());

        let child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn gateway: {}", e))?;

        let pid = child.id();
        info!(pid = ?pid, "[GATEWAY] Gateway spawned directly");

        Ok(pid)
    }

    // ============================================
    // Stop
    // ============================================

    /// Stops the OpenClaw gateway process.
    ///
    /// Strategy:
    /// 1. Try `systemctl --user stop openclaw-gateway`
    /// 2. If that fails, find PID via pgrep and send SIGTERM
    /// 3. Wait 3 seconds for graceful shutdown
    /// 4. If still running, send SIGKILL (last resort)
    ///
    /// **Idempotent**: safe to call if already stopped.
    pub async fn stop(known_pid: Option<u32>) -> Result<(), String> {
        info!("[GATEWAY] Stopping OpenClaw gateway...");

        // Try systemd first
        let systemd_result = CommandRunner::run_as_user(
            "systemctl",
            &["--user", "stop", "openclaw-gateway"],
        ).await;

        if systemd_result.is_ok() {
            info!("[GATEWAY] ✅ Stopped via systemd");
            return Ok(());
        }

        // Fallback: kill by PID
        let pid = known_pid
            .or_else(|| {
                // Sync find — we're already in an async context so
                // we can't easily await here. Use the known_pid if available,
                // otherwise the caller should have provided it.
                None
            });

        if let Some(pid) = pid {
            // SIGTERM first (graceful)
            let _ = CommandRunner::run_as_root("kill", &["-SIGTERM", &pid.to_string()]).await;

            // Wait for process to exit
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;

            // Check if still running
            let still_running = std::path::Path::new(&format!("/proc/{}", pid)).exists();
            if still_running {
                warn!(pid = pid, "[GATEWAY] Process still alive after SIGTERM, sending SIGKILL");
                let _ = CommandRunner::run_as_root("kill", &["-SIGKILL", &pid.to_string()]).await;
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        } else {
            // No PID available — try to find via pgrep and kill
            if let Some(pid) = Self::find_pid().await {
                let _ = CommandRunner::run_as_root("kill", &["-SIGTERM", &pid.to_string()]).await;
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }

        info!("[GATEWAY] ⏹️ Gateway stopped");
        Ok(())
    }

    // ============================================
    // Restart
    // ============================================

    /// Restarts the gateway: stop → start → wait for healthy.
    ///
    /// Returns the new PID if available.
    pub async fn restart(known_pid: Option<u32>) -> Result<Option<u32>, String> {
        info!("[GATEWAY] Restarting gateway...");

        // Stop
        let _ = Self::stop(known_pid).await;

        // Brief pause to ensure port is released
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Start
        let pid = Self::start().await?;

        // Wait for health
        if !Self::wait_for_healthy(OPENCLAW_DEFAULT_PORT).await {
            return Err("Gateway restarted but health check failed".to_string());
        }

        Ok(pid)
    }

    // ============================================
    // Health Check
    // ============================================

    /// Checks if the gateway is responding on the given port.
    ///
    /// Uses a simple TCP connection test (not a full WS/HTTP handshake).
    /// This is sufficient because the gateway port only opens when the
    /// process is fully initialized and ready to accept connections.
    pub async fn check_health(port: u16) -> bool {
        let addr = format!("127.0.0.1:{}", port);
        tokio::net::TcpStream::connect(&addr).await.is_ok()
    }

    /// Waits for the gateway to become healthy, polling every 3 seconds.
    ///
    /// Returns `true` if healthy within timeout, `false` if timed out.
    pub async fn wait_for_healthy(port: u16) -> bool {
        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(GATEWAY_STARTUP_TIMEOUT_SECS);

        while tokio::time::Instant::now() < deadline {
            if Self::check_health(port).await {
                return true;
            }
            tokio::time::sleep(std::time::Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS)).await;
        }

        false
    }

    // ============================================
    // PID Discovery
    // ============================================

    /// Finds the PID of a running OpenClaw gateway process.
    ///
    /// Uses `pgrep -f "openclaw.*gateway"` to find the process.
    /// Returns the first matching PID, or None if not running.
    pub async fn find_pid() -> Option<u32> {
        let output = TokioCommand::new("pgrep")
            .args(&["-f", "openclaw.*gateway"])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .and_then(|line| line.trim().parse::<u32>().ok())
    }

    // ============================================
    // Token
    // ============================================

    /// Reads the gateway authentication token from OpenClaw config.
    ///
    /// The token is stored at `gateway.auth.token` in `openclaw.json`.
    /// Returns None if the file doesn't exist or can't be parsed.
    pub async fn read_token() -> Option<String> {
        let content = tokio::fs::read_to_string(OPENCLAW_CONFIG_PATH).await.ok()?;

        // OpenClaw uses JSON5, but for simple token extraction,
        // standard JSON parsing works (tokens don't use JSON5 features).
        let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
        parsed
            .get("gateway")
            .and_then(|g| g.get("auth"))
            .and_then(|a| a.get("token"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
    }
}
