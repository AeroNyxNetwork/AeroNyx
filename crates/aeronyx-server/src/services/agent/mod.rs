//! ============================================
//! File: crates/aeronyx-server/src/services/agent/mod.rs
//! Path: aeronyx-server/src/services/agent/mod.rs
//! ============================================
//!
//! ## Creation Reason
//! Module root for the OpenClaw agent subsystem. Extracted from the
//! monolithic `agent_manager.rs` (v1.3.2) to improve maintainability
//! and separation of concerns.
//!
//! ## Main Functionality
//! - Re-exports all sub-modules for `agent_manager.rs` to consume
//! - Defines shared constants used across the agent subsystem
//! - Provides the `CommandRunner` helper for subprocess execution
//!
//! ## Module Layout
//! ```text
//! agent/
//! ├── mod.rs            ← this file (constants + command helpers + re-exports)
//! ├── preflight.rs      ← pre-install system environment checks
//! ├── installer.rs      ← Node.js + npm + OpenClaw package installation
//! ├── gateway.rs        ← gateway process lifecycle (start/stop/health/restart)
//! ├── process_stats.rs  ← Linux /proc CPU & memory collection
//! └── env_config.rs     ← .env file management + OpenClaw config operations
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - All subprocess calls use `tokio::process::Command` (async, NOT std)
//! - Commands run as the `openclaw` user via `sudo -u openclaw` unless
//!   explicitly marked as root (e.g., `useradd`, `apt-get`)
//! - The `OPENCLAW_HOME` constant is the canonical home dir — never
//!   hardcode `/home/openclaw` elsewhere, always reference this constant
//! - `CommandRunner` methods are intentionally `pub(super)` — only
//!   `agent_manager.rs` and sibling modules should call them
//!
//! ## Last Modified
//! v1.4.0 - 🌟 Initial creation (extracted from agent_manager.rs)
//! ============================================

pub mod preflight;
pub mod installer;
pub mod gateway;
pub mod process_stats;
pub mod env_config;

use tokio::process::Command as TokioCommand;
use tracing::debug;

// ============================================
// Shared Constants
// ============================================

/// Default OpenClaw gateway port (WS + HTTP multiplex).
pub const OPENCLAW_DEFAULT_PORT: u16 = 18789;

/// OpenClaw system user name.
pub const OPENCLAW_USER: &str = "openclaw";

/// OpenClaw home directory.
pub const OPENCLAW_HOME: &str = "/home/openclaw";

/// OpenClaw config file path.
pub const OPENCLAW_CONFIG_PATH: &str = "/home/openclaw/.openclaw/openclaw.json";

/// OpenClaw .env file path (API keys).
pub const OPENCLAW_ENV_PATH: &str = "/home/openclaw/.openclaw/.env";

/// npm global prefix directory for the openclaw user.
pub const NPM_PREFIX: &str = "/home/openclaw/.npm-global";

/// Full path to the openclaw binary (user-local npm install).
pub const OPENCLAW_BIN: &str = "/home/openclaw/.npm-global/bin/openclaw";

/// PATH value that includes the npm-global bin directory.
/// Used in all `sudo -u openclaw` calls.
pub fn openclaw_path() -> String {
    format!(
        "{}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        NPM_PREFIX
    )
}

// ============================================
// Command Runner (shared subprocess helpers)
// ============================================

/// Low-level helpers for running subprocesses.
///
/// All methods are `pub(super)` — only accessible from within the
/// `services` module (i.e., `agent_manager.rs` and `agent/*` siblings).
///
/// ## Security Model
/// - `run_as_root()`: Runs as the current process user (typically root).
///   Used ONLY for system operations: `useradd`, `apt-get`, `kill`.
/// - `run_as_user()`: Runs as `openclaw` via `sudo -u openclaw`.
///   Used for all OpenClaw CLI operations.
/// - Environment variables HOME, PATH, NPM_CONFIG_PREFIX, OPENCLAW_HOME
///   are always set explicitly to avoid inheriting incorrect values.
pub struct CommandRunner;

impl CommandRunner {
    /// Runs a command as root. Returns Ok(()) on success.
    pub(super) async fn run_as_root(program: &str, args: &[&str]) -> Result<(), String> {
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
    pub(super) async fn run_as_root_output(program: &str, args: &[&str]) -> Result<String, String> {
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
    /// Automatically sets HOME, PATH, NPM_CONFIG_PREFIX, OPENCLAW_HOME.
    pub(super) async fn run_as_user(program: &str, args: &[&str]) -> Result<(), String> {
        Self::run_as_user_with_env(program, args, &[]).await
    }

    /// Runs a command as the `openclaw` user with extra environment variables.
    pub(super) async fn run_as_user_with_env(
        program: &str,
        args: &[&str],
        extra_env: &[(&str, &str)],
    ) -> Result<(), String> {
        let path_value = openclaw_path();

        debug!(
            "[AGENT_CMD] sudo -u {} {} {}",
            OPENCLAW_USER, program, args.join(" ")
        );

        let mut cmd = TokioCommand::new("sudo");
        cmd.args(&[
            "-u", OPENCLAW_USER,
            "--preserve-env=PATH,HOME,NPM_CONFIG_PREFIX,OPENCLAW_HOME",
            program,
        ]);
        cmd.args(args);
        cmd.env("HOME", OPENCLAW_HOME);
        cmd.env("OPENCLAW_HOME", OPENCLAW_HOME);
        cmd.env("PATH", &path_value);
        cmd.env("NPM_CONFIG_PREFIX", NPM_PREFIX);

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
    pub(super) async fn run_as_user_output(program: &str, args: &[&str]) -> Result<String, String> {
        let path_value = openclaw_path();

        let output = TokioCommand::new("sudo")
            .args(&[
                "-u", OPENCLAW_USER,
                "--preserve-env=PATH,HOME,NPM_CONFIG_PREFIX,OPENCLAW_HOME",
                program,
            ])
            .args(args)
            .env("HOME", OPENCLAW_HOME)
            .env("OPENCLAW_HOME", OPENCLAW_HOME)
            .env("PATH", &path_value)
            .env("NPM_CONFIG_PREFIX", NPM_PREFIX)
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
}
