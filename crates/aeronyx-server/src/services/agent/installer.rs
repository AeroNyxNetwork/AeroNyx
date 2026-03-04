//! ============================================
//! File: crates/aeronyx-server/src/services/agent/installer.rs
//! Path: aeronyx-server/src/services/agent/installer.rs
//! ============================================
//!
//! ## Creation Reason
//! Extracted from `agent_manager.rs` to isolate the installation logic
//! (system user creation, Node.js setup, npm package install, onboarding).
//! Each step has its own error handling and is independently testable.
//!
//! ## Main Functionality
//! - `AgentInstaller`: Stateless struct with installation step methods
//! - `ensure_system_user()`: Create the `openclaw` Linux user
//! - `ensure_nodejs()`: Install/upgrade Node.js 22+ via NodeSource
//! - `install_openclaw_package()`: Install OpenClaw via npm
//! - `run_onboarding()`: Run `openclaw onboard` in non-interactive mode
//!
//! ## Known Installation Pitfalls (from community reports)
//! 1. **EACCES permission denied**: npm tries to write to /usr/local/lib.
//!    Fix: Use NPM_CONFIG_PREFIX to redirect to ~/.npm-global.
//! 2. **Node.js < 22**: Causes module resolution errors deep in the stack.
//!    Fix: Always install via NodeSource (not distro packages which are old).
//! 3. **Missing cmake/gcc**: Some npm native dependencies need build tools.
//!    Fix: Install build-essential as a precaution.
//! 4. **npm install timeout**: Slow network causes silent failure.
//!    Fix: The installer retries once with `--prefer-offline` fallback.
//! 5. **Onboarding leaves incomplete config**: If interrupted, config is
//!    half-written. Fix: `openclaw doctor --fix` after onboard.
//!
//! ## Dependencies
//! - `CommandRunner` from parent module for subprocess execution
//! - `tokio::process::Command` for async subprocess calls
//!
//! ## ⚠️ Important Note for Next Developer
//! - Never use `sudo npm install -g` — it causes EACCES. Always use
//!   NPM_CONFIG_PREFIX to redirect to user-local directory.
//! - The openclaw user must own all files under ~/.openclaw/ and ~/.npm-global/.
//!   If ownership is wrong, onboarding and gateway startup will fail.
//! - `run_onboarding` is sensitive to environment variables. If you add new
//!   env vars, add them to the `--preserve-env=` list in the sudo command.
//! - After onboarding, always run `openclaw doctor --fix` to catch any
//!   config issues the wizard might have missed.
//!
//! ## Last Modified
//! v1.4.0 - 🌟 Initial creation (extracted from agent_manager.rs)
//! ============================================

use tokio::process::Command as TokioCommand;
use tracing::{debug, info, warn};

use super::{
    CommandRunner, OPENCLAW_HOME, OPENCLAW_USER, OPENCLAW_BIN,
    NPM_PREFIX, openclaw_path,
};

// ============================================
// AgentInstaller
// ============================================

/// Handles the step-by-step installation of OpenClaw.
///
/// Each method is a single installation step that can fail independently.
/// The caller (`AgentManager`) orchestrates the steps and reports progress.
///
/// ## Usage
/// ```ignore
/// AgentInstaller::ensure_system_user().await?;
/// AgentInstaller::ensure_nodejs().await?;
/// AgentInstaller::ensure_build_tools().await?;
/// AgentInstaller::install_openclaw_package("latest").await?;
/// AgentInstaller::run_onboarding(&params).await?;
/// AgentInstaller::post_install_doctor().await?;
/// ```
pub struct AgentInstaller;

impl AgentInstaller {
    // ============================================
    // Step 1: System User
    // ============================================

    /// Ensures the `openclaw` system user exists with correct setup.
    ///
    /// Creates the user with:
    /// - System account (no password, no aging)
    /// - Home directory at /home/openclaw
    /// - bash shell (needed for openclaw CLI)
    /// - Linger enabled (systemd user services survive logout)
    ///
    /// **Idempotent**: safe to call if user already exists.
    pub async fn ensure_system_user() -> Result<(), String> {
        let exists = CommandRunner::run_as_root("id", &[OPENCLAW_USER]).await.is_ok();
        if exists {
            debug!("[INSTALL] User '{}' already exists", OPENCLAW_USER);

            // Ensure home directory exists and has correct ownership
            Self::fix_home_directory_permissions().await?;
            return Ok(());
        }

        info!("[INSTALL] Creating system user '{}'...", OPENCLAW_USER);

        CommandRunner::run_as_root(
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
        // and survive session logout. Not fatal if it fails (container env).
        if let Err(e) = CommandRunner::run_as_root(
            "loginctl",
            &["enable-linger", OPENCLAW_USER],
        ).await {
            warn!(
                error = %e,
                "[INSTALL] ⚠️ loginctl enable-linger failed (non-fatal, may be a container)"
            );
        }

        // Ensure ~/.openclaw directory exists with correct ownership
        let openclaw_dir = format!("{}/.openclaw", OPENCLAW_HOME);
        CommandRunner::run_as_user("mkdir", &["-p", &openclaw_dir]).await
            .map_err(|e| format!("Failed to create .openclaw dir: {}", e))?;

        info!("[INSTALL] ✅ User '{}' created", OPENCLAW_USER);
        Ok(())
    }

    /// Fixes home directory permissions if they're wrong.
    ///
    /// Common after manual interventions or container UID remapping.
    async fn fix_home_directory_permissions() -> Result<(), String> {
        let home = std::path::Path::new(OPENCLAW_HOME);
        if !home.exists() {
            CommandRunner::run_as_root(
                "mkdir", &["-p", OPENCLAW_HOME]
            ).await.map_err(|e| format!("mkdir home failed: {}", e))?;
        }

        // Fix ownership (idempotent)
        CommandRunner::run_as_root(
            "chown",
            &["-R", &format!("{}:{}", OPENCLAW_USER, OPENCLAW_USER), OPENCLAW_HOME],
        ).await.map_err(|e| format!("chown home failed: {}", e))?;

        Ok(())
    }

    // ============================================
    // Step 2: Node.js
    // ============================================

    /// Ensures Node.js 22+ is installed.
    ///
    /// Strategy:
    /// 1. Check if Node.js is already installed and >= 22
    /// 2. If missing or too old, install via NodeSource setup script
    /// 3. Verify the installed version
    ///
    /// Uses NodeSource rather than distro packages because Ubuntu/Debian
    /// apt repositories typically ship Node.js 18 or older.
    pub async fn ensure_nodejs() -> Result<(), String> {
        if let Ok(version_str) = CommandRunner::run_as_root_output("node", &["--version"]).await {
            let major: Option<u32> = version_str
                .trim()
                .trim_start_matches('v')
                .split('.')
                .next()
                .and_then(|s| s.parse().ok());

            if let Some(m) = major {
                if m >= 22 {
                    info!("[INSTALL] Node.js {} already installed", version_str.trim());
                    return Ok(());
                }
            }
            info!("[INSTALL] Node.js {} too old, upgrading to 22...", version_str.trim());
        }

        info!("[INSTALL] Installing Node.js 22 via NodeSource...");

        CommandRunner::run_as_root(
            "bash",
            &["-c", "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"],
        ).await.map_err(|e| format!("NodeSource setup failed: {}. Ensure curl is installed and network is available.", e))?;

        CommandRunner::run_as_root(
            "apt-get",
            &["install", "-y", "nodejs"],
        ).await.map_err(|e| format!("apt-get install nodejs failed: {}", e))?;

        // Verify
        let version = CommandRunner::run_as_root_output("node", &["--version"]).await
            .map_err(|e| format!("Node.js verification failed after install: {}", e))?;

        info!("[INSTALL] ✅ Node.js {} installed", version.trim());
        Ok(())
    }

    // ============================================
    // Step 3: Build Tools (optional but prevents failures)
    // ============================================

    /// Ensures build-essential is installed (gcc, make, etc.).
    ///
    /// Some npm packages (like `sharp`) compile native addons during
    /// install. Without build tools, you get cryptic `node-gyp` errors.
    /// This is a best-effort step — if it fails, we continue anyway
    /// since not all installs need native compilation.
    pub async fn ensure_build_tools() -> Result<(), String> {
        // Check if gcc is available as a proxy for build tools
        if CommandRunner::run_as_root("which", &["gcc"]).await.is_ok() {
            debug!("[INSTALL] Build tools already available");
            return Ok(());
        }

        info!("[INSTALL] Installing build-essential (for native npm packages)...");

        if let Err(e) = CommandRunner::run_as_root(
            "apt-get",
            &["install", "-y", "build-essential", "python3"],
        ).await {
            warn!(
                error = %e,
                "[INSTALL] ⚠️ build-essential install failed (non-fatal)"
            );
            // Non-fatal: continue and hope npm packages don't need native compilation
        }

        Ok(())
    }

    // ============================================
    // Step 4: OpenClaw Package
    // ============================================

    /// Installs the OpenClaw npm package as the openclaw user.
    ///
    /// Uses `NPM_CONFIG_PREFIX` to install into `~/.npm-global/` instead
    /// of `/usr/lib/node_modules/` which requires root and causes EACCES.
    ///
    /// Retries once with `--prefer-offline` if the first attempt fails
    /// (handles flaky networks).
    pub async fn install_openclaw_package(version: &str) -> Result<(), String> {
        let package = if version == "latest" {
            "openclaw@latest".to_string()
        } else {
            format!("openclaw@{}", version)
        };

        info!("[INSTALL] Installing {} via npm...", package);

        // Ensure npm global prefix directory exists
        CommandRunner::run_as_user("mkdir", &["-p", NPM_PREFIX]).await
            .map_err(|e| format!("Failed to create npm prefix dir: {}", e))?;

        // First attempt: normal install
        let result = CommandRunner::run_as_user_with_env(
            "npm",
            &["install", "-g", &package],
            &[("NPM_CONFIG_PREFIX", NPM_PREFIX)],
        ).await;

        if let Err(ref first_error) = result {
            warn!(
                error = %first_error,
                "[INSTALL] ⚠️ First npm install attempt failed, retrying with --prefer-offline..."
            );

            // Retry with --prefer-offline (uses cache, faster, handles flaky network)
            CommandRunner::run_as_user_with_env(
                "npm",
                &["install", "-g", "--prefer-offline", &package],
                &[("NPM_CONFIG_PREFIX", NPM_PREFIX)],
            ).await.map_err(|e| {
                format!(
                    "npm install failed twice. First error: {}. Retry error: {}. \
                     Check network connectivity and npm registry access.",
                    first_error, e
                )
            })?;
        }

        // Verify the binary exists
        if !std::path::Path::new(OPENCLAW_BIN).exists() {
            return Err(format!(
                "openclaw binary not found at {} after install. \
                 This usually means npm silently failed. Check: \
                 sudo -u {} npm list -g openclaw",
                OPENCLAW_BIN, OPENCLAW_USER
            ));
        }

        info!("[INSTALL] ✅ {} installed at {}", package, OPENCLAW_BIN);
        Ok(())
    }

    // ============================================
    // Step 5: Onboarding
    // ============================================

    /// Runs OpenClaw onboarding in non-interactive mode.
    ///
    /// This creates the initial configuration at `~/.openclaw/openclaw.json`,
    /// sets up the gateway auth token, and optionally installs the systemd
    /// service.
    ///
    /// ## Environment Variables
    /// - `ANTHROPIC_API_KEY`: If CMS provides an API key, it's passed here
    /// - `HOME`, `PATH`, `NPM_CONFIG_PREFIX`: Standard user environment
    ///
    /// ## Post-Onboard
    /// Always run `post_install_doctor()` after this to catch any config
    /// issues the wizard might have missed.
    pub async fn run_onboarding(params: &serde_json::Value) -> Result<(), String> {
        info!("[INSTALL] Running onboarding (non-interactive)...");

        let path_value = openclaw_path();

        let api_key = params.get("api_key").and_then(|v| v.as_str());

        let mut cmd = TokioCommand::new("sudo");
        cmd.args(&[
            "-u", OPENCLAW_USER,
            "--preserve-env=PATH,HOME,OPENCLAW_HOME,NPM_CONFIG_PREFIX",
            OPENCLAW_BIN,
            "onboard",
            "--install-daemon",
        ]);
        cmd.env("HOME", OPENCLAW_HOME);
        cmd.env("OPENCLAW_HOME", OPENCLAW_HOME);
        cmd.env("PATH", &path_value);
        cmd.env("NPM_CONFIG_PREFIX", NPM_PREFIX);

        if let Some(key) = api_key {
            cmd.env("ANTHROPIC_API_KEY", key);
        }

        // Timeout: onboarding should complete in under 60 seconds.
        // If it hangs, it's probably waiting for interactive input
        // that we failed to suppress.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            cmd.output(),
        ).await;

        match result {
            Ok(Ok(output)) => {
                if output.status.success() {
                    info!("[INSTALL] ✅ Onboarding complete");
                    Ok(())
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    Err(format!(
                        "Onboarding failed (exit code {:?}). stderr: {}. stdout: {}",
                        output.status.code(),
                        stderr.chars().take(500).collect::<String>(),
                        stdout.chars().take(500).collect::<String>(),
                    ))
                }
            }
            Ok(Err(e)) => Err(format!("Onboarding command failed to execute: {}", e)),
            Err(_) => Err(
                "Onboarding timed out (120s). It may be waiting for interactive input. \
                 Try running manually: sudo -u openclaw openclaw onboard --install-daemon"
                    .to_string()
            ),
        }
    }

    // ============================================
    // Step 6: Post-Install Doctor
    // ============================================

    /// Runs `openclaw doctor --fix` after installation to catch and
    /// auto-repair common config issues.
    ///
    /// Known issues doctor fixes:
    /// - Missing gateway.auth.token (auto-generates one)
    /// - Missing directories (auto-creates)
    /// - Permission problems on ~/.openclaw/ files
    /// - Incomplete config from interrupted onboarding
    ///
    /// **Non-fatal**: if doctor fails, we log a warning and continue.
    /// The gateway may still start successfully.
    pub async fn post_install_doctor() -> Result<(), String> {
        info!("[INSTALL] Running openclaw doctor --fix (post-install)...");

        let result = CommandRunner::run_as_user(
            "openclaw",
            &["doctor", "--fix", "--yes"],
        ).await;

        match result {
            Ok(()) => {
                info!("[INSTALL] ✅ Doctor completed — config is healthy");
                Ok(())
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "[INSTALL] ⚠️ Doctor failed (non-fatal) — gateway may still start"
                );
                // Non-fatal: continue with installation
                Ok(())
            }
        }
    }

    // ============================================
    // Uninstall
    // ============================================

    /// Removes OpenClaw npm package and optionally cleans config.
    ///
    /// ## Steps
    /// 1. Disable systemd service (if exists)
    /// 2. Uninstall npm package
    /// 3. Remove ~/.openclaw/ directory
    ///
    /// **Does NOT delete the `openclaw` system user** — that's a more
    /// destructive operation that should require explicit confirmation.
    pub async fn uninstall(clean_config: bool) -> Result<(), String> {
        info!("[UNINSTALL] Removing OpenClaw...");

        // Disable systemd service (best-effort)
        let _ = CommandRunner::run_as_user(
            "systemctl",
            &["--user", "disable", "openclaw-gateway"],
        ).await;

        // Uninstall npm package
        let _ = CommandRunner::run_as_user_with_env(
            "npm",
            &["uninstall", "-g", "openclaw"],
            &[("NPM_CONFIG_PREFIX", NPM_PREFIX)],
        ).await;

        if clean_config {
            let openclaw_config_dir = format!("{}/.openclaw", OPENCLAW_HOME);
            CommandRunner::run_as_root("rm", &["-rf", &openclaw_config_dir]).await
                .map_err(|e| format!("Failed to remove config dir: {}", e))?;
            info!("[UNINSTALL] Config directory removed");
        }

        info!("[UNINSTALL] ✅ OpenClaw uninstalled");
        Ok(())
    }

    // ============================================
    // Version Query
    // ============================================

    /// Gets the installed OpenClaw version string.
    ///
    /// Returns `None` if not installed or version can't be determined.
    pub async fn get_installed_version() -> Option<String> {
        CommandRunner::run_as_user_output("openclaw", &["--version"])
            .await
            .ok()
    }
}
