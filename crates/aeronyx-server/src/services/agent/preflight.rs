// ============================================
// File: crates/aeronyx-server/src/services/agent/preflight.rs
// Path: aeronyx-server/src/services/agent/preflight.rs
// ============================================
//!
//! ## Creation Reason
//! Extracted from `agent_manager.rs` to provide thorough pre-installation
//! environment validation. Previously, install failures produced cryptic
//! errors deep in the npm/Node.js stack. Now we catch problems BEFORE
//! starting installation and give the user (via CMS dashboard) clear,
//! actionable error messages.
//!
//! ## Main Functionality
//! - `PreflightChecker`: Runs 12 system environment checks
//! - `PreflightReport`: Aggregated results with pass/warn/fail counts
//! - `PreflightCheck`: Individual check result with fix hints
//! - Each check is independent and non-destructive (read-only)
//!
//! ## Checks Performed (in order)
//! 1.  Operating system (Linux only)
//! 2.  CPU architecture (x86_64 or aarch64)
//! 3.  RAM (minimum 2GB, recommended 4GB)
//! 4.  Disk space (minimum 5GB free on OPENCLAW_HOME partition)
//! 5.  curl availability (required for Node.js installer)
//! 6.  Network connectivity (can reach npm registry)
//! 7.  Node.js version (must be 22+, or absent for fresh install)
//! 8.  npm version (must be 9+, or absent for fresh install)
//! 9.  System user (`openclaw` user existence and home dir permissions)
//! 10. Port availability (18789 not already bound)
//! 11. systemd availability (warn if absent — affects gateway startup method)
//! 12. Existing installation (detect leftover/corrupt installs)
//!
//! ## Design Decisions
//! - All checks return `PreflightStatus::Pass`, `Warn`, or `Fail`
//! - `Warn` = installation can proceed but may have issues
//! - `Fail` = installation should NOT proceed
//! - Each check includes a `fix_hint` string for the CMS dashboard
//! - Checks are ordered from cheapest/fastest to most expensive
//! - Network check has a 5-second timeout to avoid blocking
//!
//! ## Dependencies
//! - `tokio::fs` for async file reads (RAM, disk)
//! - `tokio::net::TcpStream` for port check
//! - `tokio::process::Command` for tool detection
//! - Parent module's `CommandRunner` for subprocess execution
//!
//! ## ⚠️ Important Note for Next Developer
//! - All checks are READ-ONLY — preflight never modifies the system
//! - The `fix_hint` messages are shown to end users via CMS dashboard,
//!   so keep them clear, specific, and non-technical where possible
//! - RAM/disk checks read from /proc and /sys — Linux only
//! - Network check uses `tokio::time::timeout` to avoid hanging
//! - If you add a new check, add it to `run_all()` AND update the
//!   check count in the module doc above
//! - NEVER use `block_on()` inside async functions — it panics in tokio.
//!   Use sequential `.await` calls instead (see check_disk fix).
//!
//! ## Last Modified
//! v1.4.0 - 🌟 Initial creation (extracted from agent_manager.rs)
//! v2.3.0 - 🐛 Fixed check_disk panic: replaced `block_on()` inside
//!   `or_else()` closure with sequential `.await` calls. The old code
//!   called `tokio::runtime::Handle::current().block_on()` from within
//!   an async task, which panics because tokio forbids blocking a thread
//!   that is driving async tasks.
// ============================================

use std::path::Path;
use tokio::net::TcpStream;
use tracing::{debug, info, warn};

use super::{CommandRunner, OPENCLAW_HOME, OPENCLAW_BIN, OPENCLAW_DEFAULT_PORT, OPENCLAW_USER};

// ============================================
// Types
// ============================================

/// Result status for a single preflight check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightStatus {
    /// Check passed — no issues.
    Pass,
    /// Check passed with warnings — installation can proceed but
    /// the user should be aware of potential issues.
    Warn,
    /// Check failed — installation should NOT proceed.
    Fail,
}

impl PreflightStatus {
    pub fn is_fail(&self) -> bool {
        matches!(self, PreflightStatus::Fail)
    }
}

impl std::fmt::Display for PreflightStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreflightStatus::Pass => write!(f, "✅ PASS"),
            PreflightStatus::Warn => write!(f, "⚠️ WARN"),
            PreflightStatus::Fail => write!(f, "❌ FAIL"),
        }
    }
}

/// A single preflight check result.
#[derive(Debug, Clone)]
pub struct PreflightCheck {
    /// Short name for the check (e.g., "ram", "nodejs", "disk").
    pub name: String,
    /// Pass/Warn/Fail status.
    pub status: PreflightStatus,
    /// Human-readable description of what was found.
    pub message: String,
    /// Actionable fix hint for the user/CMS dashboard.
    /// Empty string if status is Pass.
    pub fix_hint: String,
}

impl PreflightCheck {
    fn pass(name: &str, message: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status: PreflightStatus::Pass,
            message: message.into(),
            fix_hint: String::new(),
        }
    }

    fn warn(name: &str, message: impl Into<String>, fix_hint: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status: PreflightStatus::Warn,
            message: message.into(),
            fix_hint: fix_hint.into(),
        }
    }

    fn fail(name: &str, message: impl Into<String>, fix_hint: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status: PreflightStatus::Fail,
            message: message.into(),
            fix_hint: fix_hint.into(),
        }
    }
}

/// Aggregated results from all preflight checks.
#[derive(Debug, Clone)]
pub struct PreflightReport {
    /// All individual check results.
    pub checks: Vec<PreflightCheck>,
}

impl PreflightReport {
    /// Returns true if ANY check failed (installation should not proceed).
    pub fn has_failures(&self) -> bool {
        self.checks.iter().any(|c| c.status.is_fail())
    }

    /// Count of failed checks.
    pub fn fail_count(&self) -> usize {
        self.checks.iter().filter(|c| c.status == PreflightStatus::Fail).count()
    }

    /// Count of warning checks.
    pub fn warn_count(&self) -> usize {
        self.checks.iter().filter(|c| c.status == PreflightStatus::Warn).count()
    }

    /// Count of passed checks.
    pub fn pass_count(&self) -> usize {
        self.checks.iter().filter(|c| c.status == PreflightStatus::Pass).count()
    }

    /// Generates a human-readable summary string for CMS reporting.
    ///
    /// Example: "Preflight: 10 passed, 1 warning, 1 failed. Fix: RAM below 2GB minimum (have 1.8GB). Upgrade to at least 2GB RAM."
    pub fn summary(&self) -> String {
        let mut parts = vec![
            format!(
                "Preflight: {} passed, {} warnings, {} failed",
                self.pass_count(),
                self.warn_count(),
                self.fail_count()
            ),
        ];

        // Append failure details
        for check in &self.checks {
            if check.status == PreflightStatus::Fail {
                parts.push(format!("[{}] {}", check.name, check.message));
                if !check.fix_hint.is_empty() {
                    parts.push(format!("  Fix: {}", check.fix_hint));
                }
            }
        }

        // Append warning details (abbreviated)
        for check in &self.checks {
            if check.status == PreflightStatus::Warn {
                parts.push(format!("[{}] ⚠️ {}", check.name, check.message));
            }
        }

        parts.join(". ")
    }
}

// ============================================
// PreflightChecker
// ============================================

/// Runs pre-installation system environment checks.
///
/// All checks are read-only and non-destructive. They verify that the
/// system meets the minimum requirements for OpenClaw installation.
///
/// ## Usage
/// ```ignore
/// let report = PreflightChecker::run_all().await;
/// if report.has_failures() {
///     // Report failures to CMS and abort installation
///     return Err(report.summary());
/// }
/// // Proceed with installation
/// ```
pub struct PreflightChecker;

impl PreflightChecker {
    /// Runs all preflight checks and returns an aggregated report.
    ///
    /// Checks are run sequentially (they're fast — total < 2 seconds).
    /// Order: cheapest/fastest first, network last.
    pub async fn run_all() -> PreflightReport {
        info!("[PREFLIGHT] Running pre-installation system checks...");

        let mut checks = Vec::with_capacity(12);

        checks.push(Self::check_os().await);
        checks.push(Self::check_arch().await);
        checks.push(Self::check_ram().await);
        checks.push(Self::check_disk().await);
        checks.push(Self::check_curl().await);
        checks.push(Self::check_nodejs().await);
        checks.push(Self::check_npm().await);
        checks.push(Self::check_system_user().await);
        checks.push(Self::check_port().await);
        checks.push(Self::check_systemd().await);
        checks.push(Self::check_existing_install().await);
        checks.push(Self::check_network().await);

        let report = PreflightReport { checks };

        info!(
            pass = report.pass_count(),
            warn = report.warn_count(),
            fail = report.fail_count(),
            "[PREFLIGHT] Checks complete"
        );

        for check in &report.checks {
            match check.status {
                PreflightStatus::Pass => debug!(
                    name = %check.name,
                    "[PREFLIGHT] {} {}",
                    check.status, check.message
                ),
                PreflightStatus::Warn => warn!(
                    name = %check.name,
                    fix = %check.fix_hint,
                    "[PREFLIGHT] {} {}",
                    check.status, check.message
                ),
                PreflightStatus::Fail => warn!(
                    name = %check.name,
                    fix = %check.fix_hint,
                    "[PREFLIGHT] {} {}",
                    check.status, check.message
                ),
            }
        }

        report
    }

    // ============================================
    // Individual Checks
    // ============================================

    /// Check 1: Operating system must be Linux.
    async fn check_os() -> PreflightCheck {
        if cfg!(target_os = "linux") {
            PreflightCheck::pass("os", "Linux detected")
        } else {
            PreflightCheck::fail(
                "os",
                "OpenClaw requires Linux. This system is not Linux.",
                "Deploy on a Linux server (Ubuntu 22.04+ recommended)",
            )
        }
    }

    /// Check 2: CPU architecture must be x86_64 or aarch64.
    async fn check_arch() -> PreflightCheck {
        let arch = std::env::consts::ARCH;
        match arch {
            "x86_64" | "aarch64" => {
                PreflightCheck::pass("arch", format!("Architecture: {}", arch))
            }
            _ => PreflightCheck::fail(
                "arch",
                format!("Unsupported architecture: {}. OpenClaw requires x86_64 or aarch64.", arch),
                "Deploy on a server with x86_64 (AMD64) or aarch64 (ARM64) CPU",
            ),
        }
    }

    /// Check 3: RAM must be >= 2GB (warn if < 4GB).
    ///
    /// Reads from /proc/meminfo (Linux). OpenClaw uses 200-400MB at idle,
    /// but onboarding and npm install can spike to 1.5GB+. Below 2GB,
    /// the gateway crashes during startup.
    async fn check_ram() -> PreflightCheck {
        let total_mb = Self::read_total_ram_mb().await;

        match total_mb {
            Some(mb) if mb >= 4096 => {
                PreflightCheck::pass("ram", format!("RAM: {}MB ({}GB) — excellent", mb, mb / 1024))
            }
            Some(mb) if mb >= 2048 => {
                PreflightCheck::warn(
                    "ram",
                    format!("RAM: {}MB ({}GB) — meets minimum but may be tight", mb, mb / 1024),
                    "4GB+ RAM recommended for stable operation. Consider upgrading if you plan to use browser automation.",
                )
            }
            Some(mb) => {
                PreflightCheck::fail(
                    "ram",
                    format!("RAM: {}MB — below 2GB minimum. OpenClaw will crash during startup.", mb),
                    format!("Upgrade to at least 2GB RAM (4GB recommended). Current: {}MB.", mb),
                )
            }
            None => {
                PreflightCheck::warn(
                    "ram",
                    "Could not read /proc/meminfo to determine RAM",
                    "Ensure the system has at least 2GB RAM",
                )
            }
        }
    }

    /// Check 4: Disk space must be >= 5GB free on the partition
    /// containing OPENCLAW_HOME.
    ///
    /// OpenClaw install is ~500MB, but logs, workspace, and node_modules
    /// grow over time. 5GB is the recommended minimum.
    ///
    /// ## v2.3.0 Bug Fix
    /// Previously used `tokio::runtime::Handle::current().block_on()` inside
    /// a synchronous `or_else()` closure to check the fallback "/" path.
    /// This panics because `block_on()` blocks the current thread, which tokio
    /// forbids when that thread is already driving async tasks.
    /// Fixed by using sequential `.await` calls instead.
    async fn check_disk() -> PreflightCheck {
        // Try OPENCLAW_HOME first; if it doesn't exist yet, fall back to root partition.
        // 🐛 v2.3.0 fix: Previously used `block_on()` inside `or_else()` closure,
        // which panics when called from within a tokio async context.
        // Now uses sequential `.await` calls instead.
        let free_mb = match Self::read_free_disk_mb(OPENCLAW_HOME).await {
            Some(mb) => Some(mb),
            None => Self::read_free_disk_mb("/").await,
        };

        match free_mb {
            Some(mb) if mb >= 5120 => {
                PreflightCheck::pass("disk", format!("Disk: {}MB free ({}GB)", mb, mb / 1024))
            }
            Some(mb) if mb >= 2048 => {
                PreflightCheck::warn(
                    "disk",
                    format!("Disk: {}MB free — below 5GB recommended", mb),
                    "OpenClaw needs ~500MB for install + room for logs and workspace. Free up space or expand the disk.",
                )
            }
            Some(mb) => {
                PreflightCheck::fail(
                    "disk",
                    format!("Disk: {}MB free — below 2GB minimum", mb),
                    format!("Free up disk space. Need at least 5GB, have {}MB.", mb),
                )
            }
            None => {
                PreflightCheck::warn(
                    "disk",
                    "Could not determine free disk space",
                    "Ensure at least 5GB of free disk space",
                )
            }
        }
    }

    /// Check 5: `curl` must be available (used by Node.js installer).
    async fn check_curl() -> PreflightCheck {
        match CommandRunner::run_as_root("which", &["curl"]).await {
            Ok(_) => PreflightCheck::pass("curl", "curl is available"),
            Err(_) => PreflightCheck::fail(
                "curl",
                "curl is not installed. Required for Node.js installation.",
                "Install curl: sudo apt-get install -y curl",
            ),
        }
    }

    /// Check 6: Node.js version (must be 22+ if present).
    ///
    /// If Node.js is not installed, that's OK — the installer will
    /// handle it. But if an OLD version is installed, it must be
    /// upgraded first (version < 22 causes module resolution errors).
    async fn check_nodejs() -> PreflightCheck {
        let version_result = CommandRunner::run_as_root_output("node", &["--version"]).await;

        match version_result {
            Ok(version_str) => {
                let major = version_str
                    .trim()
                    .trim_start_matches('v')
                    .split('.')
                    .next()
                    .and_then(|s| s.parse::<u32>().ok());

                match major {
                    Some(m) if m >= 22 => {
                        PreflightCheck::pass("nodejs", format!("Node.js {} installed", version_str.trim()))
                    }
                    Some(m) => {
                        PreflightCheck::warn(
                            "nodejs",
                            format!("Node.js v{} is too old (need 22+). Will be upgraded during install.", m),
                            "The installer will upgrade Node.js automatically. If upgrade fails, run: curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -",
                        )
                    }
                    None => {
                        PreflightCheck::warn(
                            "nodejs",
                            format!("Could not parse Node.js version: {}", version_str.trim()),
                            "The installer will handle Node.js installation",
                        )
                    }
                }
            }
            Err(_) => {
                // Not installed — that's fine, installer will handle it
                PreflightCheck::pass("nodejs", "Node.js not installed (will be installed automatically)")
            }
        }
    }

    /// Check 7: npm version (must be 9+ if present).
    async fn check_npm() -> PreflightCheck {
        let version_result = CommandRunner::run_as_root_output("npm", &["--version"]).await;

        match version_result {
            Ok(version_str) => {
                let major = version_str
                    .trim()
                    .split('.')
                    .next()
                    .and_then(|s| s.parse::<u32>().ok());

                match major {
                    Some(m) if m >= 9 => {
                        PreflightCheck::pass("npm", format!("npm {} installed", version_str.trim()))
                    }
                    Some(m) => {
                        PreflightCheck::warn(
                            "npm",
                            format!("npm v{} is old (need 9+). Will be updated with Node.js.", m),
                            "npm will be updated automatically when Node.js is upgraded",
                        )
                    }
                    None => {
                        PreflightCheck::warn(
                            "npm",
                            format!("Could not parse npm version: {}", version_str.trim()),
                            "npm ships with Node.js and will be installed automatically",
                        )
                    }
                }
            }
            Err(_) => {
                PreflightCheck::pass("npm", "npm not installed (ships with Node.js)")
            }
        }
    }

    /// Check 8: System user and home directory.
    ///
    /// If the `openclaw` user exists, verify the home directory is
    /// accessible and has correct ownership. If the user doesn't
    /// exist, that's fine — the installer will create it.
    async fn check_system_user() -> PreflightCheck {
        // Check if user exists
        let user_exists = CommandRunner::run_as_root("id", &[OPENCLAW_USER]).await.is_ok();

        if !user_exists {
            return PreflightCheck::pass(
                "user",
                format!("User '{}' does not exist (will be created during install)", OPENCLAW_USER),
            );
        }

        // User exists — check home directory
        let home = Path::new(OPENCLAW_HOME);
        if !home.exists() {
            return PreflightCheck::warn(
                "user",
                format!("User '{}' exists but home directory {} is missing", OPENCLAW_USER, OPENCLAW_HOME),
                format!("Run: sudo mkdir -p {} && sudo chown {}:{} {}", OPENCLAW_HOME, OPENCLAW_USER, OPENCLAW_USER, OPENCLAW_HOME),
            );
        }

        // Check ownership
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::MetadataExt;
            if let Ok(meta) = std::fs::metadata(OPENCLAW_HOME) {
                let uid = meta.uid();
                // Get expected UID for the openclaw user
                let expected_uid = CommandRunner::run_as_root_output("id", &["-u", OPENCLAW_USER])
                    .await
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok());

                if let Some(expected) = expected_uid {
                    if uid != expected {
                        return PreflightCheck::fail(
                            "user",
                            format!(
                                "Home directory {} is owned by UID {} but user '{}' has UID {}",
                                OPENCLAW_HOME, uid, OPENCLAW_USER, expected
                            ),
                            format!(
                                "Fix ownership: sudo chown -R {}:{} {}",
                                OPENCLAW_USER, OPENCLAW_USER, OPENCLAW_HOME
                            ),
                        );
                    }
                }
            }
        }

        PreflightCheck::pass(
            "user",
            format!("User '{}' exists with correct home directory", OPENCLAW_USER),
        )
    }

    /// Check 9: Port 18789 must not already be in use.
    ///
    /// If something else is bound to this port, the gateway can't start.
    /// Common culprit: a leftover gateway process from a previous install.
    async fn check_port() -> PreflightCheck {
        let addr = format!("127.0.0.1:{}", OPENCLAW_DEFAULT_PORT);

        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            TcpStream::connect(&addr),
        ).await {
            Ok(Ok(_)) => {
                // Something is listening — could be an existing gateway
                PreflightCheck::warn(
                    "port",
                    format!("Port {} is already in use", OPENCLAW_DEFAULT_PORT),
                    format!(
                        "An existing process is using port {}. If it's an old OpenClaw gateway, stop it first: \
                         sudo -u openclaw openclaw gateway stop, or find the process: ss -tlnp | grep {}",
                        OPENCLAW_DEFAULT_PORT, OPENCLAW_DEFAULT_PORT
                    ),
                )
            }
            Ok(Err(_)) | Err(_) => {
                // Can't connect — port is free
                PreflightCheck::pass("port", format!("Port {} is available", OPENCLAW_DEFAULT_PORT))
            }
        }
    }

    /// Check 10: systemd user services availability.
    ///
    /// OpenClaw prefers systemd for gateway management. Without it,
    /// we fall back to direct process spawning (which works but is
    /// less robust for auto-restart on crash).
    async fn check_systemd() -> PreflightCheck {
        let systemd_available = CommandRunner::run_as_root("systemctl", &["--version"]).await.is_ok();

        if systemd_available {
            // Check if user linger is supported (needed for user services)
            let linger_available = CommandRunner::run_as_root("loginctl", &["--version"]).await.is_ok();
            if linger_available {
                PreflightCheck::pass("systemd", "systemd with user services available")
            } else {
                PreflightCheck::warn(
                    "systemd",
                    "systemd is available but loginctl (user linger) may not work",
                    "Gateway will use direct process spawning as fallback. For auto-restart on crash, ensure systemd user services work.",
                )
            }
        } else {
            PreflightCheck::warn(
                "systemd",
                "systemd is not available. Gateway will run via direct process spawning.",
                "This is OK for most setups. For production, consider using a process manager (tmux, screen, or supervisor).",
            )
        }
    }

    /// Check 11: Detect existing (possibly corrupt) installation.
    ///
    /// If OpenClaw binary exists but config is missing or broken,
    /// the user should clean up before reinstalling.
    async fn check_existing_install() -> PreflightCheck {
        let bin_exists = Path::new(OPENCLAW_BIN).exists();
        let config_exists = Path::new(super::OPENCLAW_CONFIG_PATH).exists();

        match (bin_exists, config_exists) {
            (false, false) => {
                PreflightCheck::pass("existing", "No existing OpenClaw installation detected")
            }
            (true, true) => {
                // Full install exists — not an error, but note it
                PreflightCheck::warn(
                    "existing",
                    "OpenClaw is already installed. Re-installing will update the existing installation.",
                    "If you want a clean install, uninstall first: openclaw uninstall, or delete ~/.openclaw/ and ~/.npm-global/",
                )
            }
            (true, false) => {
                // Binary exists but no config — possibly a failed install
                PreflightCheck::warn(
                    "existing",
                    "OpenClaw binary found but config is missing. This may be a partial/failed installation.",
                    format!(
                        "Try running: sudo -u {} {} doctor --fix. If that fails, clean up: sudo rm -rf {}/.openclaw {}",
                        OPENCLAW_USER, OPENCLAW_BIN, OPENCLAW_HOME, OPENCLAW_BIN
                    ),
                )
            }
            (false, true) => {
                // Config exists but no binary — leftover from uninstall
                PreflightCheck::warn(
                    "existing",
                    "OpenClaw config exists but binary is missing. This is leftover from a previous uninstall.",
                    format!("Clean up: sudo rm -rf {}/.openclaw", OPENCLAW_HOME),
                )
            }
        }
    }

    /// Check 12: Network connectivity to npm registry.
    ///
    /// The installer needs to download packages from npm. If the server
    /// can't reach the registry, installation will fail with timeouts.
    /// Uses a 5-second timeout to avoid blocking.
    async fn check_network() -> PreflightCheck {
        let check = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            TcpStream::connect("registry.npmjs.org:443"),
        ).await;

        match check {
            Ok(Ok(_)) => {
                PreflightCheck::pass("network", "Can reach npm registry (registry.npmjs.org)")
            }
            Ok(Err(e)) => {
                PreflightCheck::fail(
                    "network",
                    format!("Cannot connect to npm registry: {}", e),
                    "Ensure the server has internet access. Check DNS, firewall, and proxy settings. \
                     Try: curl -sS https://registry.npmjs.org/ | head -1",
                )
            }
            Err(_) => {
                PreflightCheck::fail(
                    "network",
                    "Connection to npm registry timed out (5s)",
                    "Network is too slow or blocked. Check firewall rules and DNS. \
                     If behind a proxy, configure npm: npm config set proxy http://proxy:port",
                )
            }
        }
    }

    // ============================================
    // Helper: Read System Info
    // ============================================

    /// Reads total RAM in MB from /proc/meminfo.
    async fn read_total_ram_mb() -> Option<u64> {
        let content = tokio::fs::read_to_string("/proc/meminfo").await.ok()?;

        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)?
                    .parse()
                    .ok()?;
                return Some(kb / 1024);
            }
        }
        None
    }

    /// Reads free disk space in MB for the partition containing the given path.
    ///
    /// Uses `statvfs` syscall via `nix` crate if available, otherwise
    /// falls back to parsing `df` output.
    async fn read_free_disk_mb(path: &str) -> Option<u64> {
        // Use `df` command — works everywhere on Linux
        let output = tokio::process::Command::new("df")
            .args(&["-BM", "--output=avail", path])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        // df output:
        //  Avail
        // 45678M
        stdout
            .lines()
            .skip(1) // skip header
            .next()
            .and_then(|line| {
                line.trim()
                    .trim_end_matches('M')
                    .parse::<u64>()
                    .ok()
            })
    }
}
