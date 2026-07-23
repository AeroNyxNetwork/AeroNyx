// ============================================
// File: crates/aeronyx-transport/src/tun/linux.rs
// ============================================
//! # Linux TUN Device Implementation
//!
//! ## Creation Reason
//! Provides Linux-specific TUN device implementation using the
//! `/dev/net/tun` interface.
//!
//! ## Main Functionality
//! - TUN device creation via ioctl
//! - IP address and route configuration
//! - Async read/write via Tokio
//! - Non-blocking operating-system interface control
//! - Device cleanup on drop
//! - Serialized interface lifecycle transitions
//! - Bounded external command execution with failed-start rollback
//!
//! ## Linux TUN Interface
//! Linux provides TUN devices through `/dev/net/tun`. The process:
//! 1. Open `/dev/net/tun`
//! 2. Use `TUNSETIFF` ioctl to configure
//! 3. Set device IP address via netlink or ip command
//! 4. Bring device up
//! 5. Read/write IP packets
//!
//! ## Required Capabilities
//! - `CAP_NET_ADMIN`: For creating and configuring TUN devices
//! - Or run as root
//!
//! ## ⚠️ Important Note for Next Developer
//! - This implementation requires Linux-specific headers
//! - Device creation may fail without proper permissions
//! - Always set `IFF_NO_PI` to avoid packet info headers
//! - Test with mock implementation when possible
//! - Keep unsafe ioctl calls inside the audited helpers in this module
//! - Keep every lifecycle command behind `lifecycle`; concurrent `up` and
//!   `down` calls must never interleave
//!
//! ## Last Modified
//! v0.3.0 - Serialized lifecycle transitions, bounded `ip` command execution,
//!          and rolled back partially applied activation state.
//! v0.2.0 - Moved `ip` operations to Tokio process I/O, centralized ioctl
//!          safety boundaries, and reused canonical netmask validation.
//! v0.1.1 - Flush existing global TUN addresses before applying the configured
//!          address so CIDR changes such as /24 -> /22 take effect on restart.
//! v0.1.0 - Initial Linux TUN implementation

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::Ipv4Addr;
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use nix::fcntl::{fcntl, FcntlArg, OFlag};
use nix::libc;
use tokio::io::unix::AsyncFd;
use tokio::io::Interest;
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::error::{Result, TransportError};
use crate::traits::{TunConfig, TunDevice};

// ============================================
// Constants
// ============================================

/// Path to the TUN device clone device.
const TUN_DEVICE_PATH: &str = "/dev/net/tun";

/// `IFF_TUN` flag - TUN device (no Ethernet headers).
const IFF_TUN: libc::c_short = 0x0001;

/// `IFF_NO_PI` flag - Do not provide packet information.
const IFF_NO_PI: libc::c_short = 0x1000;

/// TUNSETIFF ioctl number.
const TUNSETIFF: libc::c_ulong = 0x4004_54ca;

/// TUNSETPERSIST ioctl number.
const TUNSETPERSIST: libc::c_ulong = 0x4004_54cb;

/// Maximum time allowed for one operating-system interface operation.
const IP_COMMAND_TIMEOUT: Duration = Duration::from_secs(10);

// ============================================
// ifreq Structure
// ============================================

/// Interface request structure for ioctl calls.
#[repr(C)]
struct IfReq {
    ifr_name: [libc::c_char; libc::IFNAMSIZ],
    ifr_flags: libc::c_short,
    _padding: [u8; 22],
}

impl IfReq {
    fn new(name: &str) -> Self {
        let mut ifr = Self {
            ifr_name: [0; libc::IFNAMSIZ],
            ifr_flags: 0,
            _padding: [0; 22],
        };

        // Copy name into ifr_name (truncate if too long)
        let name_bytes = name.as_bytes();
        let copy_len = name_bytes.len().min(libc::IFNAMSIZ - 1);
        for (i, &byte) in name_bytes[..copy_len].iter().enumerate() {
            ifr.ifr_name[i] = libc::c_char::from_ne_bytes([byte]);
        }

        ifr
    }

    const fn with_flags(mut self, flags: libc::c_short) -> Self {
        self.ifr_flags = flags;
        self
    }

    fn name(&self) -> String {
        let bytes: Vec<u8> = self
            .ifr_name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| u8::from_ne_bytes(c.to_ne_bytes()))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// ============================================
// Audited Linux syscall boundary
// ============================================

/// Binds an open TUN clone descriptor to the requested interface.
#[allow(unsafe_code)]
fn set_tun_interface(fd: RawFd, request: &mut IfReq) -> std::io::Result<()> {
    // [TUN-SYSCALL 2026-07-23 by Codex] `request` is repr(C), fully
    // initialized, mutable for the duration of ioctl, and `fd` is owned by the
    // live File in `LinuxTun::create`.
    let result = unsafe { libc::ioctl(fd, TUNSETIFF, request) };
    if result < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Updates persistence for an already configured TUN descriptor.
#[allow(unsafe_code)]
fn set_tun_persistence(fd: RawFd, persist: bool) -> std::io::Result<()> {
    // [TUN-SYSCALL 2026-07-23 by Codex] TUNSETPERSIST accepts an integer value
    // and does not retain a pointer. The descriptor remains owned by File.
    let result = unsafe { libc::ioctl(fd, TUNSETPERSIST, i32::from(persist)) };
    if result < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Enables non-blocking mode while preserving every existing descriptor flag.
fn set_nonblocking(fd: RawFd) -> nix::Result<()> {
    let flags = OFlag::from_bits_truncate(fcntl(fd, FcntlArg::F_GETFL)?);
    fcntl(fd, FcntlArg::F_SETFL(flags | OFlag::O_NONBLOCK))?;
    Ok(())
}

/// Runs a child process with a bounded lifetime.
///
/// `kill_on_drop` is required because `timeout` cancels by dropping the output
/// future. Without it, a timed-out `ip` process could continue mutating the
/// interface after `LinuxTun::up` has started its rollback.
async fn run_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout_duration: Duration,
) -> std::io::Result<std::process::Output> {
    // [TUN-COMMAND-TIMEOUT 2026-07-23 by Codex] Keep process cancellation
    // semantics in one helper so every lifecycle command has the same bound.
    let mut command = Command::new(program);
    command.args(args).kill_on_drop(true);

    tokio::time::timeout(timeout_duration, command.output())
        .await
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "{program} command exceeded {} ms",
                    timeout_duration.as_millis()
                ),
            )
        })?
}

// ============================================
// LinuxTun
// ============================================

/// Linux TUN device implementation.
///
/// # Features
/// - Creates TUN device via `/dev/net/tun`
/// - Configures IP address and routing
/// - Async read/write via Tokio `AsyncFd`
/// - Automatic cleanup on drop
///
/// # Example
/// ```ignore
/// use aeronyx_transport::tun::LinuxTun;
/// use aeronyx_transport::traits::TunConfig;
///
/// let config = TunConfig::new("aeronyx0")
///     .with_address(Ipv4Addr::new(100, 64, 0, 1))
///     .with_mtu(1420);
///
/// let tun = LinuxTun::create(config).await?;
/// tun.up().await?;
///
/// // Read/write IP packets
/// let mut buf = [0u8; 1500];
/// let len = tun.read(&mut buf).await?;
/// ```
pub struct LinuxTun {
    /// Async file descriptor wrapper
    async_fd: AsyncFd<File>,
    /// Device configuration
    config: TunConfig,
    /// Whether the device is up
    is_up: AtomicBool,
    /// Serializes operating-system lifecycle changes.
    lifecycle: Mutex<()>,
}

impl LinuxTun {
    /// Creates a new TUN device with the given configuration.
    ///
    /// # Arguments
    /// * `config` - TUN device configuration
    ///
    /// # Errors
    /// - `TunCreateFailed`: If device creation fails
    /// - `PermissionDenied`: If lacking `CAP_NET_ADMIN`
    ///
    /// # Requirements
    /// - Must run as root or have `CAP_NET_ADMIN`
    /// - `/dev/net/tun` must exist
    #[allow(clippy::unused_async)]
    pub async fn create(config: TunConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        info!("Creating TUN device: {}", config.name);

        // Open /dev/net/tun
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(TUN_DEVICE_PATH)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    TransportError::PermissionDenied {
                        operation: format!("open {TUN_DEVICE_PATH}"),
                    }
                } else {
                    TransportError::tun_create_failed(&config.name, e.to_string())
                }
            })?;

        let fd = file.as_raw_fd();

        // Configure TUN device via ioctl
        let mut ifr = IfReq::new(&config.name).with_flags(IFF_TUN | IFF_NO_PI);

        if let Err(err) = set_tun_interface(fd, &mut ifr) {
            return Err(TransportError::tun_create_failed(
                &config.name,
                format!("TUNSETIFF failed: {err}"),
            ));
        }

        // Get the actual device name (may differ if we used a pattern)
        let actual_name = ifr.name();
        debug!("TUN device created: {}", actual_name);

        // Set persistence if requested
        if config.persist {
            if let Err(err) = set_tun_persistence(fd, true) {
                warn!("Failed to set TUN persistence: {}", err);
            }
        }

        // Set non-blocking mode
        set_nonblocking(fd).map_err(|err| {
            TransportError::tun_create_failed(
                &config.name,
                format!("Failed to set non-blocking mode: {err}"),
            )
        })?;

        // Create async fd wrapper
        let async_fd = AsyncFd::new(file).map_err(|e| {
            TransportError::tun_create_failed(&config.name, format!("AsyncFd creation failed: {e}"))
        })?;

        // Update config with actual name
        let mut config = config;
        config.name = actual_name;

        Ok(Self {
            async_fd,
            config,
            is_up: AtomicBool::new(false),
            lifecycle: Mutex::new(()),
        })
    }

    /// Configures the device IP address using `ip` command.
    ///
    /// This is a simpler alternative to netlink-based configuration.
    async fn configure_address(&self) -> Result<()> {
        let addr = format!("{}/{}", self.config.address, self.config.netmask_prefix()?);

        debug!("Configuring TUN address: {} on {}", addr, self.config.name);

        // The TUN device is owned by AeroNyx. Flush global addresses first so
        // maintenance changes to the configured CIDR cannot be masked by an
        // older address such as 100.64.0.1/24 already being present.
        self.run_ip(
            &["addr", "flush", "dev", &self.config.name, "scope", "global"],
            "ip addr flush",
        )
        .await?;
        self.run_ip(
            &["addr", "add", &addr, "dev", &self.config.name],
            "ip addr add",
        )
        .await
    }

    /// Sets the device MTU.
    async fn configure_mtu(&self) -> Result<()> {
        debug!("Setting MTU to {} on {}", self.config.mtu, self.config.name);

        let mtu = self.config.mtu.to_string();
        self.run_ip(
            &["link", "set", "dev", &self.config.name, "mtu", &mtu],
            "ip link set mtu",
        )
        .await
    }

    /// Marks the interface administratively up.
    async fn set_link_up(&self) -> Result<()> {
        self.run_ip(
            &["link", "set", "dev", &self.config.name, "up"],
            "ip link set up",
        )
        .await
    }

    /// Marks the interface administratively down.
    async fn set_link_down(&self) -> Result<()> {
        self.run_ip(
            &["link", "set", "dev", &self.config.name, "down"],
            "ip link set down",
        )
        .await
    }

    /// Applies all activation steps while the caller holds `lifecycle`.
    async fn activate(&self) -> Result<()> {
        self.configure_address().await?;
        self.configure_mtu().await?;
        self.set_link_up().await
    }

    /// Best-effort cleanup after an activation step returns an error.
    ///
    /// Rollback errors are logged but never replace the original activation
    /// error, which contains the actionable failure reported to the server.
    async fn rollback_failed_activation(&self) {
        // [TUN-LIFECYCLE 2026-07-23 by Codex] A failed MTU or link operation
        // must not leave a partially configured address behind for a later
        // restart to inherit.
        if let Err(err) = self.set_link_down().await {
            warn!(
                device = %self.config.name,
                error = %err,
                "Failed to bring TUN device down during activation rollback"
            );
        }

        if let Err(err) = self
            .run_ip(
                &["addr", "flush", "dev", &self.config.name, "scope", "global"],
                "ip addr flush during rollback",
            )
            .await
        {
            warn!(
                device = %self.config.name,
                error = %err,
                "Failed to flush TUN address during activation rollback"
            );
        }
    }

    /// Runs one `ip` operation without blocking the Tokio executor.
    async fn run_ip(&self, args: &[&str], operation: &str) -> Result<()> {
        // [ASYNC-TUN-CONTROL 2026-07-23 by Codex] Interface operations can
        // wait on udev/netlink. Tokio's process driver keeps that wait off the
        // runtime worker that forwards encrypted packets.
        let output = run_command_with_timeout("ip", args, IP_COMMAND_TIMEOUT)
            .await
            .map_err(|err| {
                if err.kind() == std::io::ErrorKind::TimedOut {
                    TransportError::Timeout {
                        operation: format!("{operation} for TUN '{}'", self.config.name),
                    }
                } else {
                    TransportError::TunConfigFailed {
                        name: self.config.name.clone(),
                        reason: format!("Failed to run {operation}: {err}"),
                    }
                }
            })?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(TransportError::TunConfigFailed {
            name: self.config.name.clone(),
            reason: format!("{operation} failed: {}", stderr.trim()),
        })
    }
}

#[async_trait]
impl TunDevice for LinuxTun {
    async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        loop {
            let mut guard = self.async_fd.ready(Interest::READABLE).await.map_err(|e| {
                TransportError::TunReadFailed {
                    reason: e.to_string(),
                }
            })?;

            match guard.try_io(|inner| {
                let mut file = inner.get_ref();
                file.read(buf)
            }) {
                Ok(Ok(len)) => return Ok(len),
                Ok(Err(e)) => {
                    return Err(TransportError::TunReadFailed {
                        reason: e.to_string(),
                    })
                }
                Err(_would_block) => {}
            }
        }
    }

    async fn write(&self, buf: &[u8]) -> Result<usize> {
        loop {
            let mut guard = self.async_fd.ready(Interest::WRITABLE).await.map_err(|e| {
                TransportError::TunWriteFailed {
                    reason: e.to_string(),
                }
            })?;

            match guard.try_io(|inner| {
                let mut file = inner.get_ref();
                file.write(buf)
            }) {
                Ok(Ok(len)) => return Ok(len),
                Ok(Err(e)) => {
                    return Err(TransportError::TunWriteFailed {
                        reason: e.to_string(),
                    })
                }
                Err(_would_block) => {}
            }
        }
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn mtu(&self) -> u16 {
        self.config.mtu
    }

    fn ip_addr(&self) -> Ipv4Addr {
        self.config.address
    }

    fn netmask(&self) -> Ipv4Addr {
        self.config.netmask
    }

    async fn up(&self) -> Result<()> {
        // [TUN-LIFECYCLE 2026-07-23 by Codex] Keep the guard across the whole
        // transition and rollback so `down` cannot interleave OS mutations.
        let _lifecycle_guard = self.lifecycle.lock().await;
        info!("Bringing up TUN device: {}", self.config.name);

        if let Err(err) = self.activate().await {
            self.is_up.store(false, Ordering::Release);
            warn!(
                device = %self.config.name,
                error = %err,
                "TUN activation failed; rolling back partial configuration"
            );
            self.rollback_failed_activation().await;
            return Err(err);
        }

        self.is_up.store(true, Ordering::Release);
        info!(
            "TUN device {} is up with IP {}",
            self.config.name, self.config.address
        );

        Ok(())
    }

    async fn down(&self) -> Result<()> {
        let _lifecycle_guard = self.lifecycle.lock().await;
        info!("Bringing down TUN device: {}", self.config.name);

        if let Err(err) = self.set_link_down().await {
            warn!(
                device = %self.config.name,
                error = %err,
                "Failed to bring TUN device down"
            );
        }

        self.is_up.store(false, Ordering::Release);
        info!("TUN device {} is down", self.config.name);

        Ok(())
    }

    fn is_up(&self) -> bool {
        self.is_up.load(Ordering::Acquire)
    }
}

impl Drop for LinuxTun {
    fn drop(&mut self) {
        debug!("Dropping TUN device: {}", self.config.name);
        // The file descriptor will be closed automatically
        // For non-persistent devices, the kernel will clean up
    }
}

impl std::fmt::Debug for LinuxTun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinuxTun")
            .field("name", &self.config.name)
            .field("address", &self.config.address)
            .field("mtu", &self.config.mtu)
            .field("is_up", &self.is_up())
            .finish_non_exhaustive()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Most TUN tests require root privileges and are skipped
    // in normal test runs. Use integration tests with proper setup.

    #[test]
    fn test_ifreq_creation() {
        let ifr = IfReq::new("test0").with_flags(IFF_TUN | IFF_NO_PI);

        assert_eq!(ifr.name(), "test0");
        assert_eq!(ifr.ifr_flags, IFF_TUN | IFF_NO_PI);
    }

    #[test]
    fn test_ifreq_name_truncation() {
        let long_name = "a".repeat(20);
        let ifr = IfReq::new(&long_name);

        // Name should be truncated to IFNAMSIZ - 1
        assert!(ifr.name().len() < libc::IFNAMSIZ);
    }

    #[tokio::test]
    async fn command_runner_returns_successful_output() {
        let output =
            run_command_with_timeout("/bin/sh", &["-c", "printf aeronyx"], Duration::from_secs(1))
                .await
                .expect("short command should complete");

        assert!(output.status.success());
        assert_eq!(output.stdout, b"aeronyx");
    }

    #[tokio::test]
    async fn command_runner_times_out() {
        let error = run_command_with_timeout("/bin/sleep", &["1"], Duration::from_millis(25))
            .await
            .expect_err("long command should time out");

        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    }
}
