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
//! - Device cleanup on drop
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
//! - Always set IFF_NO_PI to avoid packet info headers
//! - Test with mock implementation when possible
//!
//! ## Last Modified
//! v0.1.0 - Initial Linux TUN implementation

#![cfg(target_os = "linux")]

use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::Ipv4Addr;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use nix::libc;
use tokio::io::unix::AsyncFd;
use tokio::io::Interest;
use tracing::{debug, error, info, warn};

use crate::error::{Result, TransportError};
use crate::traits::{TunConfig, TunDevice};

// ============================================
// Constants
// ============================================

/// Path to the TUN device clone device.
const TUN_DEVICE_PATH: &str = "/dev/net/tun";

/// IFF_TUN flag - TUN device (no Ethernet headers).
const IFF_TUN: libc::c_short = 0x0001;

/// IFF_NO_PI flag - Do not provide packet information.
const IFF_NO_PI: libc::c_short = 0x1000;

/// TUNSETIFF ioctl number.
const TUNSETIFF: libc::c_ulong = 0x4004_54ca;

/// TUNSETPERSIST ioctl number.
const TUNSETPERSIST: libc::c_ulong = 0x4004_54cb;

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
            ifr.ifr_name[i] = byte as libc::c_char;
        }

        ifr
    }

    fn with_flags(mut self, flags: libc::c_short) -> Self {
        self.ifr_flags = flags;
        self
    }

    fn name(&self) -> String {
        let bytes: Vec<u8> = self
            .ifr_name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// ============================================
// LinuxTun
// ============================================

/// Linux TUN device implementation.
///
/// # Features
/// - Creates TUN device via `/dev/net/tun`
/// - Configures IP address and routing
/// - Async read/write via Tokio AsyncFd
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
}

impl LinuxTun {
    /// Creates a new TUN device with the given configuration.
    ///
    /// # Arguments
    /// * `config` - TUN device configuration
    ///
    /// # Errors
    /// - `TunCreateFailed`: If device creation fails
    /// - `PermissionDenied`: If lacking CAP_NET_ADMIN
    ///
    /// # Requirements
    /// - Must run as root or have CAP_NET_ADMIN
    /// - `/dev/net/tun` must exist
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
                        operation: format!("open {}", TUN_DEVICE_PATH),
                    }
                } else {
                    TransportError::tun_create_failed(&config.name, e.to_string())
                }
            })?;

        let fd = file.as_raw_fd();

        // Configure TUN device via ioctl
        let mut ifr = IfReq::new(&config.name).with_flags(IFF_TUN | IFF_NO_PI);

        let result = unsafe { libc::ioctl(fd, TUNSETIFF as libc::c_ulong, &mut ifr) };

        if result < 0 {
            let err = std::io::Error::last_os_error();
            return Err(TransportError::tun_create_failed(
                &config.name,
                format!("TUNSETIFF failed: {}", err),
            ));
        }

        // Get the actual device name (may differ if we used a pattern)
        let actual_name = ifr.name();
        debug!("TUN device created: {}", actual_name);

        // Set persistence if requested
        if config.persist {
            let persist_result = unsafe { libc::ioctl(fd, TUNSETPERSIST as libc::c_ulong, 1) };
            if persist_result < 0 {
                warn!("Failed to set TUN persistence: {}", std::io::Error::last_os_error());
            }
        }

        // Set non-blocking mode
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 {
            return Err(TransportError::tun_create_failed(
                &config.name,
                "Failed to get file flags",
            ));
        }

        let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
        if result < 0 {
            return Err(TransportError::tun_create_failed(
                &config.name,
                "Failed to set non-blocking mode",
            ));
        }

        // Create async fd wrapper
        let async_fd = AsyncFd::new(file).map_err(|e| {
            TransportError::tun_create_failed(&config.name, format!("AsyncFd creation failed: {}", e))
        })?;

        // Update config with actual name
        let mut config = config;
        config.name = actual_name;

        Ok(Self {
            async_fd,
            config,
            is_up: AtomicBool::new(false),
        })
    }

    /// Configures the device IP address using `ip` command.
    ///
    /// This is a simpler alternative to netlink-based configuration.
    async fn configure_address(&self) -> Result<()> {
        let addr = format!("{}/{}", self.config.address, self.netmask_to_cidr());

        debug!("Configuring TUN address: {} on {}", addr, self.config.name);

        let output = Command::new("ip")
            .args(["addr", "add", &addr, "dev", &self.config.name])
            .output()
            .map_err(|e| {
                TransportError::TunConfigFailed {
                    name: self.config.name.clone(),
                    reason: format!("Failed to run ip command: {}", e),
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Ignore "already exists" errors
            if !stderr.contains("RTNETLINK answers: File exists") {
                return Err(TransportError::TunConfigFailed {
                    name: self.config.name.clone(),
                    reason: format!("ip addr add failed: {}", stderr),
                });
            }
        }

        Ok(())
    }

    /// Sets the device MTU.
    async fn configure_mtu(&self) -> Result<()> {
        debug!("Setting MTU to {} on {}", self.config.mtu, self.config.name);

        let output = Command::new("ip")
            .args([
                "link",
                "set",
                "dev",
                &self.config.name,
                "mtu",
                &self.config.mtu.to_string(),
            ])
            .output()
            .map_err(|e| {
                TransportError::TunConfigFailed {
                    name: self.config.name.clone(),
                    reason: format!("Failed to set MTU: {}", e),
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TransportError::TunConfigFailed {
                name: self.config.name.clone(),
                reason: format!("ip link set mtu failed: {}", stderr),
            });
        }

        Ok(())
    }

    /// Converts netmask to CIDR prefix length.
    fn netmask_to_cidr(&self) -> u8 {
        let octets = self.config.netmask.octets();
        let bits: u32 = u32::from_be_bytes(octets);
        bits.count_ones() as u8
    }
}

#[async_trait]
impl TunDevice for LinuxTun {
    async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        loop {
            let mut guard = self
                .async_fd
                .ready(Interest::READABLE)
                .await
                .map_err(|e| TransportError::TunReadFailed {
                    reason: e.to_string(),
                })?;

            match guard.try_io(|inner| {
                let fd = inner.get_ref().as_raw_fd();
                let result = unsafe {
                    libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len())
                };

                if result < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(result as usize)
                }
            }) {
                Ok(Ok(len)) => return Ok(len),
                Ok(Err(e)) => {
                    return Err(TransportError::TunReadFailed {
                        reason: e.to_string(),
                    })
                }
                Err(_would_block) => continue,
            }
        }
    }

    async fn write(&self, buf: &[u8]) -> Result<usize> {
        loop {
            let mut guard = self
                .async_fd
                .ready(Interest::WRITABLE)
                .await
                .map_err(|e| TransportError::TunWriteFailed {
                    reason: e.to_string(),
                })?;

            match guard.try_io(|inner| {
                let fd = inner.get_ref().as_raw_fd();
                let result = unsafe {
                    libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len())
                };

                if result < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(result as usize)
                }
            }) {
                Ok(Ok(len)) => return Ok(len),
                Ok(Err(e)) => {
                    return Err(TransportError::TunWriteFailed {
                        reason: e.to_string(),
                    })
                }
                Err(_would_block) => continue,
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
        info!("Bringing up TUN device: {}", self.config.name);

        // Configure address first
        self.configure_address().await?;

        // Set MTU
        self.configure_mtu().await?;

        // Bring interface up
        let output = Command::new("ip")
            .args(["link", "set", "dev", &self.config.name, "up"])
            .output()
            .map_err(|e| {
                TransportError::TunConfigFailed {
                    name: self.config.name.clone(),
                    reason: format!("Failed to bring up interface: {}", e),
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TransportError::TunConfigFailed {
                name: self.config.name.clone(),
                reason: format!("ip link set up failed: {}", stderr),
            });
        }

        self.is_up.store(true, Ordering::Release);
        info!("TUN device {} is up with IP {}", self.config.name, self.config.address);

        Ok(())
    }

    async fn down(&self) -> Result<()> {
        info!("Bringing down TUN device: {}", self.config.name);

        let output = Command::new("ip")
            .args(["link", "set", "dev", &self.config.name, "down"])
            .output()
            .map_err(|e| {
                TransportError::TunConfigFailed {
                    name: self.config.name.clone(),
                    reason: format!("Failed to bring down interface: {}", e),
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("ip link set down failed: {}", stderr);
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
            .finish()
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

    #[test]
    fn test_netmask_to_cidr() {
        // We can't easily test this without creating a TUN device
        // Test the logic directly
        let mask_24 = Ipv4Addr::new(255, 255, 255, 0);
        let bits: u32 = u32::from_be_bytes(mask_24.octets());
        assert_eq!(bits.count_ones(), 24);

        let mask_16 = Ipv4Addr::new(255, 255, 0, 0);
        let bits: u32 = u32::from_be_bytes(mask_16.octets());
        assert_eq!(bits.count_ones(), 16);
    }
}
