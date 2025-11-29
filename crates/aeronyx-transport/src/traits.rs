// ============================================
// File: crates/aeronyx-transport/src/traits.rs
// ============================================
//! # Transport Traits
//!
//! ## Creation Reason
//! Defines abstract interfaces for transport operations, enabling
//! testability and flexibility in implementation choices.
//!
//! ## Main Functionality
//! - `Transport`: UDP-like datagram transport interface
//! - `TunDevice`: TUN device read/write interface
//! - `PacketSource`: Metadata about received packets
//!
//! ## Design Philosophy
//! - Traits enable mock implementations for testing
//! - Async-first design with `async_trait`
//! - Zero-copy interfaces where practical
//! - Platform-agnostic definitions
//!
//! ## ⚠️ Important Note for Next Developer
//! - All trait methods are async for consistency
//! - Implementations must be Send + Sync for use in async contexts
//! - Buffer management is caller's responsibility
//!
//! ## Last Modified
//! v0.1.0 - Initial trait definitions

use std::net::SocketAddr;
use std::time::Instant;

use async_trait::async_trait;

use crate::error::Result;

// ============================================
// PacketSource
// ============================================

/// Metadata about the source of a received packet.
///
/// # Purpose
/// Provides information about where a packet came from,
/// used for routing responses and session management.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PacketSource {
    /// Source address (IP and port).
    pub addr: SocketAddr,
    /// When the packet was received.
    pub timestamp: Instant,
}

impl PacketSource {
    /// Creates a new `PacketSource`.
    #[must_use]
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            timestamp: Instant::now(),
        }
    }

    /// Creates a `PacketSource` with a specific timestamp.
    #[must_use]
    pub const fn with_timestamp(addr: SocketAddr, timestamp: Instant) -> Self {
        Self { addr, timestamp }
    }

    /// Returns the age of this packet (time since received).
    #[must_use]
    pub fn age(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }
}

// ============================================
// Transport Trait
// ============================================

/// Abstract interface for datagram-based transport.
///
/// # Purpose
/// Provides a unified interface for sending and receiving packets,
/// abstracting over the underlying transport mechanism (UDP, etc.).
///
/// # Thread Safety
/// Implementations must be `Send + Sync` to allow sharing across
/// async tasks.
///
/// # Example
/// ```ignore
/// async fn handle_packets<T: Transport>(transport: &T) -> Result<()> {
///     let mut buf = [0u8; 1500];
///     loop {
///         let (len, source) = transport.recv(&mut buf).await?;
///         let response = process_packet(&buf[..len]);
///         transport.send(&response, &source.addr).await?;
///     }
/// }
/// ```
#[async_trait]
pub trait Transport: Send + Sync {
    /// Receives a packet from the transport.
    ///
    /// # Arguments
    /// * `buf` - Buffer to store received data
    ///
    /// # Returns
    /// Tuple of (bytes received, packet source)
    ///
    /// # Errors
    /// Returns error if receive fails
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, PacketSource)>;

    /// Sends a packet via the transport.
    ///
    /// # Arguments
    /// * `buf` - Data to send
    /// * `dest` - Destination address
    ///
    /// # Returns
    /// Number of bytes sent
    ///
    /// # Errors
    /// Returns error if send fails
    async fn send(&self, buf: &[u8], dest: &SocketAddr) -> Result<usize>;

    /// Returns the local address this transport is bound to.
    ///
    /// # Errors
    /// Returns error if address cannot be determined
    fn local_addr(&self) -> Result<SocketAddr>;

    /// Gracefully shuts down the transport.
    ///
    /// After shutdown, all operations will return errors.
    ///
    /// # Errors
    /// Returns error if shutdown fails
    async fn shutdown(&self) -> Result<()>;

    /// Returns `true` if the transport is still active.
    fn is_active(&self) -> bool;
}

// ============================================
// TunDevice Trait
// ============================================

/// Abstract interface for TUN device operations.
///
/// # Purpose
/// Provides a unified interface for reading and writing IP packets
/// to/from a virtual network device.
///
/// # Data Format
/// Data read from and written to the TUN device is raw IP packets
/// (no Ethernet headers).
///
/// # Example
/// ```ignore
/// async fn forward_packets<T: TunDevice>(tun: &T) -> Result<()> {
///     let mut buf = [0u8; 1500];
///     loop {
///         let len = tun.read(&mut buf).await?;
///         // Process IP packet in buf[..len]
///     }
/// }
/// ```
#[async_trait]
pub trait TunDevice: Send + Sync {
    /// Reads an IP packet from the TUN device.
    ///
    /// # Arguments
    /// * `buf` - Buffer to store the IP packet
    ///
    /// # Returns
    /// Number of bytes read
    ///
    /// # Errors
    /// Returns error if read fails
    async fn read(&self, buf: &mut [u8]) -> Result<usize>;

    /// Writes an IP packet to the TUN device.
    ///
    /// # Arguments
    /// * `buf` - IP packet data to write
    ///
    /// # Returns
    /// Number of bytes written
    ///
    /// # Errors
    /// Returns error if write fails
    async fn write(&self, buf: &[u8]) -> Result<usize>;

    /// Returns the device name.
    fn name(&self) -> &str;

    /// Returns the MTU (Maximum Transmission Unit).
    fn mtu(&self) -> u16;

    /// Returns the device's assigned IP address.
    fn ip_addr(&self) -> std::net::Ipv4Addr;

    /// Returns the network mask.
    fn netmask(&self) -> std::net::Ipv4Addr;

    /// Brings the device up (activates it).
    ///
    /// # Errors
    /// Returns error if activation fails
    async fn up(&self) -> Result<()>;

    /// Brings the device down (deactivates it).
    ///
    /// # Errors
    /// Returns error if deactivation fails
    async fn down(&self) -> Result<()>;

    /// Returns `true` if the device is up and active.
    fn is_up(&self) -> bool;
}

// ============================================
// TunConfig
// ============================================

/// Configuration for TUN device creation.
///
/// # Example
/// ```
/// use aeronyx_transport::traits::TunConfig;
/// use std::net::Ipv4Addr;
///
/// let config = TunConfig::new("aeronyx0")
///     .with_address(Ipv4Addr::new(100, 64, 0, 1))
///     .with_netmask(Ipv4Addr::new(255, 255, 255, 0))
///     .with_mtu(1420);
/// ```
#[derive(Debug, Clone)]
pub struct TunConfig {
    /// Device name (e.g., "tun0", "aeronyx0").
    pub name: String,
    /// IP address to assign to the device.
    pub address: std::net::Ipv4Addr,
    /// Network mask.
    pub netmask: std::net::Ipv4Addr,
    /// MTU size.
    pub mtu: u16,
    /// Whether to persist the device after process exit.
    pub persist: bool,
}

impl TunConfig {
    /// Creates a new TUN configuration with defaults.
    ///
    /// # Arguments
    /// * `name` - Device name
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            address: std::net::Ipv4Addr::new(100, 64, 0, 1),
            netmask: std::net::Ipv4Addr::new(255, 255, 255, 0),
            mtu: 1420,
            persist: false,
        }
    }

    /// Sets the IP address.
    #[must_use]
    pub const fn with_address(mut self, address: std::net::Ipv4Addr) -> Self {
        self.address = address;
        self
    }

    /// Sets the network mask.
    #[must_use]
    pub const fn with_netmask(mut self, netmask: std::net::Ipv4Addr) -> Self {
        self.netmask = netmask;
        self
    }

    /// Sets the MTU.
    #[must_use]
    pub const fn with_mtu(mut self, mtu: u16) -> Self {
        self.mtu = mtu;
        self
    }

    /// Sets whether the device should persist.
    #[must_use]
    pub const fn with_persist(mut self, persist: bool) -> Self {
        self.persist = persist;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    /// Returns error if configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        use crate::error::TransportError;

        if self.name.is_empty() {
            return Err(TransportError::invalid_config(
                "name",
                "device name cannot be empty",
            ));
        }

        if self.name.len() > 15 {
            return Err(TransportError::invalid_config(
                "name",
                "device name cannot exceed 15 characters",
            ));
        }

        if self.mtu < 576 {
            return Err(TransportError::invalid_config(
                "mtu",
                "MTU must be at least 576 bytes",
            ));
        }

        if self.mtu > 9000 {
            return Err(TransportError::invalid_config(
                "mtu",
                "MTU cannot exceed 9000 bytes",
            ));
        }

        Ok(())
    }
}

impl Default for TunConfig {
    fn default() -> Self {
        Self::new("aeronyx0")
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_source() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let source = PacketSource::new(addr);
        
        assert_eq!(source.addr, addr);
        assert!(source.age() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn test_tun_config_defaults() {
        let config = TunConfig::new("tun0");
        
        assert_eq!(config.name, "tun0");
        assert_eq!(config.address, std::net::Ipv4Addr::new(100, 64, 0, 1));
        assert_eq!(config.mtu, 1420);
    }

    #[test]
    fn test_tun_config_builder() {
        let config = TunConfig::new("test0")
            .with_address(std::net::Ipv4Addr::new(10, 0, 0, 1))
            .with_netmask(std::net::Ipv4Addr::new(255, 255, 0, 0))
            .with_mtu(1500)
            .with_persist(true);

        assert_eq!(config.address, std::net::Ipv4Addr::new(10, 0, 0, 1));
        assert_eq!(config.netmask, std::net::Ipv4Addr::new(255, 255, 0, 0));
        assert_eq!(config.mtu, 1500);
        assert!(config.persist);
    }

    #[test]
    fn test_tun_config_validation() {
        // Valid config
        let config = TunConfig::new("tun0");
        assert!(config.validate().is_ok());

        // Empty name
        let config = TunConfig::new("");
        assert!(config.validate().is_err());

        // Name too long
        let config = TunConfig::new("a".repeat(20));
        assert!(config.validate().is_err());

        // MTU too small
        let config = TunConfig::new("tun0").with_mtu(100);
        assert!(config.validate().is_err());

        // MTU too large
        let config = TunConfig::new("tun0").with_mtu(10000);
        assert!(config.validate().is_err());
    }
}
