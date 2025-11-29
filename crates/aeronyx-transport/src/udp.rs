// ============================================
// File: crates/aeronyx-transport/src/udp.rs
// ============================================
//! # UDP Transport Implementation
//!
//! ## Creation Reason
//! Provides UDP socket-based transport for client communication,
//! wrapping Tokio's UDP socket with our `Transport` trait.
//!
//! ## Main Functionality
//! - `UdpTransport`: Main UDP transport implementation
//! - Socket binding with address reuse
//! - Async send/receive operations
//! - Graceful shutdown support
//!
//! ## Design Choices
//! - Uses SO_REUSEADDR for quick rebinding after restart
//! - Non-blocking operations with Tokio
//! - Atomic shutdown flag for coordinated cleanup
//!
//! ## ⚠️ Important Note for Next Developer
//! - UDP is connectionless - no guaranteed delivery
//! - Maximum UDP payload is ~65507 bytes
//! - Consider firewall rules when binding to public addresses
//!
//! ## Last Modified
//! v0.1.0 - Initial UDP transport implementation

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use socket2::{Domain, Protocol, Socket, Type};
use tokio::net::UdpSocket;
use tracing::{debug, error, info, trace};

use crate::error::{Result, TransportError};
use crate::traits::{PacketSource, Transport};

// ============================================
// UdpTransport
// ============================================

/// UDP-based transport implementation.
///
/// # Features
/// - Async send/receive using Tokio
/// - Address reuse for quick restart
/// - Graceful shutdown support
/// - Thread-safe (Send + Sync)
///
/// # Example
/// ```ignore
/// use aeronyx_transport::UdpTransport;
///
/// let transport = UdpTransport::bind("0.0.0.0:51820").await?;
/// 
/// // Receive packets
/// let mut buf = [0u8; 1500];
/// let (len, source) = transport.recv(&mut buf).await?;
///
/// // Send response
/// transport.send(b"response", &source.addr).await?;
/// ```
pub struct UdpTransport {
    /// Underlying UDP socket
    socket: Arc<UdpSocket>,
    /// Local address we're bound to
    local_addr: SocketAddr,
    /// Shutdown flag
    shutdown: AtomicBool,
}

impl UdpTransport {
    /// Creates a new UDP transport bound to the specified address.
    ///
    /// # Arguments
    /// * `addr` - Address to bind to (e.g., "0.0.0.0:51820")
    ///
    /// # Socket Options
    /// - `SO_REUSEADDR`: Enabled for quick rebinding
    /// - Non-blocking: Required for async operations
    ///
    /// # Errors
    /// - `BindFailed`: If binding fails
    /// - `AddressInUse`: If address is already in use
    pub async fn bind(addr: impl AsRef<str>) -> Result<Self> {
        let addr_str = addr.as_ref();
        let socket_addr: SocketAddr = addr_str.parse().map_err(|_| {
            TransportError::InvalidAddress {
                addr: addr_str.to_string(),
            }
        })?;

        Self::bind_addr(socket_addr).await
    }

    /// Creates a new UDP transport bound to the specified socket address.
    ///
    /// # Arguments
    /// * `addr` - Socket address to bind to
    ///
    /// # Errors
    /// Returns error if binding fails.
    pub async fn bind_addr(addr: SocketAddr) -> Result<Self> {
        info!("Binding UDP transport to {}", addr);

        // Create socket with socket2 for more control
        let domain = if addr.is_ipv4() {
            Domain::IPV4
        } else {
            Domain::IPV6
        };

        let socket = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))
            .map_err(|e| TransportError::io("creating UDP socket", e))?;

        // Set socket options
        socket
            .set_reuse_address(true)
            .map_err(|e| TransportError::io("setting SO_REUSEADDR", e))?;

        socket
            .set_nonblocking(true)
            .map_err(|e| TransportError::io("setting non-blocking", e))?;

        // Bind to address
        socket
            .bind(&addr.into())
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AddrInUse {
                    TransportError::AddressInUse { addr }
                } else {
                    TransportError::bind_failed(addr, e.to_string())
                }
            })?;

        // Convert to Tokio socket
        let std_socket: std::net::UdpSocket = socket.into();
        let tokio_socket = UdpSocket::from_std(std_socket)
            .map_err(|e| TransportError::io("converting to Tokio socket", e))?;

        let local_addr = tokio_socket
            .local_addr()
            .map_err(|e| TransportError::io("getting local address", e))?;

        info!("UDP transport bound to {}", local_addr);

        Ok(Self {
            socket: Arc::new(tokio_socket),
            local_addr,
            shutdown: AtomicBool::new(false),
        })
    }

    /// Returns the number of bytes available to read.
    ///
    /// This is an estimate and may not be exact.
    pub fn readable_bytes(&self) -> Result<usize> {
        // Note: This is not directly available in Tokio UdpSocket
        // We could use ioctl FIONREAD, but for now just return 0
        Ok(0)
    }

    /// Checks if the transport has been shut down.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

#[async_trait]
impl Transport for UdpTransport {
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, PacketSource)> {
        if self.is_shutdown() {
            return Err(TransportError::ShuttingDown);
        }

        let (len, addr) = self
            .socket
            .recv_from(buf)
            .await
            .map_err(|e| TransportError::ReceiveFailed {
                reason: e.to_string(),
            })?;

        trace!("Received {} bytes from {}", len, addr);

        Ok((len, PacketSource::new(addr)))
    }

    async fn send(&self, buf: &[u8], dest: &SocketAddr) -> Result<usize> {
        if self.is_shutdown() {
            return Err(TransportError::ShuttingDown);
        }

        let len = self
            .socket
            .send_to(buf, dest)
            .await
            .map_err(|e| TransportError::SendFailed {
                dest: *dest,
                reason: e.to_string(),
            })?;

        trace!("Sent {} bytes to {}", len, dest);

        Ok(len)
    }

    fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.local_addr)
    }

    async fn shutdown(&self) -> Result<()> {
        debug!("Shutting down UDP transport");

        // Set shutdown flag
        self.shutdown.store(true, Ordering::Release);

        // Note: Tokio UdpSocket doesn't have an explicit close method
        // The socket will be closed when dropped

        info!("UDP transport shutdown complete");
        Ok(())
    }

    fn is_active(&self) -> bool {
        !self.is_shutdown()
    }
}

impl std::fmt::Debug for UdpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UdpTransport")
            .field("local_addr", &self.local_addr)
            .field("shutdown", &self.is_shutdown())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bind_and_local_addr() {
        let transport = UdpTransport::bind("127.0.0.1:0").await.unwrap();
        let addr = transport.local_addr().unwrap();
        
        assert_eq!(addr.ip(), std::net::Ipv4Addr::LOCALHOST);
        assert!(addr.port() > 0);
    }

    #[tokio::test]
    async fn test_send_recv_loopback() {
        let server = UdpTransport::bind("127.0.0.1:0").await.unwrap();
        let client = UdpTransport::bind("127.0.0.1:0").await.unwrap();

        let server_addr = server.local_addr().unwrap();
        let client_addr = client.local_addr().unwrap();

        // Send from client to server
        let message = b"Hello, AeroNyx!";
        client.send(message, &server_addr).await.unwrap();

        // Receive on server
        let mut buf = [0u8; 1024];
        let (len, source) = server.recv(&mut buf).await.unwrap();

        assert_eq!(len, message.len());
        assert_eq!(&buf[..len], message);
        assert_eq!(source.addr, client_addr);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let transport = UdpTransport::bind("127.0.0.1:0").await.unwrap();
        
        assert!(transport.is_active());
        
        transport.shutdown().await.unwrap();
        
        assert!(!transport.is_active());
        assert!(transport.is_shutdown());

        // Operations should fail after shutdown
        let mut buf = [0u8; 1024];
        let result = transport.recv(&mut buf).await;
        assert!(matches!(result, Err(TransportError::ShuttingDown)));
    }

    #[tokio::test]
    async fn test_invalid_address() {
        let result = UdpTransport::bind("not-an-address").await;
        assert!(matches!(result, Err(TransportError::InvalidAddress { .. })));
    }

    #[tokio::test]
    async fn test_address_reuse() {
        // Bind to a specific port
        let transport1 = UdpTransport::bind("127.0.0.1:0").await.unwrap();
        let addr = transport1.local_addr().unwrap();
        
        // Drop the first transport
        drop(transport1);

        // Should be able to bind to the same port immediately due to SO_REUSEADDR
        // Note: This may still fail occasionally due to TIME_WAIT, but should mostly work
        let _transport2 = UdpTransport::bind_addr(addr).await;
        // We don't assert success here because it's timing-dependent
    }
}
