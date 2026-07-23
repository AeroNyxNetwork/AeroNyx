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
//! - Cancellation of in-flight socket operations during shutdown
//!
//! ## Design Choices
//! - Uses `SO_REUSEADDR` for quick rebinding after restart
//! - Non-blocking operations with Tokio
//! - Tokio watch channel for race-free shutdown notification
//!
//! ## ⚠️ Important Note for Next Developer
//! - UDP is connectionless - no guaranteed delivery
//! - Maximum UDP payload is ~65507 bytes
//! - Consider firewall rules when binding to public addresses
//! - Keep socket operations inside `run_until_shutdown` so future operations
//!   preserve the in-flight cancellation contract
//!
//! ## Last Modified
//! v0.2.0 - Added race-free in-flight I/O cancellation and removed the
//!          redundant internal socket `Arc`.
//! v0.1.1 - Removed a stale tracing import after transport lint review.
//! v0.1.0 - Initial UDP transport implementation

use std::future::Future;
use std::net::SocketAddr;

use async_trait::async_trait;
use socket2::{Domain, Protocol, Socket, Type};
use tokio::net::UdpSocket;
use tokio::sync::watch;
use tracing::{debug, info, trace};

use crate::error::{Result, TransportError};
use crate::traits::{PacketSource, Transport};

// ============================================
// UdpTransport
// ============================================

/// One-way, process-local shutdown signal shared by all in-flight operations.
#[derive(Debug)]
struct ShutdownSignal {
    sender: watch::Sender<bool>,
}

impl ShutdownSignal {
    /// Creates an active signal.
    fn new() -> Self {
        let (sender, _receiver) = watch::channel(false);
        Self { sender }
    }

    /// Returns whether shutdown has been requested.
    fn is_triggered(&self) -> bool {
        *self.sender.borrow()
    }

    /// Triggers shutdown and returns `true` only for the first caller.
    fn trigger(&self) -> bool {
        !self.sender.send_replace(true)
    }

    /// Waits until shutdown is requested.
    async fn cancelled(&self) {
        let mut receiver = self.sender.subscribe();
        let _closed = receiver.wait_for(|is_shutdown| *is_shutdown).await;
    }
}

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
    socket: UdpSocket,
    /// Local address we're bound to
    local_addr: SocketAddr,
    /// Race-free shutdown notification for current and future operations
    shutdown: ShutdownSignal,
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
        let socket_addr: SocketAddr =
            addr_str
                .parse()
                .map_err(|_| TransportError::InvalidAddress {
                    addr: addr_str.to_string(),
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
    #[allow(clippy::unused_async)]
    pub async fn bind_addr(addr: SocketAddr) -> Result<Self> {
        // [UDP-BIND-COMPAT 2026-07-23 by Codex] This public method remains
        // async for API compatibility. Socket creation is non-blocking and
        // must complete before Tokio can safely adopt the descriptor.
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
        socket.bind(&addr.into()).map_err(|e| {
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
            socket: tokio_socket,
            local_addr,
            shutdown: ShutdownSignal::new(),
        })
    }

    /// Returns the number of bytes available to read.
    ///
    /// This is an estimate and may not be exact.
    ///
    /// # Errors
    /// This compatibility method currently does not return an error.
    pub const fn readable_bytes(&self) -> Result<usize> {
        // Note: This is not directly available in Tokio UdpSocket
        // We could use ioctl FIONREAD, but for now just return 0
        Ok(0)
    }

    /// Checks if the transport has been shut down.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.is_triggered()
    }

    /// Runs one cancel-safe socket future until completion or shutdown.
    ///
    /// Tokio UDP send/receive futures are cancel-safe. The final state check
    /// closes the narrow race where both branches become ready together.
    async fn run_until_shutdown<T>(&self, operation: impl Future<Output = Result<T>>) -> Result<T> {
        // [UDP-SHUTDOWN 2026-07-23 by Codex] A watch channel retains the
        // shutdown value, so an operation cannot miss notification between
        // its initial state check and registration with `select!`.
        if self.is_shutdown() {
            return Err(TransportError::ShuttingDown);
        }

        let result = tokio::select! {
            biased;
            () = self.shutdown.cancelled() => Err(TransportError::ShuttingDown),
            result = operation => result,
        }?;

        if self.is_shutdown() {
            Err(TransportError::ShuttingDown)
        } else {
            Ok(result)
        }
    }
}

#[async_trait]
impl Transport for UdpTransport {
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, PacketSource)> {
        let receive = async {
            self.socket
                .recv_from(buf)
                .await
                .map_err(|error| TransportError::ReceiveFailed {
                    reason: error.to_string(),
                })
        };
        let (len, addr) = self.run_until_shutdown(receive).await?;

        trace!("Received {} bytes from {}", len, addr);

        Ok((len, PacketSource::new(addr)))
    }

    async fn send(&self, buf: &[u8], dest: &SocketAddr) -> Result<usize> {
        let send = async {
            self.socket
                .send_to(buf, dest)
                .await
                .map_err(|error| TransportError::SendFailed {
                    dest: *dest,
                    reason: error.to_string(),
                })
        };
        let len = self.run_until_shutdown(send).await?;

        trace!("Sent {} bytes to {}", len, dest);

        Ok(len)
    }

    fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.local_addr)
    }

    async fn shutdown(&self) -> Result<()> {
        if self.shutdown.trigger() {
            info!("UDP transport shutdown requested");
        } else {
            debug!("UDP transport shutdown already requested");
        }

        // Note: Tokio UdpSocket doesn't have an explicit close method
        // The socket will be closed when dropped

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
            .finish_non_exhaustive()
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
        let local_addr = transport.local_addr().unwrap();

        assert!(transport.is_active());

        transport.shutdown().await.unwrap();
        transport.shutdown().await.unwrap();

        assert!(!transport.is_active());
        assert!(transport.is_shutdown());

        // Operations should fail after shutdown
        let mut buf = [0u8; 1024];
        let result = transport.recv(&mut buf).await;
        assert!(matches!(result, Err(TransportError::ShuttingDown)));

        let result = transport.send(b"closed", &local_addr).await;
        assert!(matches!(result, Err(TransportError::ShuttingDown)));
    }

    #[tokio::test]
    async fn test_shutdown_cancels_all_pending_receivers() {
        use std::sync::Arc;
        use std::time::Duration;

        const RECEIVER_COUNT: usize = 8;

        let transport = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let mut receivers = Vec::with_capacity(RECEIVER_COUNT);

        for _ in 0..RECEIVER_COUNT {
            let transport = Arc::clone(&transport);
            receivers.push(tokio::spawn(async move {
                let mut buf = [0u8; 64];
                transport.recv(&mut buf).await
            }));
        }

        tokio::task::yield_now().await;
        transport.shutdown().await.unwrap();

        for receiver in receivers {
            let result = tokio::time::timeout(Duration::from_secs(1), receiver)
                .await
                .expect("pending receive should wake during shutdown")
                .expect("receive task should not panic");
            assert!(matches!(result, Err(TransportError::ShuttingDown)));
        }
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

        // UDP has no TCP TIME_WAIT state, so rebinding after dropping the
        // previous owner is deterministic.
        UdpTransport::bind_addr(addr)
            .await
            .expect("released UDP address should be immediately reusable");
    }
}
