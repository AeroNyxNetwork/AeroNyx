// ============================================
// File: crates/aeronyx-transport/src/tun/mock.rs
// ============================================
//! # Mock TUN Device Implementation
//!
//! ## Creation Reason
//! Provides a mock TUN device for testing without requiring
//! actual network device creation or root privileges.
//!
//! ## Main Functionality
//! - In-memory packet queues
//! - Simulated read/write operations
//! - Configurable behavior for testing
//! - No system privileges required
//!
//! ## Usage in Tests
//! ```
//! use aeronyx_transport::tun::MockTun;
//! use aeronyx_transport::traits::TunConfig;
//!
//! #[tokio::test]
//! async fn test_with_mock_tun() {
//!     let tun = MockTun::new(TunConfig::new("mock0"));
//!     
//!     // Inject a packet to be read
//!     tun.inject_packet(b"test packet".to_vec()).await;
//!     
//!     // Read it back
//!     let mut buf = [0u8; 1500];
//!     let len = tun.read(&mut buf).await.unwrap();
//!     assert_eq!(&buf[..len], b"test packet");
//! }
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - This is for testing only - do not use in production
//! - Packet queues are bounded to prevent memory issues
//! - Simulates async behavior but runs synchronously
//!
//! ## Last Modified
//! v0.1.0 - Initial mock implementation

use std::collections::VecDeque;
use std::net::Ipv4Addr;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::error::{Result, TransportError};
use crate::traits::{TunConfig, TunDevice};

// ============================================
// Constants
// ============================================

/// Maximum number of packets to queue.
const MAX_QUEUE_SIZE: usize = 1000;

// ============================================
// MockTun
// ============================================

/// Mock TUN device for testing.
///
/// # Features
/// - In-memory packet queues
/// - No system privileges required
/// - Configurable packet injection
/// - Packet capture for verification
///
/// # Example
/// ```
/// use aeronyx_transport::tun::MockTun;
/// use aeronyx_transport::traits::{TunConfig, TunDevice};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tun = MockTun::new(TunConfig::new("mock0"));
///
/// // Test writing
/// tun.write(b"test packet").await?;
///
/// // Verify packet was captured
/// let captured = tun.take_written_packets();
/// assert_eq!(captured.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct MockTun {
    /// Device configuration
    config: TunConfig,
    /// Packets waiting to be read (injected for testing)
    read_queue: Mutex<VecDeque<Vec<u8>>>,
    /// Packets that have been written (captured for verification)
    write_queue: Mutex<VecDeque<Vec<u8>>>,
    /// Whether the device is up
    is_up: AtomicBool,
    /// Notify when new packets are available
    read_notify: Notify,
}

impl MockTun {
    /// Creates a new mock TUN device.
    ///
    /// # Arguments
    /// * `config` - Device configuration
    #[must_use]
    pub fn new(config: TunConfig) -> Self {
        Self {
            config,
            read_queue: Mutex::new(VecDeque::with_capacity(100)),
            write_queue: Mutex::new(VecDeque::with_capacity(100)),
            is_up: AtomicBool::new(false),
            read_notify: Notify::new(),
        }
    }

    /// Injects a packet to be returned by the next `read()` call.
    ///
    /// # Arguments
    /// * `packet` - Packet data to inject
    ///
    /// # Panics
    /// Panics if the queue is full (> MAX_QUEUE_SIZE packets).
    pub async fn inject_packet(&self, packet: Vec<u8>) {
        let mut queue = self.read_queue.lock();
        if queue.len() >= MAX_QUEUE_SIZE {
            panic!("Mock TUN read queue overflow");
        }
        queue.push_back(packet);
        drop(queue);
        self.read_notify.notify_one();
    }

    /// Injects multiple packets at once.
    pub async fn inject_packets(&self, packets: Vec<Vec<u8>>) {
        let mut queue = self.read_queue.lock();
        for packet in packets {
            if queue.len() >= MAX_QUEUE_SIZE {
                panic!("Mock TUN read queue overflow");
            }
            queue.push_back(packet);
        }
        drop(queue);
        self.read_notify.notify_waiters();
    }

    /// Takes all packets that have been written to the device.
    ///
    /// This clears the write queue.
    #[must_use]
    pub fn take_written_packets(&self) -> Vec<Vec<u8>> {
        let mut queue = self.write_queue.lock();
        queue.drain(..).collect()
    }

    /// Returns the number of packets waiting to be read.
    #[must_use]
    pub fn pending_read_count(&self) -> usize {
        self.read_queue.lock().len()
    }

    /// Returns the number of packets that have been written.
    #[must_use]
    pub fn written_count(&self) -> usize {
        self.write_queue.lock().len()
    }

    /// Clears all queues.
    pub fn clear(&self) {
        self.read_queue.lock().clear();
        self.write_queue.lock().clear();
    }
}

#[async_trait]
impl TunDevice for MockTun {
    async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        loop {
            // Try to get a packet from the queue
            {
                let mut queue = self.read_queue.lock();
                if let Some(packet) = queue.pop_front() {
                    let len = packet.len().min(buf.len());
                    buf[..len].copy_from_slice(&packet[..len]);
                    return Ok(len);
                }
            }

            // Wait for a packet to be injected
            self.read_notify.notified().await;
        }
    }

    async fn write(&self, buf: &[u8]) -> Result<usize> {
        let mut queue = self.write_queue.lock();
        if queue.len() >= MAX_QUEUE_SIZE {
            return Err(TransportError::TunWriteFailed {
                reason: "Write queue full".into(),
            });
        }
        queue.push_back(buf.to_vec());
        Ok(buf.len())
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
        self.is_up.store(true, Ordering::Release);
        Ok(())
    }

    async fn down(&self) -> Result<()> {
        self.is_up.store(false, Ordering::Release);
        Ok(())
    }

    fn is_up(&self) -> bool {
        self.is_up.load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for MockTun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockTun")
            .field("name", &self.config.name)
            .field("address", &self.config.address)
            .field("mtu", &self.config.mtu)
            .field("is_up", &self.is_up())
            .field("pending_reads", &self.pending_read_count())
            .field("written_packets", &self.written_count())
            .finish()
    }
}

impl Default for MockTun {
    fn default() -> Self {
        Self::new(TunConfig::default())
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_tun_basic() {
        let tun = MockTun::new(TunConfig::new("mock0"));
        
        assert_eq!(tun.name(), "mock0");
        assert!(!tun.is_up());
    }

    #[tokio::test]
    async fn test_mock_tun_up_down() {
        let tun = MockTun::new(TunConfig::new("mock0"));
        
        assert!(!tun.is_up());
        
        tun.up().await.unwrap();
        assert!(tun.is_up());
        
        tun.down().await.unwrap();
        assert!(!tun.is_up());
    }

    #[tokio::test]
    async fn test_mock_tun_inject_read() {
        let tun = MockTun::new(TunConfig::new("mock0"));
        
        // Inject packet
        tun.inject_packet(b"test packet".to_vec()).await;
        assert_eq!(tun.pending_read_count(), 1);
        
        // Read it back
        let mut buf = [0u8; 100];
        let len = tun.read(&mut buf).await.unwrap();
        
        assert_eq!(&buf[..len], b"test packet");
        assert_eq!(tun.pending_read_count(), 0);
    }

    #[tokio::test]
    async fn test_mock_tun_write_capture() {
        let tun = MockTun::new(TunConfig::new("mock0"));
        
        // Write packets
        tun.write(b"packet 1").await.unwrap();
        tun.write(b"packet 2").await.unwrap();
        assert_eq!(tun.written_count(), 2);
        
        // Capture them
        let captured = tun.take_written_packets();
        
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0], b"packet 1");
        assert_eq!(captured[1], b"packet 2");
        assert_eq!(tun.written_count(), 0);
    }

    #[tokio::test]
    async fn test_mock_tun_multiple_packets() {
        let tun = MockTun::new(TunConfig::new("mock0"));
        
        let packets = vec![
            b"packet 1".to_vec(),
            b"packet 2".to_vec(),
            b"packet 3".to_vec(),
        ];
        
        tun.inject_packets(packets).await;
        assert_eq!(tun.pending_read_count(), 3);
        
        let mut buf = [0u8; 100];
        
        let len = tun.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"packet 1");
        
        let len = tun.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"packet 2");
        
        let len = tun.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..len], b"packet 3");
    }

    #[tokio::test]
    async fn test_mock_tun_clear() {
        let tun = MockTun::new(TunConfig::new("mock0"));
        
        tun.inject_packet(b"test".to_vec()).await;
        tun.write(b"test").await.unwrap();
        
        assert_eq!(tun.pending_read_count(), 1);
        assert_eq!(tun.written_count(), 1);
        
        tun.clear();
        
        assert_eq!(tun.pending_read_count(), 0);
        assert_eq!(tun.written_count(), 0);
    }

    #[tokio::test]
    async fn test_mock_tun_buffer_truncation() {
        let tun = MockTun::new(TunConfig::new("mock0"));
        
        // Inject large packet
        tun.inject_packet(vec![0x42; 1000]).await;
        
        // Read into small buffer
        let mut buf = [0u8; 10];
        let len = tun.read(&mut buf).await.unwrap();
        
        // Should be truncated
        assert_eq!(len, 10);
        assert_eq!(buf, [0x42; 10]);
    }
}
