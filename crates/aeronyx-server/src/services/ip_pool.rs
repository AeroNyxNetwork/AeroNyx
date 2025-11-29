// ============================================
// File: crates/aeronyx-server/src/services/ip_pool.rs
// ============================================
//! # IP Address Pool Service
//!
//! ## Creation Reason
//! Manages allocation of virtual IP addresses for connected clients,
//! ensuring each client gets a unique address from the configured range.
//!
//! ## Main Functionality
//! - `IpPoolService`: IP address pool management
//! - Address allocation and release
//! - Range parsing and validation
//! - Thread-safe operations
//!
//! ## IP Allocation Strategy
//! - Uses a bitmap to track allocated addresses
//! - Skips network address (.0) and gateway address
//! - First-fit allocation for simplicity
//! - O(n) allocation, O(1) release
//!
//! ## Example
//! ```
//! use aeronyx_server::services::IpPoolService;
//! use std::net::Ipv4Addr;
//!
//! let pool = IpPoolService::new(
//!     Ipv4Addr::new(100, 64, 0, 0),
//!     24,
//!     Ipv4Addr::new(100, 64, 0, 1),
//! ).unwrap();
//!
//! let ip = pool.allocate().unwrap();
//! // ip is now 100.64.0.2 (first available after gateway)
//!
//! pool.release(ip);
//! // ip is now available again
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Gateway IP is reserved and never allocated
//! - Network address (.0) is reserved
//! - Broadcast address (.255 for /24) is reserved
//! - Consider switching to more efficient bitmap for large ranges
//!
//! ## Last Modified
//! v0.1.0 - Initial IP pool implementation

use std::collections::HashSet;
use std::net::Ipv4Addr;

use parking_lot::Mutex;
use tracing::{debug, warn};

use crate::error::{Result, ServerError};

// ============================================
// IpPoolService
// ============================================

/// Virtual IP address pool service.
///
/// # Thread Safety
/// Uses internal locking for thread-safe operations.
///
/// # Capacity
/// For a /24 network: 254 addresses (256 - network - broadcast)
/// Minus gateway: 253 available addresses
pub struct IpPoolService {
    /// Network address (e.g., 100.64.0.0)
    network: Ipv4Addr,
    /// Prefix length (e.g., 24)
    prefix_len: u8,
    /// Gateway address (reserved)
    gateway: Ipv4Addr,
    /// Set of allocated addresses
    allocated: Mutex<HashSet<Ipv4Addr>>,
    /// Total addresses in range
    total_addresses: u32,
    /// First usable address offset
    first_usable: u32,
    /// Last usable address offset
    last_usable: u32,
}

impl IpPoolService {
    /// Creates a new IP pool service.
    ///
    /// # Arguments
    /// * `network` - Network address (e.g., 100.64.0.0)
    /// * `prefix_len` - CIDR prefix length (e.g., 24)
    /// * `gateway` - Gateway address to reserve
    ///
    /// # Errors
    /// Returns error if configuration is invalid.
    pub fn new(network: Ipv4Addr, prefix_len: u8, gateway: Ipv4Addr) -> Result<Self> {
        // Validate prefix length
        if prefix_len > 30 {
            return Err(ServerError::config_invalid(
                "ip_pool.prefix_len",
                "prefix length must be <= 30 for usable addresses",
            ));
        }

        // Calculate address range
        let total_addresses = 1u32 << (32 - prefix_len);
        let first_usable = 1; // Skip network address
        let last_usable = total_addresses - 2; // Skip broadcast address

        // Validate gateway is in range
        let network_u32 = u32::from(network);
        let gateway_u32 = u32::from(gateway);
        let gateway_offset = gateway_u32.wrapping_sub(network_u32);

        if gateway_offset >= total_addresses {
            return Err(ServerError::config_invalid(
                "ip_pool.gateway",
                "gateway address is not in network range",
            ));
        }

        debug!(
            "IP pool initialized: {}/{}, gateway={}, available={}",
            network,
            prefix_len,
            gateway,
            last_usable - first_usable // -1 for gateway
        );

        Ok(Self {
            network,
            prefix_len,
            gateway,
            allocated: Mutex::new(HashSet::new()),
            total_addresses,
            first_usable,
            last_usable,
        })
    }

    /// Allocates a new virtual IP address.
    ///
    /// # Returns
    /// An available IP address from the pool.
    ///
    /// # Errors
    /// Returns `IpPoolExhausted` if no addresses are available.
    pub fn allocate(&self) -> Result<Ipv4Addr> {
        let mut allocated = self.allocated.lock();
        let network_u32 = u32::from(self.network);

        // Find first available address
        for offset in self.first_usable..=self.last_usable {
            let ip = Ipv4Addr::from(network_u32 + offset);

            // Skip gateway
            if ip == self.gateway {
                continue;
            }

            // Skip if already allocated
            if allocated.contains(&ip) {
                continue;
            }

            // Allocate this address
            allocated.insert(ip);
            debug!("Allocated IP: {} ({} in use)", ip, allocated.len());
            return Ok(ip);
        }

        warn!("IP pool exhausted ({} addresses in use)", allocated.len());
        Err(ServerError::IpPoolExhausted)
    }

    /// Releases a previously allocated IP address.
    ///
    /// # Arguments
    /// * `ip` - The IP address to release
    ///
    /// # Returns
    /// `true` if the address was released, `false` if it wasn't allocated.
    pub fn release(&self, ip: Ipv4Addr) -> bool {
        let mut allocated = self.allocated.lock();
        let removed = allocated.remove(&ip);
        
        if removed {
            debug!("Released IP: {} ({} in use)", ip, allocated.len());
        } else {
            warn!("Attempted to release unallocated IP: {}", ip);
        }

        removed
    }

    /// Checks if an IP address is currently allocated.
    #[must_use]
    pub fn is_allocated(&self, ip: Ipv4Addr) -> bool {
        self.allocated.lock().contains(&ip)
    }

    /// Returns the number of allocated addresses.
    #[must_use]
    pub fn allocated_count(&self) -> usize {
        self.allocated.lock().len()
    }

    /// Returns the number of available addresses.
    #[must_use]
    pub fn available_count(&self) -> usize {
        let total_usable = (self.last_usable - self.first_usable + 1) as usize;
        let gateway_reserved = 1;
        total_usable - gateway_reserved - self.allocated_count()
    }

    /// Returns the total capacity of the pool.
    #[must_use]
    pub fn capacity(&self) -> usize {
        let total_usable = (self.last_usable - self.first_usable + 1) as usize;
        let gateway_reserved = 1;
        total_usable - gateway_reserved
    }

    /// Returns the network address.
    #[must_use]
    pub const fn network(&self) -> Ipv4Addr {
        self.network
    }

    /// Returns the prefix length.
    #[must_use]
    pub const fn prefix_len(&self) -> u8 {
        self.prefix_len
    }

    /// Returns the gateway address.
    #[must_use]
    pub const fn gateway(&self) -> Ipv4Addr {
        self.gateway
    }

    /// Checks if an IP address is within this pool's range.
    #[must_use]
    pub fn contains(&self, ip: Ipv4Addr) -> bool {
        let network_u32 = u32::from(self.network);
        let ip_u32 = u32::from(ip);
        let offset = ip_u32.wrapping_sub(network_u32);
        
        offset >= self.first_usable && offset <= self.last_usable && ip != self.gateway
    }

    /// Clears all allocations (useful for testing).
    #[cfg(test)]
    pub fn clear(&self) {
        self.allocated.lock().clear();
    }
}

impl std::fmt::Debug for IpPoolService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpPoolService")
            .field("network", &format!("{}/{}", self.network, self.prefix_len))
            .field("gateway", &self.gateway)
            .field("allocated", &self.allocated_count())
            .field("available", &self.available_count())
            .field("capacity", &self.capacity())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool() -> IpPoolService {
        IpPoolService::new(
            Ipv4Addr::new(100, 64, 0, 0),
            24,
            Ipv4Addr::new(100, 64, 0, 1),
        )
        .unwrap()
    }

    #[test]
    fn test_pool_creation() {
        let pool = create_test_pool();
        
        assert_eq!(pool.network(), Ipv4Addr::new(100, 64, 0, 0));
        assert_eq!(pool.prefix_len(), 24);
        assert_eq!(pool.gateway(), Ipv4Addr::new(100, 64, 0, 1));
        assert_eq!(pool.allocated_count(), 0);
        // 254 total - 1 gateway = 253
        assert_eq!(pool.capacity(), 253);
    }

    #[test]
    fn test_allocate_first() {
        let pool = create_test_pool();
        
        let ip = pool.allocate().unwrap();
        
        // First allocation should be .2 (skip .0 network and .1 gateway)
        assert_eq!(ip, Ipv4Addr::new(100, 64, 0, 2));
        assert_eq!(pool.allocated_count(), 1);
        assert!(pool.is_allocated(ip));
    }

    #[test]
    fn test_allocate_multiple() {
        let pool = create_test_pool();
        
        let ip1 = pool.allocate().unwrap();
        let ip2 = pool.allocate().unwrap();
        let ip3 = pool.allocate().unwrap();
        
        assert_eq!(ip1, Ipv4Addr::new(100, 64, 0, 2));
        assert_eq!(ip2, Ipv4Addr::new(100, 64, 0, 3));
        assert_eq!(ip3, Ipv4Addr::new(100, 64, 0, 4));
        assert_eq!(pool.allocated_count(), 3);
    }

    #[test]
    fn test_release() {
        let pool = create_test_pool();
        
        let ip = pool.allocate().unwrap();
        assert!(pool.is_allocated(ip));
        
        let released = pool.release(ip);
        assert!(released);
        assert!(!pool.is_allocated(ip));
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn test_release_unallocated() {
        let pool = create_test_pool();
        
        let released = pool.release(Ipv4Addr::new(100, 64, 0, 100));
        assert!(!released);
    }

    #[test]
    fn test_reallocation_after_release() {
        let pool = create_test_pool();
        
        let ip1 = pool.allocate().unwrap();
        let ip2 = pool.allocate().unwrap();
        
        pool.release(ip1);
        
        // Next allocation should reuse ip1
        let ip3 = pool.allocate().unwrap();
        assert_eq!(ip3, ip1);
    }

    #[test]
    fn test_pool_exhaustion() {
        // Use a /30 network (4 addresses: 1 network, 1 broadcast, 1 gateway, 1 usable)
        let pool = IpPoolService::new(
            Ipv4Addr::new(10, 0, 0, 0),
            30,
            Ipv4Addr::new(10, 0, 0, 1),
        )
        .unwrap();

        // Should be able to allocate 1 address (.2)
        let ip = pool.allocate().unwrap();
        assert_eq!(ip, Ipv4Addr::new(10, 0, 0, 2));

        // Pool should now be exhausted
        let result = pool.allocate();
        assert!(matches!(result, Err(ServerError::IpPoolExhausted)));
    }

    #[test]
    fn test_contains() {
        let pool = create_test_pool();
        
        // Valid addresses
        assert!(pool.contains(Ipv4Addr::new(100, 64, 0, 2)));
        assert!(pool.contains(Ipv4Addr::new(100, 64, 0, 254)));
        
        // Gateway is not allocatable
        assert!(!pool.contains(Ipv4Addr::new(100, 64, 0, 1)));
        
        // Outside range
        assert!(!pool.contains(Ipv4Addr::new(100, 64, 1, 1)));
        assert!(!pool.contains(Ipv4Addr::new(192, 168, 0, 1)));
    }

    #[test]
    fn test_invalid_prefix() {
        let result = IpPoolService::new(
            Ipv4Addr::new(10, 0, 0, 0),
            31, // Too small
            Ipv4Addr::new(10, 0, 0, 1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gateway_outside_range() {
        let result = IpPoolService::new(
            Ipv4Addr::new(10, 0, 0, 0),
            24,
            Ipv4Addr::new(192, 168, 0, 1), // Outside 10.0.0.0/24
        );
        assert!(result.is_err());
    }
}
