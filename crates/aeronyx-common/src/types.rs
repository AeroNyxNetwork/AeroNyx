// ============================================
// File: crates/aeronyx-common/src/types.rs
// ============================================
//! # Core Type Definitions
//!
//! ## Creation Reason
//! Centralizes fundamental type definitions used throughout the AeroNyx
//! privacy network, ensuring type safety and consistent representations.
//!
//! ## Modification Notes
//! - Adapted for zeroize 1.3 (no ZeroizeOnDrop derive, manual Drop impl)
//! - Adapted for base64 0.21 API (DecodeError variants differ from 0.22)
//!
//! ## Main Functionality
//! - `SessionId`: Unique identifier for active sessions (16 bytes)
//! - `VirtualIp`: Wrapper for virtual IP addresses in the tunnel
//! - Type conversions and serialization implementations
//!
//! ## Main Logical Flow
//! 1. Types are created during handshake or configuration
//! 2. Used as keys in hashmaps and routing tables
//! 3. Serialized for network transmission
//! 4. Securely zeroed on drop for sensitive types
//!
//! ## ⚠️ Important Note for Next Developer
//! - SessionId is security-critical - always use cryptographically secure random
//! - SessionId implements Zeroize and manual Drop (zeroize 1.3 compatibility)
//! - zeroize 1.3 does NOT have ZeroizeOnDrop derive macro
//! - Maintain backward-compatible serialization formats
//!
//! ## Last Modified
//! v0.1.2 - Fixed: Compatibility with zeroize 1.3 and base64 0.21

use std::fmt;
use std::net::Ipv4Addr;
use std::str::FromStr;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

// ============================================
// Constants
// ============================================

/// Size of SessionId in bytes
pub const SESSION_ID_SIZE: usize = 16;

// ============================================
// SessionId Error Type
// ============================================

/// Error type for SessionId parsing failures
#[derive(Debug, Clone)]
pub enum SessionIdError {
    /// Base64 decoding failed
    InvalidBase64(String),
    /// Decoded bytes have wrong length
    InvalidLength { expected: usize, actual: usize },
}

impl fmt::Display for SessionIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBase64(msg) => write!(f, "Invalid base64: {}", msg),
            Self::InvalidLength { expected, actual } => {
                write!(f, "Invalid length: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for SessionIdError {}

// ============================================
// SessionId
// ============================================

/// Unique identifier for an active session.
///
/// # Security Properties
/// - Generated using cryptographically secure random number generator
/// - Fixed 16-byte size (128 bits of entropy)
/// - Implements `Zeroize` for secure memory cleanup
/// - Manual `Drop` implementation ensures zeroization (zeroize 1.3 compat)
/// - Does NOT implement `Copy` due to secure drop behavior
///
/// # Wire Format
/// ```text
/// ┌────────────────────────────────────┐
/// │       Session ID (16 bytes)        │
/// │  Used as packet routing identifier │
/// └────────────────────────────────────┘
/// ```
///
/// # Example
/// ```
/// use aeronyx_common::types::SessionId;
///
/// // Generate new random session ID
/// let session_id = SessionId::generate();
///
/// // Convert to/from bytes
/// let bytes = session_id.as_bytes();
/// let restored = SessionId::from_bytes(bytes).unwrap();
///
/// assert_eq!(session_id, restored);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Zeroize)]
pub struct SessionId([u8; SESSION_ID_SIZE]);

// Manual Drop implementation for secure zeroization
// Required because zeroize 1.3 does not have ZeroizeOnDrop derive
impl Drop for SessionId {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl SessionId {
    /// Creates a new `SessionId` from raw bytes.
    ///
    /// # Arguments
    /// * `bytes` - Exactly 16 bytes for the session ID
    ///
    /// # Returns
    /// - `Some(SessionId)` if bytes length is correct
    /// - `None` if bytes length is not 16
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != SESSION_ID_SIZE {
            return None;
        }
        let mut id = [0u8; SESSION_ID_SIZE];
        id.copy_from_slice(bytes);
        Some(Self(id))
    }

    /// Generates a new cryptographically random `SessionId`.
    ///
    /// Uses the system's secure random number generator.
    #[must_use]
    pub fn generate() -> Self {
        let mut id = [0u8; SESSION_ID_SIZE];
        rand::thread_rng().fill_bytes(&mut id);
        Self(id)
    }

    /// Returns the raw bytes of the session ID.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; SESSION_ID_SIZE] {
        &self.0
    }

    /// Creates a zeroed `SessionId` (useful for testing only).
    ///
    /// # Warning
    /// Do not use in production - a zero session ID is predictable.
    #[cfg(test)]
    #[must_use]
    pub const fn zero() -> Self {
        Self([0u8; SESSION_ID_SIZE])
    }
}

impl fmt::Debug for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only show first 4 bytes in debug output for privacy
        write!(
            f,
            "SessionId({:02x}{:02x}{:02x}{:02x}...)",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Base64 encode for human-readable display
        write!(f, "{}", BASE64.encode(self.0))
    }
}

impl FromStr for SessionId {
    type Err = SessionIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = BASE64.decode(s)
            .map_err(|e| SessionIdError::InvalidBase64(e.to_string()))?;
        
        Self::from_bytes(&bytes).ok_or_else(|| SessionIdError::InvalidLength {
            expected: SESSION_ID_SIZE,
            actual: bytes.len(),
        })
    }
}

impl Serialize for SessionId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            serializer.serialize_str(&BASE64.encode(self.0))
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            s.parse().map_err(serde::de::Error::custom)
        } else {
            let bytes = <Vec<u8>>::deserialize(deserializer)?;
            Self::from_bytes(&bytes).ok_or_else(|| {
                serde::de::Error::invalid_length(bytes.len(), &"16 bytes")
            })
        }
    }
}

impl AsRef<[u8]> for SessionId {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ============================================
// VirtualIp
// ============================================

/// Virtual IP address assigned to a client within the tunnel.
///
/// # Purpose
/// Wraps `Ipv4Addr` to provide type safety and prevent confusion
/// between physical and virtual IP addresses.
///
/// # Default Range
/// The default virtual IP range is `100.64.0.0/24` (Carrier-grade NAT range).
///
/// # Example
/// ```
/// use aeronyx_common::types::VirtualIp;
/// use std::net::Ipv4Addr;
///
/// let vip = VirtualIp::new(Ipv4Addr::new(100, 64, 0, 2));
/// assert_eq!(vip.to_string(), "100.64.0.2");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VirtualIp(Ipv4Addr);

impl VirtualIp {
    /// Creates a new `VirtualIp` from an `Ipv4Addr`.
    #[must_use]
    pub const fn new(addr: Ipv4Addr) -> Self {
        Self(addr)
    }

    /// Returns the underlying `Ipv4Addr`.
    #[must_use]
    pub const fn inner(&self) -> Ipv4Addr {
        self.0
    }

    /// Returns the IP address as a 4-byte array.
    #[must_use]
    pub const fn octets(&self) -> [u8; 4] {
        self.0.octets()
    }

    /// Creates a `VirtualIp` from 4 octets.
    #[must_use]
    pub const fn from_octets(octets: [u8; 4]) -> Self {
        Self(Ipv4Addr::new(octets[0], octets[1], octets[2], octets[3]))
    }

    /// Checks if this IP is within a given CIDR range.
    ///
    /// # Arguments
    /// * `network` - Network address
    /// * `prefix_len` - CIDR prefix length (0-32)
    ///
    /// # Example
    /// ```
    /// use aeronyx_common::types::VirtualIp;
    /// use std::net::Ipv4Addr;
    ///
    /// let vip = VirtualIp::new(Ipv4Addr::new(100, 64, 0, 50));
    /// assert!(vip.is_in_range(Ipv4Addr::new(100, 64, 0, 0), 24));
    /// assert!(!vip.is_in_range(Ipv4Addr::new(192, 168, 0, 0), 24));
    /// ```
    #[must_use]
    pub fn is_in_range(&self, network: Ipv4Addr, prefix_len: u8) -> bool {
        if prefix_len > 32 {
            return false;
        }
        let mask = if prefix_len == 0 {
            0
        } else {
            !0u32 << (32 - prefix_len)
        };
        let ip_bits = u32::from(self.0);
        let network_bits = u32::from(network);
        (ip_bits & mask) == (network_bits & mask)
    }
}

impl fmt::Display for VirtualIp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Ipv4Addr> for VirtualIp {
    fn from(addr: Ipv4Addr) -> Self {
        Self(addr)
    }
}

impl From<VirtualIp> for Ipv4Addr {
    fn from(vip: VirtualIp) -> Self {
        vip.0
    }
}

impl From<[u8; 4]> for VirtualIp {
    fn from(octets: [u8; 4]) -> Self {
        Self::from_octets(octets)
    }
}

// ============================================
// Counter
// ============================================

/// Monotonically increasing counter for replay protection.
///
/// Used as part of the nonce in ChaCha20-Poly1305 encryption.
/// Each packet must have a unique counter value to prevent
/// nonce reuse attacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct PacketCounter(u64);

impl PacketCounter {
    /// Creates a new counter with value 0.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Creates a counter from a raw u64 value.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw counter value.
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.0
    }

    /// Returns the counter as little-endian bytes.
    #[must_use]
    pub const fn to_le_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Increments the counter, returning the new value.
    ///
    /// # Panics
    /// Panics on overflow (after 2^64 - 1 packets, which is practically impossible).
    #[must_use]
    pub fn increment(&mut self) -> Self {
        self.0 = self.0.checked_add(1).expect("Counter overflow");
        *self
    }

    /// Checks if this counter is greater than another.
    /// Used for replay protection.
    #[must_use]
    pub const fn is_newer_than(&self, other: &Self) -> bool {
        self.0 > other.0
    }
}

impl From<u64> for PacketCounter {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<PacketCounter> for u64 {
    fn from(counter: PacketCounter) -> Self {
        counter.0
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_generation() {
        let id1 = SessionId::generate();
        let id2 = SessionId::generate();
        
        // Two random IDs should be different
        assert_ne!(id1, id2);
        
        // Should be correct size
        assert_eq!(id1.as_bytes().len(), SESSION_ID_SIZE);
    }

    #[test]
    fn test_session_id_roundtrip() {
        let original = SessionId::generate();
        
        // Byte roundtrip
        let bytes = original.as_bytes();
        let restored = SessionId::from_bytes(bytes).unwrap();
        assert_eq!(original, restored);
        
        // String roundtrip
        let s = original.to_string();
        let parsed: SessionId = s.parse().unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_session_id_invalid_length() {
        let short = [0u8; 8];
        assert!(SessionId::from_bytes(&short).is_none());
        
        let long = [0u8; 32];
        assert!(SessionId::from_bytes(&long).is_none());
    }

    #[test]
    fn test_virtual_ip_in_range() {
        let vip = VirtualIp::new(Ipv4Addr::new(100, 64, 0, 50));
        
        // Should be in /24 range
        assert!(vip.is_in_range(Ipv4Addr::new(100, 64, 0, 0), 24));
        
        // Should not be in different /24
        assert!(!vip.is_in_range(Ipv4Addr::new(192, 168, 0, 0), 24));
        
        // Should be in /16 range
        assert!(vip.is_in_range(Ipv4Addr::new(100, 64, 0, 0), 16));
    }

    #[test]
    fn test_packet_counter() {
        let mut counter = PacketCounter::new();
        assert_eq!(counter.value(), 0);
        
        counter.increment();
        assert_eq!(counter.value(), 1);
        
        let old = PacketCounter::from_raw(5);
        let new = PacketCounter::from_raw(10);
        assert!(new.is_newer_than(&old));
        assert!(!old.is_newer_than(&new));
    }

    #[test]
    fn test_session_id_json_serialization() {
        let original = SessionId::generate();
        let json = serde_json::to_string(&original).unwrap();
        let restored: SessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_session_id_clone() {
        let original = SessionId::generate();
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_session_id_error_display() {
        let err = SessionIdError::InvalidLength { expected: 16, actual: 8 };
        assert!(err.to_string().contains("16"));
        assert!(err.to_string().contains("8"));
    }
}
