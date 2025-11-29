// ============================================
// File: crates/aeronyx-common/src/time.rs
// ============================================
//! # Time Utilities
//!
//! ## Creation Reason
//! Provides time-related utilities including atomic timestamps for
//! concurrent access in session management.
//!
//! ## Main Functionality
//! - `AtomicInstant`: Thread-safe wrapper around `Instant`
//! - `Timestamp`: Unix timestamp with validation
//! - Utility functions for time operations
//!
//! ## Main Logical Flow
//! 1. Sessions store `AtomicInstant` for last activity tracking
//! 2. Background tasks read these atomically for cleanup decisions
//! 3. Packet handlers update atomically without locks
//!
//! ## ⚠️ Important Note for Next Developer
//! - `AtomicInstant` uses `AtomicU64` internally (nanoseconds since start)
//! - Be aware of potential overflow after ~584 years of uptime
//! - Timestamps should be validated against reasonable bounds
//!
//! ## Last Modified
//! v0.1.0 - Initial time utilities

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ============================================
// Constants
// ============================================

/// Maximum acceptable clock skew for timestamp validation (30 seconds).
pub const MAX_CLOCK_SKEW_SECS: i64 = 30;

/// Minimum valid Unix timestamp (year 2020).
pub const MIN_VALID_TIMESTAMP: i64 = 1_577_836_800;

/// Maximum valid Unix timestamp (year 2100).
pub const MAX_VALID_TIMESTAMP: i64 = 4_102_444_800;

// ============================================
// AtomicInstant
// ============================================

/// Thread-safe wrapper around [`Instant`] for concurrent access.
///
/// # Purpose
/// Allows multiple threads to read/write timestamps without locks,
/// essential for high-performance session management.
///
/// # Implementation
/// Stores nanoseconds elapsed since a reference instant (program start).
/// Uses `AtomicU64` with relaxed ordering for performance.
///
/// # Example
/// ```
/// use aeronyx_common::time::AtomicInstant;
/// use std::time::Instant;
///
/// let atomic = AtomicInstant::now();
/// let instant = atomic.load();
///
/// // Update from another thread
/// atomic.store(Instant::now());
/// ```
#[derive(Debug)]
pub struct AtomicInstant {
    /// Nanoseconds since the reference instant
    nanos: AtomicU64,
}

impl AtomicInstant {
    /// Reference instant (lazily initialized at program start).
    fn reference() -> Instant {
        use std::sync::OnceLock;
        static REFERENCE: OnceLock<Instant> = OnceLock::new();
        *REFERENCE.get_or_init(Instant::now)
    }

    /// Creates a new `AtomicInstant` set to the current time.
    #[must_use]
    pub fn now() -> Self {
        Self::from_instant(Instant::now())
    }

    /// Creates a new `AtomicInstant` from an `Instant`.
    #[must_use]
    pub fn from_instant(instant: Instant) -> Self {
        let reference = Self::reference();
        let nanos = instant
            .checked_duration_since(reference)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self {
            nanos: AtomicU64::new(nanos),
        }
    }

    /// Loads the stored instant.
    ///
    /// Uses `Relaxed` ordering for best performance.
    #[must_use]
    pub fn load(&self) -> Instant {
        let nanos = self.nanos.load(Ordering::Relaxed);
        Self::reference() + Duration::from_nanos(nanos)
    }

    /// Stores a new instant.
    ///
    /// Uses `Relaxed` ordering for best performance.
    pub fn store(&self, instant: Instant) {
        let reference = Self::reference();
        let nanos = instant
            .checked_duration_since(reference)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.nanos.store(nanos, Ordering::Relaxed);
    }

    /// Updates to the current time and returns the previous value.
    pub fn touch(&self) -> Instant {
        let old = self.load();
        self.store(Instant::now());
        old
    }

    /// Returns the elapsed time since the stored instant.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.load().elapsed()
    }

    /// Checks if more than `duration` has elapsed since the stored instant.
    #[must_use]
    pub fn has_elapsed(&self, duration: Duration) -> bool {
        self.elapsed() > duration
    }
}

impl Default for AtomicInstant {
    fn default() -> Self {
        Self::now()
    }
}

impl Clone for AtomicInstant {
    fn clone(&self) -> Self {
        Self {
            nanos: AtomicU64::new(self.nanos.load(Ordering::Relaxed)),
        }
    }
}

// ============================================
// Timestamp
// ============================================

/// Unix timestamp in seconds.
///
/// # Purpose
/// Used in protocol messages for time-based validation and
/// replay attack prevention.
///
/// # Validation
/// Timestamps are validated to be within reasonable bounds
/// (2020-2100) and not too far from current time.
///
/// # Example
/// ```
/// use aeronyx_common::time::Timestamp;
///
/// let now = Timestamp::now();
/// assert!(now.is_valid());
/// assert!(now.is_recent(30)); // Within 30 seconds
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Creates a new timestamp from Unix seconds.
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Self(secs)
    }

    /// Creates a timestamp for the current time.
    #[must_use]
    pub fn now() -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before Unix epoch")
            .as_secs() as i64;
        Self(secs)
    }

    /// Returns the Unix timestamp in seconds.
    #[must_use]
    pub const fn as_secs(&self) -> i64 {
        self.0
    }

    /// Returns the timestamp as little-endian bytes.
    #[must_use]
    pub const fn to_le_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Creates a timestamp from little-endian bytes.
    #[must_use]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(i64::from_le_bytes(bytes))
    }

    /// Checks if the timestamp is within valid bounds.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.0 >= MIN_VALID_TIMESTAMP && self.0 <= MAX_VALID_TIMESTAMP
    }

    /// Checks if the timestamp is recent (within `max_age_secs` of now).
    ///
    /// # Arguments
    /// * `max_age_secs` - Maximum allowed difference from current time
    ///
    /// # Returns
    /// `true` if `|timestamp - now| <= max_age_secs`
    #[must_use]
    pub fn is_recent(&self, max_age_secs: u64) -> bool {
        let now = Self::now().0;
        let diff = (self.0 - now).unsigned_abs();
        diff <= max_age_secs
    }

    /// Validates the timestamp for protocol use.
    ///
    /// Checks both validity and recency using default bounds.
    #[must_use]
    pub fn validate(&self) -> Result<(), TimestampError> {
        if !self.is_valid() {
            return Err(TimestampError::OutOfBounds {
                value: self.0,
                min: MIN_VALID_TIMESTAMP,
                max: MAX_VALID_TIMESTAMP,
            });
        }

        if !self.is_recent(MAX_CLOCK_SKEW_SECS as u64) {
            return Err(TimestampError::ClockSkew {
                timestamp: self.0,
                now: Self::now().0,
                max_skew: MAX_CLOCK_SKEW_SECS,
            });
        }

        Ok(())
    }

    /// Returns the difference from the current time in seconds.
    ///
    /// Positive values mean the timestamp is in the future.
    #[must_use]
    pub fn offset_from_now(&self) -> i64 {
        self.0 - Self::now().0
    }
}

impl From<i64> for Timestamp {
    fn from(secs: i64) -> Self {
        Self(secs)
    }
}

impl From<Timestamp> for i64 {
    fn from(ts: Timestamp) -> Self {
        ts.0
    }
}

// ============================================
// TimestampError
// ============================================

/// Errors that can occur during timestamp validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TimestampError {
    /// Timestamp is outside valid bounds.
    #[error("Timestamp {value} out of bounds [{min}, {max}]")]
    OutOfBounds {
        /// The invalid timestamp value
        value: i64,
        /// Minimum valid value
        min: i64,
        /// Maximum valid value
        max: i64,
    },

    /// Timestamp differs too much from current time.
    #[error("Clock skew detected: timestamp={timestamp}, now={now}, max_skew={max_skew}s")]
    ClockSkew {
        /// The timestamp value
        timestamp: i64,
        /// Current time
        now: i64,
        /// Maximum allowed skew
        max_skew: i64,
    },
}

// ============================================
// Utility Functions
// ============================================

/// Returns the current Unix timestamp in seconds.
#[must_use]
pub fn unix_timestamp() -> i64 {
    Timestamp::now().as_secs()
}

/// Returns the current Unix timestamp in milliseconds.
#[must_use]
pub fn unix_timestamp_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time before Unix epoch")
        .as_millis() as i64
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_atomic_instant_basic() {
        let atomic = AtomicInstant::now();
        let loaded = atomic.load();
        
        // Should be very close to now
        assert!(loaded.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn test_atomic_instant_store() {
        let atomic = AtomicInstant::now();
        thread::sleep(Duration::from_millis(10));
        
        let before = atomic.load();
        atomic.store(Instant::now());
        let after = atomic.load();
        
        assert!(after > before);
    }

    #[test]
    fn test_atomic_instant_elapsed() {
        let atomic = AtomicInstant::now();
        thread::sleep(Duration::from_millis(10));
        
        assert!(atomic.elapsed() >= Duration::from_millis(10));
        assert!(atomic.has_elapsed(Duration::from_millis(5)));
    }

    #[test]
    fn test_timestamp_now() {
        let ts = Timestamp::now();
        assert!(ts.is_valid());
        assert!(ts.is_recent(1));
    }

    #[test]
    fn test_timestamp_validation() {
        // Valid recent timestamp
        let now = Timestamp::now();
        assert!(now.validate().is_ok());

        // Too old
        let old = Timestamp::from_secs(Timestamp::now().as_secs() - 60);
        assert!(matches!(
            old.validate(),
            Err(TimestampError::ClockSkew { .. })
        ));

        // Out of bounds
        let invalid = Timestamp::from_secs(0);
        assert!(matches!(
            invalid.validate(),
            Err(TimestampError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn test_timestamp_bytes_roundtrip() {
        let original = Timestamp::now();
        let bytes = original.to_le_bytes();
        let restored = Timestamp::from_le_bytes(bytes);
        assert_eq!(original, restored);
    }
}
