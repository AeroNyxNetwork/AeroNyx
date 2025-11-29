// ============================================
// File: crates/aeronyx-transport/src/tun/mod.rs
// ============================================
//! # TUN Device Module
//!
//! ## Creation Reason
//! Provides TUN (network tunnel) device management for IP packet
//! handling in the privacy network.
//!
//! ## Main Functionality
//! - Platform-specific TUN implementations
//! - Mock implementation for testing
//! - Unified interface via `TunDevice` trait
//!
//! ## Platform Implementations
//! - `linux`: Uses `/dev/net/tun` with IFF_TUN
//! - `mock`: In-memory implementation for testing
//!
//! ## What is a TUN Device?
//! A TUN device is a virtual network interface that operates at
//! Layer 3 (IP). It allows userspace programs to read and write
//! IP packets directly, enabling VPN-like functionality.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │                     User Space                            │
//! │  ┌────────────────┐          ┌────────────────────────┐  │
//! │  │  Application   │          │    AeroNyx Server      │  │
//! │  │  (browser etc) │          │                        │  │
//! │  └───────┬────────┘          └───────────┬────────────┘  │
//! │          │                               │               │
//! │          │ IP packets                    │ read/write    │
//! │          ▼                               ▼               │
//! ├──────────────────────────────────────────────────────────┤
//! │                     Kernel Space                          │
//! │  ┌────────────────────────────────────────────────────┐  │
//! │  │                  TUN Device (tun0)                  │  │
//! │  │               Virtual Network Interface             │  │
//! │  └────────────────────────────────────────────────────┘  │
//! │          │                               │               │
//! │          │ routing                       │               │
//! │          ▼                               ▼               │
//! │  ┌────────────────┐          ┌────────────────────────┐  │
//! │  │ Physical NIC   │          │     IP Routing         │  │
//! │  │   (eth0)       │          │                        │  │
//! │  └───────┬────────┘          └────────────────────────┘  │
//! └──────────┼────────────────────────────────────────────────┘
//!            │
//!            ▼
//!        Internet
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - TUN operations require root or CAP_NET_ADMIN capability
//! - TUN packets are raw IP (no Ethernet header)
//! - Device names are limited to 15 characters on Linux
//! - Always clean up devices on shutdown
//!
//! ## Last Modified
//! v0.1.0 - Initial TUN module structure

// Platform-specific implementations
#[cfg(target_os = "linux")]
pub mod linux;

// Mock implementation for testing
#[cfg(any(test, feature = "mock"))]
pub mod mock;

// Re-export based on platform
#[cfg(target_os = "linux")]
pub use linux::LinuxTun;

#[cfg(any(test, feature = "mock"))]
pub use mock::MockTun;
