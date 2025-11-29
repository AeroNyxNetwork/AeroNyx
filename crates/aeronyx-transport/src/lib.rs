// ============================================
// File: crates/aeronyx-transport/src/lib.rs
// ============================================
//! # AeroNyx Transport - Network I/O Layer
//!
//! ## Creation Reason
//! Provides network transport abstractions for the AeroNyx privacy network,
//! including UDP sockets for client communication and TUN devices for
//! IP packet tunneling.
//!
//! ## Main Functionality
//!
//! ### Modules
//! - [`traits`]: Transport trait definitions for abstraction
//! - [`udp`]: UDP socket implementation
//! - [`tun`]: TUN device management (Linux)
//! - [`error`]: Transport-specific error types
//!
//! ## Architecture Position
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              aeronyx-server                         │
//! │                    │                                │
//! │         ┌──────────┴──────────┐                    │
//! │         ▼                     ▼                    │
//! │   aeronyx-core         aeronyx-transport           │
//! │                        You are here ◄──            │
//! │         │                     │                    │
//! │         └──────────┬──────────┘                    │
//! │                    ▼                               │
//! │             aeronyx-common                         │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Data Flow
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    Internet                              │
//! │                       ▲                                  │
//! │                       │                                  │
//! │            ┌──────────┴──────────┐                      │
//! │            │    TUN Device       │                      │
//! │            │   (IP packets)      │                      │
//! │            └──────────┬──────────┘                      │
//! │                       │                                  │
//! │            ┌──────────┴──────────┐                      │
//! │            │   Server Logic      │                      │
//! │            │  (encrypt/decrypt)  │                      │
//! │            └──────────┬──────────┘                      │
//! │                       │                                  │
//! │            ┌──────────┴──────────┐                      │
//! │            │    UDP Socket       │                      │
//! │            │ (encrypted packets) │                      │
//! │            └──────────┬──────────┘                      │
//! │                       │                                  │
//! │                       ▼                                  │
//! │                    Clients                               │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Platform Support
//! | Platform | UDP | TUN |
//! |----------|-----|-----|
//! | Linux | ✅ | ✅ |
//! | macOS | ✅ | ⚠️ (utun) |
//! | Windows | ✅ | ❌ |
//!
//! ## ⚠️ Important Note for Next Developer
//! - TUN operations require elevated privileges
//! - Always use traits for testability
//! - Platform-specific code must be isolated
//! - Mock implementations available with `mock` feature
//!
//! ## Last Modified
//! v0.1.0 - Initial transport layer implementation

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod traits;
pub mod tun;
pub mod udp;

// Re-export primary types
pub use error::{TransportError, Result};
pub use traits::{PacketSource, Transport, TunDevice};
pub use udp::UdpTransport;

#[cfg(target_os = "linux")]
pub use tun::linux::LinuxTun;
