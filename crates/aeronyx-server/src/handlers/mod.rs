// ============================================
// File: crates/aeronyx-server/src/handlers/mod.rs
// ============================================
//! # Packet Handlers
//!
//! ## Creation Reason
//! Provides packet processing logic for both UDP (client-facing)
//! and TUN (internet-facing) interfaces.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`packet`]: Data packet encryption/decryption and forwarding
//!
//! ## Handler Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Handlers                                │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  ┌───────────────────────────────────────────────────────┐ │
//! │  │                 PacketHandler                         │ │
//! │  │                                                       │ │
//! │  │  ┌─────────────┐          ┌─────────────┐           │ │
//! │  │  │ handle_udp  │          │ handle_tun  │           │ │
//! │  │  │             │          │             │           │ │
//! │  │  │ - Decrypt   │          │ - Lookup    │           │ │
//! │  │  │ - Validate  │          │   route     │           │ │
//! │  │  │ - Forward   │          │ - Encrypt   │           │ │
//! │  │  │   to TUN    │          │ - Send via  │           │ │
//! │  │  │             │          │   UDP       │           │ │
//! │  │  └─────────────┘          └─────────────┘           │ │
//! │  └───────────────────────────────────────────────────────┘ │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Data Flow
//! ```text
//! Client → UDP:
//!   1. Receive encrypted packet
//!   2. Extract session_id, lookup session
//!   3. Decrypt packet
//!   4. Validate source IP
//!   5. Write to TUN device
//!
//! TUN → Client:
//!   1. Read IP packet from TUN
//!   2. Extract destination IP
//!   3. Lookup route → session
//!   4. Encrypt packet
//!   5. Send via UDP to client
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Handlers must be fast (critical path)
//! - Use zero-copy where possible
//! - Update session activity on every packet
//! - Log suspicious activity (replay, auth failure)
//!
//! ## Last Modified
//! v0.1.0 - Initial handlers structure

pub mod packet;

pub use packet::PacketHandler;
