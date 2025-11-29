// ============================================
// File: crates/aeronyx-core/src/protocol/mod.rs
// ============================================
//! # Protocol Module
//!
//! ## Creation Reason
//! Defines the wire protocol for AeroNyx privacy network communication,
//! including message types, formats, and serialization.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`messages`]: Protocol message structures
//! - [`codec`]: Binary serialization/deserialization
//! - [`version`]: Protocol versioning
//!
//! ### Message Types
//! - `ClientHello`: Initial handshake from client
//! - `ServerHello`: Server response with session parameters
//! - `DataPacket`: Encrypted tunnel data
//!
//! ## Protocol Overview
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Handshake Phase                          │
//! │                                                             │
//! │  Client ──────── ClientHello (138 bytes) ──────────► Server │
//! │  Client ◄─────── ServerHello (150 bytes) ────────── Server │
//! │                                                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Transport Phase                          │
//! │                                                             │
//! │  Client ══════ DataPacket (encrypted) ══════════════ Server │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Wire Format Principles
//! - Little-endian byte order for multi-byte integers
//! - Fixed-size fields where possible for fast parsing
//! - No padding or alignment requirements
//! - Version field in all messages for forward compatibility
//!
//! ## ⚠️ Important Note for Next Developer
//! - ANY protocol change requires version bump
//! - Maintain backward compatibility where possible
//! - Test vectors should be maintained for all message types
//! - Consider endianness carefully
//!
//! ## Last Modified
//! v0.1.0 - Initial protocol definitions

pub mod codec;
pub mod messages;
pub mod version;

// Re-export primary types
pub use codec::{Codec, ProtocolCodec};
pub use messages::{ClientHello, DataPacket, MessageType, ServerHello};
pub use version::{ProtocolVersion, CURRENT_PROTOCOL_VERSION};
