// ============================================
// File: crates/aeronyx-core/src/protocol/mod.rs
// ============================================
//! # Protocol Module
//!
//! ## Creation Reason
//! Defines the wire protocol for AeroNyx privacy network communication,
//! including message types, formats, and serialization.
//!
//! ## Modification Reason
//! Added `memchain` submodule for MemChain P2P memory synchronisation
//! messages. These messages travel **inside** existing encrypted DataPackets,
//! multiplexed by a single magic byte (0xAE) after decryption.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`messages`]: Protocol message structures
//! - [`codec`]: Binary serialization/deserialization
//! - [`version`]: Protocol versioning
//! - [`memchain`]: 🌟 MemChain application-layer messages (NEW)
//!
//! ### Message Types
//! - `ClientHello`: Initial handshake from client
//! - `ServerHello`: Server response with session parameters
//! - `DataPacket`: Encrypted tunnel data
//! - `MemChainMessage`: 🌟 AI memory sync messages (inside DataPacket)
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
//! │         ┌─────────────────────────────────────────┐        │
//! │         │ After decrypt, peek plaintext[0]:       │        │
//! │         │   0x45/0x60 → VPN IP packet → TUN       │        │
//! │         │   0xAE      → MemChain msg  → MemPool   │        │
//! │         └─────────────────────────────────────────┘        │
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
//! - The `memchain` module does NOT touch outer protocol wire format;
//!   it only defines the inner plaintext payload schema.
//!
//! ## Last Modified
//! v0.1.0 - Initial protocol definitions
//! v0.2.0 - Added memchain submodule for MemChain P2P memory sync

pub mod codec;
pub mod memchain;
pub mod messages;
pub mod version;

// Re-export primary types
pub use codec::{Codec, ProtocolCodec};
pub use memchain::{MemChainMessage, MEMCHAIN_MAGIC, decode_memchain, encode_memchain};
pub use messages::{ClientHello, DataPacket, MessageType, ServerHello};
pub use version::{ProtocolVersion, CURRENT_PROTOCOL_VERSION};
