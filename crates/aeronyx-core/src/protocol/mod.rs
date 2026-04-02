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
//! - Added `memchain` submodule for MemChain P2P memory synchronisation
//!   messages. These messages travel **inside** existing encrypted DataPackets,
//!   multiplexed by a single magic byte (0xAE) after decryption.
//! - 🌟 v0.5.0: MemChainMessage now includes `BlockAnnounce(BlockHeader)`.
//! - 🌟 v1.1.0-ChatRelay: Added `chat` submodule for zero-knowledge P2P
//!   messaging data structures. `ChatEnvelope`, `ChatContentType`, and
//!   `MediaPointer` live here; they are consumed by `MemChainMessage::ChatRelay`
//!   in `memchain.rs` and by `ChatRelayService` in `aeronyx-server`.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`messages`]: Protocol message structures
//! - [`codec`]: Binary serialization/deserialization
//! - [`version`]: Protocol versioning
//! - [`memchain`]: 🌟 MemChain application-layer messages
//! - [`chat`]: 🌟 Chat Relay E2E envelope and media pointer types
//!
//! ### Message Types
//! - `ClientHello`: Initial handshake from client
//! - `ServerHello`: Server response with session parameters
//! - `DataPacket`: Encrypted tunnel data
//! - `MemChainMessage`: 🌟 AI memory sync + chat relay messages (inside DataPacket)
//! - `ChatEnvelope`: 🌟 E2E-encrypted chat message (carried by ChatRelay variant)
//!
//! ## ⚠️ Important Note for Next Developer
//! - ANY protocol change requires version bump
//! - Maintain backward compatibility where possible
//! - The `memchain` module does NOT touch outer protocol wire format
//! - The `chat` module does NOT touch outer protocol wire format
//! - `chat` module is intentionally separate from `memchain` to keep
//!   the crypto/signing logic isolated and independently testable
//!
//! ## Last Modified
//! v0.1.0 - Initial protocol definitions
//! v0.2.0 - Added memchain submodule for MemChain P2P memory sync
//! v0.5.0 - 🌟 BlockAnnounce variant added to MemChainMessage
//! v1.1.0-ChatRelay - 🌟 Added chat submodule for ChatEnvelope, ChatContentType,
//!                        MediaPointer and related signing helpers

pub mod chat;
pub mod codec;
pub mod memchain;
pub mod messages;
pub mod version;

// Re-export primary types
pub use chat::{ChatContentType, ChatEnvelope, MediaPointer, decode_envelope, encode_envelope};
pub use codec::{Codec, ProtocolCodec};
pub use memchain::{MemChainMessage, MEMCHAIN_MAGIC, decode_memchain, encode_memchain};
pub use messages::{ClientHello, DataPacket, MessageType, ServerHello};
pub use version::{ProtocolVersion, CURRENT_PROTOCOL_VERSION};
