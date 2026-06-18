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
//! - v0.1.0-DiscoveryPhase1: Added `discovery` submodule for signed node
//!   descriptors used by decentralized peer discovery and encrypted relay.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`messages`]: Protocol message structures
//! - [`codec`]: Binary serialization/deserialization
//! - [`version`]: Protocol versioning
//! - [`memchain`]: 🌟 MemChain application-layer messages
//! - [`chat`]: 🌟 Chat Relay E2E envelope and media pointer types
//! - [`discovery`]: Signed node descriptors and public capability hints
//!
//! ### Message Types
//! - `ClientHello`: Initial handshake from client
//! - `ServerHello`: Server response with session parameters
//! - `DataPacket`: Encrypted tunnel data
//! - `MemChainMessage`: 🌟 AI memory sync + chat relay messages (inside DataPacket)
//! - `ChatEnvelope`: 🌟 E2E-encrypted chat message (carried by ChatRelay variant)
//! - `SignedNodeDescriptor`: Signed node metadata for discovery snapshots/gossip
//!
//! ## ⚠️ Important Note for Next Developer
//! - ANY protocol change requires version bump
//! - Maintain backward compatibility where possible
//! - The `memchain` module does NOT touch outer protocol wire format
//! - The `chat` module does NOT touch outer protocol wire format
//! - `chat` module is intentionally separate from `memchain` to keep
//!   the crypto/signing logic isolated and independently testable
//! - The `discovery` module is control-plane metadata only; do not include
//!   client traffic, payloads, DNS contents, or private keys in descriptors
//!
//! ## Last Modified
//! v0.1.0 - Initial protocol definitions
//! v0.2.0 - Added memchain submodule for MemChain P2P memory sync
//! v0.5.0 - 🌟 BlockAnnounce variant added to MemChainMessage
//! v1.1.0-ChatRelay - 🌟 Added chat submodule for ChatEnvelope, ChatContentType,
//!                        MediaPointer and related signing helpers
//! v0.1.0-DiscoveryPhase1 - Added discovery submodule for signed descriptors
//! v0.2.0-DiscoveryPhase2 - Re-exported bootstrap snapshot type
//! v0.3.0-DiscoveryPhase4 - Re-exported discovery gossip message helpers

pub mod auth;
pub mod chat;
pub mod codec;
pub mod discovery;
pub mod memchain;
pub mod messages;
pub mod version;

// Re-export primary types
pub use auth::{
    verify_signed_message, AuthError, DOMAIN_CHAT_ACK, DOMAIN_CHAT_PULL, DOMAIN_DEVICE_REGISTER,
    DOMAIN_WALLET_PRESENCE,
};
pub use chat::{decode_envelope, encode_envelope, ChatContentType, ChatEnvelope, MediaPointer};
pub use codec::{Codec, ProtocolCodec};
pub use discovery::{
    decode_discovery_message, encode_discovery_message, NodeBootstrapSnapshot, NodeCapability,
    NodeCapacity, NodeDescriptor, NodeDiscoveryMessage, NodePolicy, SignedNodeDescriptor,
    NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION, NODE_DESCRIPTOR_SCHEMA_VERSION,
};
pub use memchain::{decode_memchain, encode_memchain, MemChainMessage, MEMCHAIN_MAGIC};
pub use messages::{ClientHello, DataPacket, MessageType, ServerHello};
pub use version::{ProtocolVersion, CURRENT_PROTOCOL_VERSION};
