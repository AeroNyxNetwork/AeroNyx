// ============================================
// File: crates/aeronyx-core/src/lib.rs
// ============================================
//! # AeroNyx Core - Protocol & Cryptography Library
//!
//! ## Creation Reason
//! Provides the foundational protocol definitions and cryptographic operations
//! for the AeroNyx privacy network. This crate is the security backbone of
//! the entire system.
//!
//! ## Modification Reason
//! - Added `ledger` module containing the MemChain data structures (Fact,
//!   future Block/Merkle). These structures are used by both `aeronyx-core`
//!   (protocol serialisation) and `aeronyx-server` (storage and routing).
//! - 🌟 v1.0.0: Added `MemoryRecord`, `MemoryLayer`, and `RecordStatus`
//!   re-exports from the new `ledger::record` submodule. These form the
//!   MRS-1 standard for the intelligent AI memory engine.
//!
//! ## Main Functionality
//!
//! ### Protocol Module ([`protocol`])
//! - Message type definitions (`ClientHello`, `ServerHello`, data packets)
//! - Binary codec for wire format serialization
//! - Protocol version management
//! - 🌟 MemChain messages (`MemChainMessage`, encode/decode helpers)
//!
//! ### Crypto Module ([`crypto`])
//! - Key types (`IdentityKeyPair`, `EphemeralKeyPair`, `SessionKey`)
//! - Handshake cryptography (signatures, key exchange)
//! - Transport encryption (ChaCha20-Poly1305)
//! - Key derivation (HKDF-SHA256)
//!
//! ### Ledger Module ([`ledger`])
//! - `Fact`: Legacy atomic AI memory record (subject-predicate-object triple)
//! - 🌟 `MemoryRecord`: MRS-1 standard AI memory unit (layered, encrypted)
//! - 🌟 `MemoryLayer`: Identity / Knowledge / Episode classification
//! - 🌟 `RecordStatus`: Active / Superseded / Revoked / Archived lifecycle
//! - `Block` / `RecordBlock`: Immutable containers for chain packing
//! - `BlockHeader`: Lightweight block summary for P2P broadcast
//! - Content-addressed hashing (SHA-256) and Ed25519 signature support
//!
//! ## Architecture Position
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              aeronyx-server                         │
//! │                    │                                │
//! │         ┌──────────┴──────────┐                    │
//! │         ▼                     ▼                    │
//! │   aeronyx-core  ◄──     aeronyx-transport          │
//! │   You are here        │                            │
//! │         │             │                            │
//! │         └──────────┬──────────┘                    │
//! │                    ▼                               │
//! │             aeronyx-common                         │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Security Guarantees
//! - **Confidentiality**: ChaCha20-Poly1305 authenticated encryption
//! - **Integrity**: Poly1305 MAC on all encrypted data
//! - **Authenticity**: Ed25519 signatures on handshake messages
//! - **Forward Secrecy**: X25519 ephemeral key exchange per session
//! - **Replay Protection**: Monotonic counters in packet headers
//!
//! ## ⚠️ Important Note for Next Developer
//! - ALL cryptographic code uses audited RustCrypto implementations
//! - NEVER implement custom crypto primitives
//! - ALL keys MUST implement Zeroize for secure cleanup
//! - Protocol changes MUST maintain backward compatibility
//! - Test vectors should match reference implementations
//! - `Fact` re-export is preserved for backward compatibility;
//!   new code should prefer `MemoryRecord`.
//!
//! ## Last Modified
//! v0.1.0 - Initial implementation
//! v0.2.0 - Added ledger module for MemChain AI memory structures
//! v1.0.0 - 🌟 Added MemoryRecord, MemoryLayer, RecordStatus (MRS-1)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod crypto;
pub mod error;
pub mod ledger;
pub mod protocol;

// Re-export commonly used items — crypto & protocol
pub use crypto::{
    EphemeralKeyPair, HandshakeCrypto, IdentityKeyPair, SessionKey, TransportCrypto,
};
pub use error::{CoreError, Result};
pub use protocol::{
    ClientHello, MessageType, ProtocolVersion, ServerHello, CURRENT_PROTOCOL_VERSION,
};

// Re-export commonly used items — ledger (legacy, deprecated)
#[allow(deprecated)]
pub use ledger::Fact;

// Re-export commonly used items — ledger (MRS-1)
pub use ledger::{MemoryLayer, MemoryRecord, RecordStatus};
