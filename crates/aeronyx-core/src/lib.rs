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
//! Added `ledger` module containing the MemChain data structures (Fact,
//! future Block/Merkle). These structures are used by both `aeronyx-core`
//! (protocol serialisation) and `aeronyx-server` (storage and routing).
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
//! ### Ledger Module ([`ledger`]) — 🌟 NEW
//! - `Fact`: Atomic AI memory record (subject-predicate-object triple)
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
//!
//! ## Last Modified
//! v0.1.0 - Initial implementation
//! v0.2.0 - Added ledger module for MemChain AI memory structures

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod crypto;
pub mod error;
pub mod ledger;
pub mod protocol;

// Re-export commonly used items
pub use crypto::{
    EphemeralKeyPair, HandshakeCrypto, IdentityKeyPair, SessionKey, TransportCrypto,
};
pub use error::{CoreError, Result};
pub use ledger::Fact;
pub use protocol::{
    ClientHello, MessageType, ProtocolVersion, ServerHello, CURRENT_PROTOCOL_VERSION,
};
