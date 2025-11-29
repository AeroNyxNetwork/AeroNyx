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
//! ## Main Functionality
//!
//! ### Protocol Module ([`protocol`])
//! - Message type definitions (`ClientHello`, `ServerHello`, data packets)
//! - Binary codec for wire format serialization
//! - Protocol version management
//!
//! ### Crypto Module ([`crypto`])
//! - Key types (`IdentityKeyPair`, `EphemeralKeyPair`, `SessionKey`)
//! - Handshake cryptography (signatures, key exchange)
//! - Transport encryption (ChaCha20-Poly1305)
//! - Key derivation (HKDF-SHA256)
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

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod crypto;
pub mod error;
pub mod protocol;

// Re-export commonly used items
pub use crypto::{
    EphemeralKeyPair, IdentityKeyPair, SessionKey,
    HandshakeCrypto, TransportCrypto,
};
pub use error::{CoreError, Result};
pub use protocol::{
    ClientHello, ServerHello, MessageType,
    ProtocolVersion, CURRENT_PROTOCOL_VERSION,
};
