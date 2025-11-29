// ============================================
// File: crates/aeronyx-core/src/crypto/mod.rs
// ============================================
//! # Cryptography Module
//!
//! ## Creation Reason
//! Centralizes all cryptographic operations for the AeroNyx privacy network,
//! using audited RustCrypto implementations.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`keys`]: Key types and generation (Ed25519, X25519)
//! - [`handshake`]: Handshake cryptography (signing, verification)
//! - [`transport`]: Transport encryption (ChaCha20-Poly1305)
//! - [`kdf`]: Key derivation functions (HKDF-SHA256)
//!
//! ### Key Types
//! - `IdentityKeyPair`: Long-term Ed25519 keys for signing
//! - `EphemeralKeyPair`: Per-session X25519 keys for key exchange
//! - `SessionKey`: Derived symmetric key for transport encryption
//!
//! ## Cryptographic Design
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Handshake Phase                          │
//! │  Client                                        Server       │
//! │    │                                              │         │
//! │    │  Ed25519 Identity Key ────────────────────► │         │
//! │    │  X25519 Ephemeral Key ────────────────────► │         │
//! │    │  Signature ───────────────────────────────► │         │
//! │    │                                              │         │
//! │    │ ◄──────────────────── X25519 Ephemeral Key  │         │
//! │    │ ◄──────────────────────────────── Signature │         │
//! │    │                                              │         │
//! │    │        X25519 Key Exchange                   │         │
//! │    │              │                               │         │
//! │    │              ▼                               │         │
//! │    │      HKDF-SHA256 ─────► Session Key         │         │
//! └─────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Transport Phase                          │
//! │                                                             │
//! │   Session Key + Counter ──► ChaCha20-Poly1305 ──► Cipher    │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Security Properties
//! - **Forward Secrecy**: New ephemeral keys per session
//! - **Authentication**: Ed25519 signatures on handshake
//! - **Confidentiality**: ChaCha20 stream cipher
//! - **Integrity**: Poly1305 authentication tag
//! - **Replay Protection**: Monotonic counters as nonce
//!
//! ## ⚠️ Important Note for Next Developer
//! - ALL implementations use RustCrypto (audited)
//! - NEVER roll your own crypto
//! - ALL sensitive keys implement Zeroize
//! - Test vectors should match specifications
//!
//! ## Last Modified
//! v0.1.0 - Initial crypto implementation

pub mod handshake;
pub mod kdf;
pub mod keys;
pub mod transport;

// Re-export primary types at module level
pub use handshake::HandshakeCrypto;
pub use keys::{EphemeralKeyPair, IdentityKeyPair, SessionKey};
pub use transport::TransportCrypto;

// ============================================
// Constants
// ============================================

/// Size of Ed25519 public key in bytes.
pub const ED25519_PUBLIC_KEY_SIZE: usize = 32;

/// Size of Ed25519 signature in bytes.
pub const ED25519_SIGNATURE_SIZE: usize = 64;

/// Size of X25519 public key in bytes.
pub const X25519_PUBLIC_KEY_SIZE: usize = 32;

/// Size of ChaCha20-Poly1305 key in bytes.
pub const CHACHA20_KEY_SIZE: usize = 32;

/// Size of ChaCha20-Poly1305 nonce in bytes.
pub const CHACHA20_NONCE_SIZE: usize = 12;

/// Size of Poly1305 authentication tag in bytes.
pub const POLY1305_TAG_SIZE: usize = 16;

/// HKDF salt for session key derivation.
pub const HKDF_SALT: &[u8] = b"aeronyx-v1";

/// HKDF info prefix for key derivation.
pub const HKDF_INFO_PREFIX: &[u8] = b"aeronyx-session-key";
