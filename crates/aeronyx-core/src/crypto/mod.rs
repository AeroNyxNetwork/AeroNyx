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
//! - `IdentityPublicKey`: Ed25519 public key for signature verification
//! - `EphemeralKeyPair`: Per-session X25519 keys for key exchange
//! - `SessionKey`: Derived symmetric key for transport encryption
//! - `E2eSession`: Per-WebSocket E2E encryption session (v2.2.0)
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
//!
//! ┌─────────────────────────────────────────────────────────────┐
//! │                E2E Chat Phase (v2.2.0)                      │
//! │  Frontend                                      Rust Node    │
//! │    │                                              │         │
//! │    │  Temp X25519 PK ─────────────────────────► │         │
//! │    │  (e2e_init)                                  │         │
//! │    │                      Ed25519 → X25519        │         │
//! │    │ ◄──────────── X25519 PK + e2e_ready         │         │
//! │    │                                              │         │
//! │    │  Both compute: shared = X25519(my_sk, peer_pk)         │
//! │    │  Encrypt/Decrypt: XSalsa20-Poly1305 (NaCl Box)         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Security Properties
//! - **Forward Secrecy**: New ephemeral keys per session
//! - **Authentication**: Ed25519 signatures on handshake
//! - **Confidentiality**: ChaCha20 stream cipher / XSalsa20 for E2E
//! - **Integrity**: Poly1305 authentication tag
//! - **Replay Protection**: Monotonic counters / unique nonces
//! - **Zero-Knowledge**: CMS cannot decrypt E2E messages
//!
//! ## ⚠️ Important Note for Next Developer
//! - ALL implementations use RustCrypto (audited)
//! - NEVER roll your own crypto
//! - ALL sensitive keys implement Zeroize
//! - Test vectors should match specifications
//! - E2E uses NaCl Box (XSalsa20-Poly1305) for browser compat (tweetnacl)
//!
//! ## Last Modified
//! v0.1.0 - Initial crypto implementation
//! v2.2.0 - 🌟 Added E2eSession for E2E encrypted chat
//! v2.3.0 - 🌟 Added IdentityPublicKey re-export (needed by MPI auth middleware
//!   for Ed25519 signature verification of remote storage requests)

pub mod handshake;
pub mod kdf;
pub mod keys;
pub mod transport;

// Re-export primary types at module level
pub use handshake::HandshakeCrypto;
pub use keys::{EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey, SessionKey};
pub use keys::E2eSession;
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

/// Size of XSalsa20-Poly1305 nonce in bytes (NaCl Box).
pub const XSALSA20_NONCE_SIZE: usize = 24;

/// HKDF salt for session key derivation.
pub const HKDF_SALT: &[u8] = b"aeronyx-v1";

/// HKDF info prefix for key derivation.
pub const HKDF_INFO_PREFIX: &[u8] = b"aeronyx-session-key";
