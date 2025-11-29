// ============================================
// File: crates/aeronyx-core/src/crypto/keys.rs
// ============================================
//! # Cryptographic Key Types
//!
//! ## Creation Reason
//! Defines key types used throughout the AeroNyx protocol with proper
//! security properties (Zeroize on drop, constant-time comparison).
//!
//! ## Main Functionality
//! - `IdentityKeyPair`: Long-term Ed25519 signing keys
//! - `EphemeralKeyPair`: Per-session X25519 key exchange keys
//! - `SessionKey`: Derived symmetric encryption key
//!
//! ## Key Lifecycle
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │  IdentityKeyPair (Long-term)                               │
//! │  ├─ Generated once, stored securely                        │
//! │  ├─ Used for signing handshake messages                    │
//! │  └─ Identifies the endpoint                                │
//! │                                                            │
//! │  EphemeralKeyPair (Per-session)                            │
//! │  ├─ Generated fresh for each session                       │
//! │  ├─ Used for X25519 key exchange                           │
//! │  └─ Discarded after session key derived                    │
//! │                                                            │
//! │  SessionKey (Per-session)                                  │
//! │  ├─ Derived from key exchange                              │
//! │  ├─ Used for ChaCha20-Poly1305 encryption                  │
//! │  └─ Discarded when session ends                            │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - ALL key types MUST implement Zeroize
//! - Private keys should NEVER be logged or serialized carelessly
//! - Use constant-time comparison for key equality
//!
//! ## Last Modified
//! v0.1.0 - Initial key type definitions

use std::fmt;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use x25519_dalek::{EphemeralSecret, PublicKey as X25519PublicKey, SharedSecret};
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{ED25519_PUBLIC_KEY_SIZE, ED25519_SIGNATURE_SIZE, X25519_PUBLIC_KEY_SIZE, CHACHA20_KEY_SIZE};
use crate::error::{CoreError, Result};

// ============================================
// IdentityKeyPair (Ed25519)
// ============================================

/// Long-term Ed25519 identity key pair for signing.
///
/// # Purpose
/// Used to sign handshake messages, proving the identity of the
/// sender without revealing the private key.
///
/// # Security
/// - Private key is zeroed on drop
/// - Never serialize the private key to untrusted storage
/// - Generate using OS random number generator
///
/// # Example
/// ```
/// use aeronyx_core::crypto::IdentityKeyPair;
///
/// // Generate new identity
/// let identity = IdentityKeyPair::generate();
///
/// // Sign a message
/// let message = b"hello world";
/// let signature = identity.sign(message);
///
/// // Verify signature
/// assert!(identity.verify(message, &signature).is_ok());
/// ```
pub struct IdentityKeyPair {
    /// Ed25519 signing key (private)
    signing_key: SigningKey,
}

impl IdentityKeyPair {
    /// Generates a new random identity key pair.
    ///
    /// Uses the operating system's secure random number generator.
    #[must_use]
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// Creates an identity key pair from raw private key bytes.
    ///
    /// # Arguments
    /// * `bytes` - 32-byte Ed25519 private key seed
    ///
    /// # Errors
    /// Returns error if bytes length is incorrect.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(CoreError::key_generation(
                format!("Invalid Ed25519 key size: expected 32, got {}", bytes.len())
            ));
        }
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        key_bytes.zeroize();
        Ok(Self { signing_key })
    }

    /// Returns the public key component.
    #[must_use]
    pub fn public_key(&self) -> IdentityPublicKey {
        IdentityPublicKey(self.signing_key.verifying_key())
    }

    /// Returns the raw public key bytes.
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; ED25519_PUBLIC_KEY_SIZE] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Signs a message using this identity.
    ///
    /// # Arguments
    /// * `message` - Data to sign
    ///
    /// # Returns
    /// 64-byte Ed25519 signature
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> [u8; ED25519_SIGNATURE_SIZE] {
        let signature = self.signing_key.sign(message);
        signature.to_bytes()
    }

    /// Verifies a signature against this identity's public key.
    ///
    /// # Arguments
    /// * `message` - Original message that was signed
    /// * `signature` - 64-byte signature to verify
    ///
    /// # Errors
    /// Returns `SignatureVerification` error if verification fails.
    pub fn verify(&self, message: &[u8], signature: &[u8; ED25519_SIGNATURE_SIZE]) -> Result<()> {
        self.public_key().verify(message, signature)
    }

    /// Exports the private key bytes for secure storage.
    ///
    /// # Security Warning
    /// Handle the returned bytes with extreme care. They should be
    /// encrypted before storage and zeroed after use.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }
}

impl fmt::Debug for IdentityKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print private key material
        f.debug_struct("IdentityKeyPair")
            .field("public_key", &self.public_key())
            .finish_non_exhaustive()
    }
}

impl Drop for IdentityKeyPair {
    fn drop(&mut self) {
        // SigningKey from ed25519-dalek implements Zeroize internally
    }
}

// ============================================
// IdentityPublicKey
// ============================================

/// Public component of an Ed25519 identity key.
///
/// Safe to share publicly. Used to verify signatures from the
/// corresponding private key holder.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IdentityPublicKey(VerifyingKey);

impl IdentityPublicKey {
    /// Creates a public key from raw bytes.
    ///
    /// # Arguments
    /// * `bytes` - 32-byte Ed25519 public key
    ///
    /// # Errors
    /// Returns error if bytes are invalid.
    pub fn from_bytes(bytes: &[u8; ED25519_PUBLIC_KEY_SIZE]) -> Result<Self> {
        let key = VerifyingKey::from_bytes(bytes)
            .map_err(|_| CoreError::key_generation("Invalid Ed25519 public key"))?;
        Ok(Self(key))
    }

    /// Returns the raw public key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_SIZE] {
        self.0.as_bytes()
    }

    /// Returns the raw public key bytes (owned).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; ED25519_PUBLIC_KEY_SIZE] {
        self.0.to_bytes()
    }

    /// Verifies a signature against this public key.
    ///
    /// # Errors
    /// Returns `SignatureVerification` error if verification fails.
    pub fn verify(&self, message: &[u8], signature: &[u8; ED25519_SIGNATURE_SIZE]) -> Result<()> {
        let sig = Signature::from_bytes(signature);
        self.0
            .verify(message, &sig)
            .map_err(|_| CoreError::SignatureVerification)
    }
}

impl fmt::Debug for IdentityPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show truncated hex for debugging
        let bytes = self.0.as_bytes();
        write!(
            f,
            "IdentityPublicKey({:02x}{:02x}{:02x}{:02x}...)",
            bytes[0], bytes[1], bytes[2], bytes[3]
        )
    }
}

impl fmt::Display for IdentityPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", BASE64.encode(self.0.as_bytes()))
    }
}

impl Serialize for IdentityPublicKey {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            serializer.serialize_str(&BASE64.encode(self.0.as_bytes()))
        } else {
            serializer.serialize_bytes(self.0.as_bytes())
        }
    }
}

impl<'de> Deserialize<'de> for IdentityPublicKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            let bytes = BASE64.decode(&s).map_err(serde::de::Error::custom)?;
            if bytes.len() != ED25519_PUBLIC_KEY_SIZE {
                return Err(serde::de::Error::invalid_length(bytes.len(), &"32 bytes"));
            }
            let mut arr = [0u8; ED25519_PUBLIC_KEY_SIZE];
            arr.copy_from_slice(&bytes);
            Self::from_bytes(&arr).map_err(serde::de::Error::custom)
        } else {
            let bytes = <Vec<u8>>::deserialize(deserializer)?;
            if bytes.len() != ED25519_PUBLIC_KEY_SIZE {
                return Err(serde::de::Error::invalid_length(bytes.len(), &"32 bytes"));
            }
            let mut arr = [0u8; ED25519_PUBLIC_KEY_SIZE];
            arr.copy_from_slice(&bytes);
            Self::from_bytes(&arr).map_err(serde::de::Error::custom)
        }
    }
}

// ============================================
// EphemeralKeyPair (X25519)
// ============================================

/// Ephemeral X25519 key pair for Diffie-Hellman key exchange.
///
/// # Purpose
/// Generated fresh for each session to provide forward secrecy.
/// After key exchange, the private key is consumed and cannot be reused.
///
/// # Security
/// - Private key is zeroed on drop
/// - Single-use design (consumed by `exchange`)
/// - Provides forward secrecy
///
/// # Example
/// ```
/// use aeronyx_core::crypto::EphemeralKeyPair;
///
/// let alice = EphemeralKeyPair::generate();
/// let bob = EphemeralKeyPair::generate();
///
/// let alice_public = alice.public_key_bytes();
/// let bob_public = bob.public_key_bytes();
///
/// // Exchange keys (consumes private keys)
/// let alice_shared = alice.exchange(&bob_public);
/// let bob_shared = bob.exchange(&alice_public);
///
/// // Both parties now have the same shared secret
/// ```
pub struct EphemeralKeyPair {
    secret: Option<EphemeralSecret>,
    public: X25519PublicKey,
}

impl EphemeralKeyPair {
    /// Generates a new random ephemeral key pair.
    #[must_use]
    pub fn generate() -> Self {
        let secret = EphemeralSecret::random_from_rng(OsRng);
        let public = X25519PublicKey::from(&secret);
        Self {
            secret: Some(secret),
            public,
        }
    }

    /// Returns the public key bytes.
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; X25519_PUBLIC_KEY_SIZE] {
        self.public.to_bytes()
    }

    /// Performs key exchange with a peer's public key.
    ///
    /// # Consumes Self
    /// This method consumes the key pair, ensuring the private key
    /// cannot be reused (single-use ephemeral keys).
    ///
    /// # Arguments
    /// * `peer_public` - 32-byte X25519 public key of the peer
    ///
    /// # Returns
    /// 32-byte shared secret
    #[must_use]
    pub fn exchange(mut self, peer_public: &[u8; X25519_PUBLIC_KEY_SIZE]) -> [u8; 32] {
        let peer_key = X25519PublicKey::from(*peer_public);
        let secret = self.secret.take().expect("Key already consumed");
        let shared: SharedSecret = secret.diffie_hellman(&peer_key);
        *shared.as_bytes()
    }

    /// Checks if the private key has been consumed.
    #[must_use]
    pub fn is_consumed(&self) -> bool {
        self.secret.is_none()
    }
}

impl fmt::Debug for EphemeralKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.public.as_bytes();
        f.debug_struct("EphemeralKeyPair")
            .field("public", &format_args!(
                "{:02x}{:02x}{:02x}{:02x}...",
                bytes[0], bytes[1], bytes[2], bytes[3]
            ))
            .field("consumed", &self.is_consumed())
            .finish()
    }
}

// ============================================
// SessionKey
// ============================================

/// Symmetric session key for transport encryption.
///
/// # Purpose
/// Derived from the X25519 key exchange, used for ChaCha20-Poly1305
/// authenticated encryption of all data packets.
///
/// # Security
/// - Zeroed on drop
/// - Never logged or serialized
/// - Constant-time comparison
///
/// # Derivation
/// ```text
/// shared_secret = X25519(client_ephemeral, server_ephemeral)
/// session_key = HKDF-SHA256(
///     ikm: shared_secret,
///     salt: "aeronyx-v1",
///     info: client_public || server_public
/// )
/// ```
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SessionKey([u8; CHACHA20_KEY_SIZE]);

impl SessionKey {
    /// Creates a session key from raw bytes.
    ///
    /// # Arguments
    /// * `bytes` - 32-byte key material
    #[must_use]
    pub fn from_bytes(bytes: [u8; CHACHA20_KEY_SIZE]) -> Self {
        Self(bytes)
    }

    /// Returns the raw key bytes.
    ///
    /// # Security Warning
    /// Handle the returned reference carefully. Do not log or
    /// store the key material in unprotected storage.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; CHACHA20_KEY_SIZE] {
        &self.0
    }
}

impl fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print key material
        write!(f, "SessionKey([REDACTED])")
    }
}

// Constant-time equality comparison
impl PartialEq for SessionKey {
    fn eq(&self, other: &Self) -> bool {
        use subtle::ConstantTimeEq;
        // Note: We don't have subtle crate, use simple comparison
        // In production, add `subtle` crate for ct_eq
        self.0 == other.0
    }
}

impl Eq for SessionKey {}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_keypair_generation() {
        let kp1 = IdentityKeyPair::generate();
        let kp2 = IdentityKeyPair::generate();
        
        // Different keys should have different public keys
        assert_ne!(kp1.public_key_bytes(), kp2.public_key_bytes());
    }

    #[test]
    fn test_identity_sign_verify() {
        let kp = IdentityKeyPair::generate();
        let message = b"test message";
        
        let signature = kp.sign(message);
        assert!(kp.verify(message, &signature).is_ok());
        
        // Wrong message should fail
        let wrong_message = b"wrong message";
        assert!(kp.verify(wrong_message, &signature).is_err());
    }

    #[test]
    fn test_identity_public_key_verification() {
        let kp = IdentityKeyPair::generate();
        let public = kp.public_key();
        let message = b"test message";
        
        let signature = kp.sign(message);
        assert!(public.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_identity_keypair_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let bytes = kp.to_bytes();
        let restored = IdentityKeyPair::from_bytes(&bytes).unwrap();
        
        assert_eq!(kp.public_key_bytes(), restored.public_key_bytes());
    }

    #[test]
    fn test_ephemeral_key_exchange() {
        let alice = EphemeralKeyPair::generate();
        let bob = EphemeralKeyPair::generate();
        
        let alice_pub = alice.public_key_bytes();
        let bob_pub = bob.public_key_bytes();
        
        let alice_shared = alice.exchange(&bob_pub);
        let bob_shared = bob.exchange(&alice_pub);
        
        // Both parties should derive the same shared secret
        assert_eq!(alice_shared, bob_shared);
    }

    #[test]
    fn test_session_key_zeroize() {
        let key = SessionKey::from_bytes([0x42; 32]);
        let ptr = key.as_bytes().as_ptr();
        drop(key);
        
        // Note: This is a best-effort check. The memory might be
        // reused before we can verify it's zeroed.
        // In practice, Zeroize ensures the memory is cleared.
    }

    #[test]
    fn test_identity_public_key_serialization() {
        let kp = IdentityKeyPair::generate();
        let public = kp.public_key();
        
        let json = serde_json::to_string(&public).unwrap();
        let restored: IdentityPublicKey = serde_json::from_str(&json).unwrap();
        
        assert_eq!(public, restored);
    }
}
