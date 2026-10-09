// ============================================
// File: crates/aeronyx-mailbox-wire/src/crypto.rs
// ============================================
//! Minimal crypto surface used by the mailbox wire modules.
//!
//! [MAILBOX-WIRE 2026-10-09 by Claude] Same names and call shapes as the
//! `aeronyx-core::crypto` types the mailbox code was written against, backed
//! by the dalek 4 family so the Flutter client can link this crate without a
//! second Ed25519 stack. Two deliberate differences:
//! - every verification is `verify_strict` (non-canonical S and small-order
//!   keys are rejected), one policy for node and client;
//! - no logging, no I/O: bytes in, bytes out.
//!
//! Byte-compatibility with core (signatures, X25519 derivation, AEAD) is
//! proven by the golden vectors and interop tests, not assumed.

use std::fmt;

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::{Digest, Sha512};
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::Zeroize;

/// Ed25519 public key size.
pub const ED25519_PUBLIC_KEY_SIZE: usize = 32;
/// Ed25519 signature size.
pub const ED25519_SIGNATURE_SIZE: usize = 64;
/// X25519 public key size.
pub const X25519_PUBLIC_KEY_SIZE: usize = 32;

/// Crypto failure. Carries no key or message material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CryptoError {
    /// A key had an invalid length or encoding.
    #[error("invalid key")]
    InvalidKey,
    /// A signature did not verify under the strict policy.
    #[error("signature rejected")]
    SignatureRejected,
    /// AEAD sealing or opening failed.
    #[error("aead failure")]
    Aead,
}

/// Ed25519 identity key pair (32-byte seed).
pub struct IdentityKeyPair {
    signing: SigningKey,
}

impl IdentityKeyPair {
    /// Generates a fresh key pair from the OS RNG.
    #[must_use]
    pub fn generate() -> Self {
        Self {
            signing: SigningKey::generate(&mut OsRng),
        }
    }

    /// Builds a key pair from a 32-byte seed.
    ///
    /// # Errors
    /// Fails if `bytes` is not exactly 32 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        let seed: [u8; 32] = bytes.try_into().map_err(|_| CryptoError::InvalidKey)?;
        Ok(Self {
            signing: SigningKey::from_bytes(&seed),
        })
    }

    /// Public key wrapper.
    #[must_use]
    pub fn public_key(&self) -> IdentityPublicKey {
        IdentityPublicKey(self.signing.verifying_key())
    }

    /// Raw public key bytes.
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; ED25519_PUBLIC_KEY_SIZE] {
        self.signing.verifying_key().to_bytes()
    }

    /// Deterministic Ed25519 signature (RFC 8032).
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> [u8; ED25519_SIGNATURE_SIZE] {
        self.signing.sign(message).to_bytes()
    }

    /// Strict verification against this key pair's public key.
    ///
    /// # Errors
    /// Fails if the signature does not verify strictly.
    pub fn verify(
        &self,
        message: &[u8],
        signature: &[u8; ED25519_SIGNATURE_SIZE],
    ) -> Result<(), CryptoError> {
        self.public_key().verify(message, signature)
    }

    /// The 32-byte seed. Callers must zeroize their copy.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    /// X25519 key pair derived exactly as core does: the first 32 bytes of
    /// SHA-512(seed), clamped by `StaticSecret`.
    #[must_use]
    pub fn to_x25519(&self) -> (StaticSecret, X25519PublicKey) {
        let mut seed = self.signing.to_bytes();
        let hash = Sha512::digest(seed);
        seed.zeroize();
        let mut secret_bytes = [0u8; 32];
        secret_bytes.copy_from_slice(&hash[..32]);
        let secret = StaticSecret::from(secret_bytes);
        secret_bytes.zeroize();
        let public = X25519PublicKey::from(&secret);
        (secret, public)
    }

    /// X25519 public key bytes for this identity.
    #[must_use]
    pub fn x25519_public_key_bytes(&self) -> [u8; X25519_PUBLIC_KEY_SIZE] {
        self.to_x25519().1.to_bytes()
    }
}

impl Clone for IdentityKeyPair {
    fn clone(&self) -> Self {
        Self {
            signing: self.signing.clone(),
        }
    }
}

impl fmt::Debug for IdentityKeyPair {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IdentityKeyPair")
            .field("secret", &"<redacted>")
            .finish()
    }
}

/// Ed25519 public key with strict verification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IdentityPublicKey(VerifyingKey);

impl IdentityPublicKey {
    /// Parses a public key. Small-order and non-canonical points are rejected
    /// at verification time by `verify_strict`.
    ///
    /// # Errors
    /// Fails if the bytes are not a valid compressed Edwards point.
    pub fn from_bytes(bytes: &[u8; ED25519_PUBLIC_KEY_SIZE]) -> Result<Self, CryptoError> {
        VerifyingKey::from_bytes(bytes)
            .map(Self)
            .map_err(|_| CryptoError::InvalidKey)
    }

    /// Raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_SIZE] {
        self.0.as_bytes()
    }

    /// Raw bytes by value.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; ED25519_PUBLIC_KEY_SIZE] {
        self.0.to_bytes()
    }

    /// Strict Ed25519 verification (the single policy of this crate).
    ///
    /// # Errors
    /// Fails for a wrong, malleable (non-canonical S) or small-order-key
    /// signature.
    pub fn verify(
        &self,
        message: &[u8],
        signature: &[u8; ED25519_SIGNATURE_SIZE],
    ) -> Result<(), CryptoError> {
        let signature = Signature::from_bytes(signature);
        self.0
            .verify_strict(message, &signature)
            .map_err(|_| CryptoError::SignatureRejected)
    }
}

impl fmt::Debug for IdentityPublicKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IdentityPublicKey({:02x?}..)",
            &self.0.as_bytes()[..4]
        )
    }
}

/// One-shot X25519 ephemeral key pair.
pub struct EphemeralKeyPair {
    secret: Option<StaticSecret>,
    public: [u8; X25519_PUBLIC_KEY_SIZE],
}

impl EphemeralKeyPair {
    /// Fresh ephemeral key from the OS RNG.
    #[must_use]
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        OsRng.fill_bytes(&mut bytes);
        let pair = Self::from_secret_bytes(bytes);
        bytes.zeroize();
        pair
    }

    /// Deterministic ephemeral key for golden-vector tests only.
    #[must_use]
    pub fn from_secret_bytes(bytes: [u8; 32]) -> Self {
        let secret = StaticSecret::from(bytes);
        let public = X25519PublicKey::from(&secret).to_bytes();
        Self {
            secret: Some(secret),
            public,
        }
    }

    /// Public key bytes.
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; X25519_PUBLIC_KEY_SIZE] {
        self.public
    }

    /// Consumes the secret and returns the raw shared secret.
    #[must_use]
    pub fn exchange(mut self, peer_public: &[u8; X25519_PUBLIC_KEY_SIZE]) -> [u8; 32] {
        let secret = self.secret.take().expect("ephemeral key used once");
        secret
            .diffie_hellman(&X25519PublicKey::from(*peer_public))
            .to_bytes()
    }
}

impl fmt::Debug for EphemeralKeyPair {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EphemeralKeyPair")
            .field("secret", &"<redacted>")
            .finish()
    }
}

/// XChaCha20-Poly1305 session over a 32-byte key, as core `E2eSession`.
pub struct E2eSession {
    key: [u8; 32],
}

impl E2eSession {
    /// The peer key is kept only for API parity with core.
    #[must_use]
    pub fn new(key: [u8; 32], _peer_public_key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Seals with an explicit 24-byte nonce.
    ///
    /// # Errors
    /// Fails only on AEAD error.
    pub fn encrypt_raw(&self, plaintext: &[u8], nonce: &[u8; 24]) -> Result<Vec<u8>, CryptoError> {
        XChaCha20Poly1305::new(Key::from_slice(&self.key))
            .encrypt(XNonce::from_slice(nonce), plaintext)
            .map_err(|_| CryptoError::Aead)
    }

    /// Opens with an explicit 24-byte nonce.
    ///
    /// # Errors
    /// Fails on a wrong key or tampered ciphertext.
    pub fn decrypt_raw(&self, ciphertext: &[u8], nonce: &[u8; 24]) -> Result<Vec<u8>, CryptoError> {
        XChaCha20Poly1305::new(Key::from_slice(&self.key))
            .decrypt(XNonce::from_slice(nonce), ciphertext)
            .map_err(|_| CryptoError::Aead)
    }
}

impl Drop for E2eSession {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

impl fmt::Debug for E2eSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("E2eSession")
            .field("key", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_policy_rejects_non_canonical_s() {
        let pair = IdentityKeyPair::from_bytes(&[7u8; 32]).unwrap();
        let message = b"aeronyx";
        let mut signature = pair.sign(message);
        assert!(pair.verify(message, &signature).is_ok());
        // S' = S + L (group order) is the classic malleable twin.
        const L: [u8; 32] = [
            0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9,
            0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x10,
        ];
        let mut carry = 0u16;
        for i in 0..32 {
            let sum = u16::from(signature[32 + i]) + u16::from(L[i]) + carry;
            signature[32 + i] = (sum & 0xff) as u8;
            carry = sum >> 8;
        }
        assert_eq!(
            pair.verify(message, &signature),
            Err(CryptoError::SignatureRejected)
        );
    }

    #[test]
    fn strict_policy_rejects_small_order_public_key() {
        // The identity point (y = 1) has order 1; any "signature" from it is
        // rejected by verify_strict.
        let mut identity_point = [0u8; 32];
        identity_point[0] = 1;
        let key = IdentityPublicKey::from_bytes(&identity_point).unwrap();
        let mut signature = [0u8; 64];
        signature[0] = 1;
        assert!(key.verify(b"anything", &signature).is_err());
    }
}
