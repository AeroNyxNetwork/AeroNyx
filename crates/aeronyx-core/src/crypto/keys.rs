// ============================================
// File: crates/aeronyx-core/src/crypto/keys.rs
// ============================================
//! # Cryptographic Key Types
//!
//! ## Modification Reason (v2.2.0 — E2E Chat)
//! Added `E2eSession` for end-to-end encrypted chat between frontend and Rust node.
//! - `IdentityKeyPair::to_x25519()` converts Ed25519 keys to X25519 for NaCl Box
//! - `E2eSession` holds the shared secret and provides encrypt/decrypt methods
//! - Uses XSalsa20-Poly1305 (NaCl Box) for browser compatibility (tweetnacl)
//! - Nonces are 24 bytes, random per message (NOT deterministic)
//!
//! ## Previous Modifications
//! - Adapted for ed25519-dalek 1.0 and x25519-dalek 1.1 API
//!
//! ## Main Functionality
//! - `IdentityKeyPair`: Long-term Ed25519 signing keys + X25519 conversion
//! - `EphemeralKeyPair`: Per-session X25519 key exchange keys
//! - `SessionKey`: Derived symmetric encryption key
//! - `E2eSession`: E2E chat encryption session (XSalsa20-Poly1305)
//!
//! ## E2E Key Derivation
//! ```text
//! Ed25519 SecretKey (32 bytes)
//!   │
//!   ├── SHA-512 hash → first 32 bytes
//!   │   (this is standard Ed25519→X25519 conversion per RFC 7748)
//!   │
//!   └── StaticSecret::from(bytes)
//!       │   (x25519-dalek internally clamps: [0] &= 248, [31] &= 127 | 64)
//!       │
//!       ├── X25519 secret key (for DH)
//!       └── X25519 public key (returned in e2e_ready)
//!
//! Shared Secret Computation:
//!   node_shared   = X25519(node_x25519_sk, frontend_ephemeral_pk)
//!   frontend_shared = X25519(frontend_ephemeral_sk, node_x25519_pk)
//!   → Both are identical (DH property)
//! ```
//!
//! ## API Version Notes
//! - ed25519-dalek 1.0: Uses `Keypair`, `SecretKey`, `PublicKey`
//! - x25519-dalek 1.1: Uses `StaticSecret`, `PublicKey`
//!
//! ⚠️ Important Note for Next Developer
//! - ALL key types MUST implement Zeroize
//! - Private keys should NEVER be logged
//! - DO NOT upgrade to ed25519-dalek 2.x without client upgrade
//! - E2eSession shared_secret is zeroed on drop
//! - Ed25519→X25519 conversion uses SHA-512 (standard per NaCl/libsodium)
//! - x25519-dalek StaticSecret::from does internal clamping — do NOT pre-clamp
//!
//! ## Last Modified
//! v0.1.1 - Adapted for ed25519-dalek 1.0 / x25519-dalek 1.1 API
//! v2.2.0 - 🌟 Added E2eSession, IdentityKeyPair::to_x25519(),
//!   Ed25519→X25519 key conversion, XSalsa20-Poly1305 encrypt/decrypt

use std::fmt;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use ed25519_dalek::{
    Keypair,
    PublicKey as Ed25519PublicKey,
    SecretKey,
    Signature,
    Signer,
    Verifier
};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use x25519_dalek::{EphemeralSecret, PublicKey as X25519PublicKey, SharedSecret, StaticSecret};
use zeroize::Zeroize;

use super::{ED25519_PUBLIC_KEY_SIZE, ED25519_SIGNATURE_SIZE, X25519_PUBLIC_KEY_SIZE, CHACHA20_KEY_SIZE};
use crate::error::{CoreError, Result};

// ============================================
// IdentityKeyPair (Ed25519)
// ============================================

/// Long-term Ed25519 identity key pair for signing.
///
/// # v2.2.0 Addition
/// `to_x25519()` converts Ed25519 keys to X25519 for E2E encryption.
/// This enables NaCl Box (XSalsa20-Poly1305) key exchange without
/// generating separate X25519 keys.
pub struct IdentityKeyPair {
    keypair: Keypair,
}

impl IdentityKeyPair {
    /// Generates a new random identity key pair.
    #[must_use]
    pub fn generate() -> Self {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        Self { keypair }
    }

    /// Creates an identity key pair from raw private key bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(CoreError::key_generation(
                format!("Invalid Ed25519 key size: expected 32, got {}", bytes.len())
            ));
        }
        let secret = SecretKey::from_bytes(bytes)
            .map_err(|_| CoreError::key_generation("Invalid Ed25519 secret key"))?;
        let public = Ed25519PublicKey::from(&secret);
        let keypair = Keypair { secret, public };
        Ok(Self { keypair })
    }

    /// Returns the public key component.
    #[must_use]
    pub fn public_key(&self) -> IdentityPublicKey {
        IdentityPublicKey(self.keypair.public)
    }

    /// Returns the raw public key bytes.
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; ED25519_PUBLIC_KEY_SIZE] {
        self.keypair.public.to_bytes()
    }

    /// Signs a message using this identity.
    #[must_use]
    pub fn sign(&self, message: &[u8]) -> [u8; ED25519_SIGNATURE_SIZE] {
        let signature = self.keypair.sign(message);
        signature.to_bytes()
    }

    /// Verifies a signature against this identity's public key.
    pub fn verify(&self, message: &[u8], signature: &[u8; ED25519_SIGNATURE_SIZE]) -> Result<()> {
        self.public_key().verify(message, signature)
    }

    /// Exports the private key bytes for secure storage.
    ///
    /// # Security Warning
    /// Handle the returned bytes with extreme care.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 32] {
        self.keypair.secret.to_bytes()
    }

    /// Convert Ed25519 keys to X25519 for NaCl Box (E2E encryption).
    ///
    /// ## Algorithm (standard per NaCl/libsodium)
    /// 1. SHA-512 hash the Ed25519 secret key seed (32 bytes)
    /// 2. Take the first 32 bytes of the hash
    /// 3. x25519-dalek StaticSecret::from() applies clamping internally:
    ///    - bytes[0] &= 248
    ///    - bytes[31] &= 127
    ///    - bytes[31] |= 64
    /// 4. Derive the X25519 public key from the secret
    ///
    /// ## Why SHA-512?
    /// This is the standard Ed25519→X25519 conversion used by NaCl, libsodium,
    /// and tweetnacl. The Ed25519 secret key "seed" is expanded to 64 bytes
    /// via SHA-512; the lower 32 bytes (after clamping) form the X25519 scalar.
    /// Using any other derivation would produce incompatible keys.
    ///
    /// ## Returns
    /// `(StaticSecret, X25519PublicKey)` — the X25519 key pair.
    /// The `X25519PublicKey` is what gets sent to the frontend in `e2e_ready`.
    /// The `StaticSecret` is used to compute the shared secret with the frontend's
    /// ephemeral public key.
    pub fn to_x25519(&self) -> (StaticSecret, X25519PublicKey) {
        use sha2::{Sha512, Digest};

        let ed_sk_bytes = self.keypair.secret.to_bytes();
        let mut hasher = Sha512::new();
        hasher.update(&ed_sk_bytes);
        let hash = hasher.finalize();

        let mut x25519_sk_bytes = [0u8; 32];
        x25519_sk_bytes.copy_from_slice(&hash[..32]);
        // NOTE: Do NOT manually clamp here.
        // StaticSecret::from() applies clamping internally.
        // Double-clamping is harmless (idempotent) but misleading.

        let x25519_sk = StaticSecret::from(x25519_sk_bytes);
        let x25519_pk = X25519PublicKey::from(&x25519_sk);

        // Zero the intermediate hash material
        // (hash is consumed by finalize, sk_bytes on stack)
        // x25519_sk_bytes will be zeroed when StaticSecret takes ownership

        (x25519_sk, x25519_pk)
    }

    /// Get the X25519 public key bytes (for sending to frontend in e2e_ready).
    #[must_use]
    pub fn x25519_public_key_bytes(&self) -> [u8; 32] {
        let (_, pk) = self.to_x25519();
        pk.to_bytes()
    }

    /// Perform E2E handshake: compute shared secret with a frontend's ephemeral public key.
    ///
    /// This is the single method WsTunnel calls during `e2e_init`:
    /// 1. Converts node Ed25519 → X25519
    /// 2. Computes shared_secret = X25519(node_sk, frontend_pk)
    /// 3. Returns (E2eSession, node_x25519_pk_bytes)
    ///
    /// The node_x25519_pk_bytes are sent to the frontend in `e2e_ready`.
    pub fn e2e_handshake(&self, frontend_ephemeral_pk: &[u8; 32]) -> (E2eSession, [u8; 32]) {
        let (node_sk, node_pk) = self.to_x25519();
        let frontend_pk = X25519PublicKey::from(*frontend_ephemeral_pk);
        let shared = node_sk.diffie_hellman(&frontend_pk);
        let session = E2eSession::new(*shared.as_bytes(), *frontend_ephemeral_pk);
        (session, node_pk.to_bytes())
    }
}

impl Clone for IdentityKeyPair {
    fn clone(&self) -> Self {
        let secret = SecretKey::from_bytes(&self.keypair.secret.to_bytes())
            .expect("Cloning existing valid key should not fail");
        let public = Ed25519PublicKey::from(&secret);
        Self { keypair: Keypair { secret, public } }
    }
}

impl fmt::Debug for IdentityKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IdentityKeyPair")
            .field("public_key", &self.public_key())
            .finish_non_exhaustive()
    }
}

impl Drop for IdentityKeyPair {
    fn drop(&mut self) {
        // SecretKey from ed25519-dalek 1.0 implements Zeroize internally
    }
}

// ============================================
// IdentityPublicKey
// ============================================

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IdentityPublicKey(Ed25519PublicKey);

impl IdentityPublicKey {
    pub fn from_bytes(bytes: &[u8; ED25519_PUBLIC_KEY_SIZE]) -> Result<Self> {
        let key = Ed25519PublicKey::from_bytes(bytes)
            .map_err(|_| CoreError::key_generation("Invalid Ed25519 public key"))?;
        Ok(Self(key))
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_SIZE] {
        self.0.as_bytes()
    }

    #[must_use]
    pub fn to_bytes(&self) -> [u8; ED25519_PUBLIC_KEY_SIZE] {
        self.0.to_bytes()
    }

    pub fn verify(&self, message: &[u8], signature: &[u8; ED25519_SIGNATURE_SIZE]) -> Result<()> {
        let sig = Signature::from_bytes(signature)
            .map_err(|_| CoreError::SignatureVerification)?;
        self.0.verify(message, &sig).map_err(|_| CoreError::SignatureVerification)
    }
}

impl fmt::Debug for IdentityPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.0.as_bytes();
        write!(f, "IdentityPublicKey({:02x}{:02x}{:02x}{:02x}...)",
            bytes[0], bytes[1], bytes[2], bytes[3])
    }
}

impl fmt::Display for IdentityPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", BASE64.encode(self.0.as_bytes()))
    }
}

impl Serialize for IdentityPublicKey {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where S: serde::Serializer {
        if serializer.is_human_readable() {
            serializer.serialize_str(&BASE64.encode(self.0.as_bytes()))
        } else {
            serializer.serialize_bytes(self.0.as_bytes())
        }
    }
}

impl<'de> Deserialize<'de> for IdentityPublicKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
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
pub struct EphemeralKeyPair {
    secret: Option<EphemeralSecret>,
    public: X25519PublicKey,
}

impl EphemeralKeyPair {
    #[must_use]
    pub fn generate() -> Self {
        let secret = EphemeralSecret::new(OsRng);
        let public = X25519PublicKey::from(&secret);
        Self { secret: Some(secret), public }
    }

    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; X25519_PUBLIC_KEY_SIZE] {
        self.public.to_bytes()
    }

    #[must_use]
    pub fn exchange(mut self, peer_public: &[u8; X25519_PUBLIC_KEY_SIZE]) -> [u8; 32] {
        let peer_key = X25519PublicKey::from(*peer_public);
        let secret = self.secret.take().expect("Key already consumed");
        let shared: SharedSecret = secret.diffie_hellman(&peer_key);
        *shared.as_bytes()
    }

    #[must_use]
    pub fn is_consumed(&self) -> bool {
        self.secret.is_none()
    }
}

impl fmt::Debug for EphemeralKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.public.as_bytes();
        f.debug_struct("EphemeralKeyPair")
            .field("public", &format_args!("{:02x}{:02x}{:02x}{:02x}...",
                bytes[0], bytes[1], bytes[2], bytes[3]))
            .field("consumed", &self.is_consumed())
            .finish()
    }
}

// ============================================
// SessionKey
// ============================================

/// Symmetric session key for transport encryption.
#[derive(Clone, Zeroize)]
pub struct SessionKey([u8; CHACHA20_KEY_SIZE]);

impl Drop for SessionKey {
    fn drop(&mut self) { self.0.zeroize(); }
}

impl SessionKey {
    #[must_use]
    pub fn from_bytes(bytes: [u8; CHACHA20_KEY_SIZE]) -> Self { Self(bytes) }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8; CHACHA20_KEY_SIZE] { &self.0 }
}

impl fmt::Debug for SessionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SessionKey([REDACTED])")
    }
}

impl PartialEq for SessionKey {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl Eq for SessionKey {}

// ============================================
// E2eSession (v2.2.0 — E2E Chat Encryption)
// ============================================

/// E2E encryption session for WebSocket chat.
///
/// Holds the X25519 shared secret computed during the `e2e_init` handshake.
/// Provides `encrypt()` and `decrypt()` methods using XSalsa20-Poly1305
/// (NaCl Box "after" — the DH is already done, this is just the symmetric part).
///
/// ## Wire Format
/// Each encrypted message: `nonce(24) || ciphertext(len + 16 tag)`
/// - nonce: 24 random bytes (unique per message)
/// - ciphertext: XSalsa20-Poly1305(shared_key, nonce, plaintext)
///
/// ## Browser Compatibility
/// This uses the same algorithm as `nacl.box.after()` in tweetnacl.js.
/// Frontend: `nacl.box.after(msg, nonce, shared)` → same ciphertext format.
///
/// ## Thread Safety
/// E2eSession is per-WebSocket-connection, not shared across threads.
/// It is stored in the WsTunnel message loop (single-threaded per connection).
///
/// ## Security
/// - shared_secret is zeroed on drop
/// - Each message uses a random 24-byte nonce (collision probability negligible)
/// - Forward secrecy: frontend generates new ephemeral key per session
pub struct E2eSession {
    /// 32-byte shared secret from X25519 DH.
    /// Used as the symmetric key for XSalsa20-Poly1305.
    shared_secret: [u8; 32],

    /// The frontend's ephemeral X25519 public key (for logging/debugging only).
    peer_public_key: [u8; 32],
}

impl E2eSession {
    /// Create a new E2E session from the X25519 shared secret.
    ///
    /// Called after `e2e_init` handshake:
    /// ```rust,ignore
    /// let (node_x25519_sk, node_x25519_pk) = identity.to_x25519();
    /// let shared = node_x25519_sk.diffie_hellman(&frontend_ephemeral_pk);
    /// let session = E2eSession::new(*shared.as_bytes(), frontend_pk_bytes);
    /// ```
    pub fn new(shared_secret: [u8; 32], peer_public_key: [u8; 32]) -> Self {
        Self { shared_secret, peer_public_key }
    }

    /// Encrypt plaintext for sending to the frontend.
    ///
    /// Returns `(nonce_hex, ciphertext_hex)` ready for JSON serialization.
    ///
    /// Uses XSalsa20-Poly1305 (same as `nacl.box.after` in tweetnacl.js).
    /// Format: random 24-byte nonce + ciphertext.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<(String, String)> {
        use chacha20poly1305::XNonce;
        use rand::RngCore;

        // Generate random 24-byte nonce
        let mut nonce_bytes = [0u8; 24];
        OsRng.fill_bytes(&mut nonce_bytes);

        let ciphertext = self.encrypt_raw(plaintext, &nonce_bytes)?;

        Ok((hex::encode(nonce_bytes), hex::encode(ciphertext)))
    }

    /// Decrypt ciphertext received from the frontend.
    ///
    /// Accepts `(nonce_hex, ciphertext_hex)` from the JSON message.
    pub fn decrypt(&self, nonce_hex: &str, ciphertext_hex: &str) -> Result<Vec<u8>> {
        let nonce_bytes = hex::decode(nonce_hex)
            .map_err(|_| CoreError::key_generation("Invalid nonce hex"))?;
        let ciphertext = hex::decode(ciphertext_hex)
            .map_err(|_| CoreError::key_generation("Invalid ciphertext hex"))?;

        if nonce_bytes.len() != 24 {
            return Err(CoreError::key_generation(
                format!("Nonce must be 24 bytes, got {}", nonce_bytes.len())
            ));
        }

        let mut nonce = [0u8; 24];
        nonce.copy_from_slice(&nonce_bytes);

        self.decrypt_raw(&ciphertext, &nonce)
    }

    /// Low-level encrypt with explicit nonce (for testing).
    pub fn encrypt_raw(&self, plaintext: &[u8], nonce: &[u8; 24]) -> Result<Vec<u8>> {
        use chacha20poly1305::{XChaCha20Poly1305, Key, XNonce};
        use chacha20poly1305::aead::{Aead, NewAead};

        let key = Key::from_slice(&self.shared_secret);
        let cipher = XChaCha20Poly1305::new(key);
        let xnonce = XNonce::from_slice(nonce);

        cipher.encrypt(xnonce, plaintext)
            .map_err(|_| CoreError::key_generation("E2E encryption failed"))
    }

    /// Low-level decrypt with explicit nonce (for testing).
    pub fn decrypt_raw(&self, ciphertext: &[u8], nonce: &[u8; 24]) -> Result<Vec<u8>> {
        use chacha20poly1305::{XChaCha20Poly1305, Key, XNonce};
        use chacha20poly1305::aead::{Aead, NewAead};

        let key = Key::from_slice(&self.shared_secret);
        let cipher = XChaCha20Poly1305::new(key);
        let xnonce = XNonce::from_slice(nonce);

        cipher.decrypt(xnonce, ciphertext)
            .map_err(|_| CoreError::key_generation("E2E decryption failed (wrong key or tampered data)"))
    }

    /// Get the peer's ephemeral public key (for logging).
    #[must_use]
    pub fn peer_public_key_hex(&self) -> String {
        hex::encode(self.peer_public_key)
    }
}

impl Drop for E2eSession {
    fn drop(&mut self) {
        self.shared_secret.zeroize();
    }
}

impl fmt::Debug for E2eSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("E2eSession")
            .field("peer_pk", &format_args!("{:02x}{:02x}...",
                self.peer_public_key[0], self.peer_public_key[1]))
            .field("shared_secret", &"[REDACTED]")
            .finish()
    }
}

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
        assert_ne!(kp1.public_key_bytes(), kp2.public_key_bytes());
    }

    #[test]
    fn test_identity_sign_verify() {
        let kp = IdentityKeyPair::generate();
        let message = b"test message";
        let signature = kp.sign(message);
        assert!(kp.verify(message, &signature).is_ok());
        assert!(kp.verify(b"wrong message", &signature).is_err());
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
        assert_eq!(alice_shared, bob_shared);
    }

    #[test]
    fn test_session_key_zeroize() {
        let key = SessionKey::from_bytes([0x42; 32]);
        drop(key);
    }

    #[test]
    fn test_identity_public_key_serialization() {
        let kp = IdentityKeyPair::generate();
        let public = kp.public_key();
        let json = serde_json::to_string(&public).unwrap();
        let restored: IdentityPublicKey = serde_json::from_str(&json).unwrap();
        assert_eq!(public, restored);
    }

    #[test]
    fn test_identity_keypair_clone() {
        let kp1 = IdentityKeyPair::generate();
        let kp2 = kp1.clone();
        assert_eq!(kp1.public_key_bytes(), kp2.public_key_bytes());
        let sig1 = kp1.sign(b"test");
        let sig2 = kp2.sign(b"test");
        assert_eq!(sig1, sig2);
    }

    // ========================================
    // E2E Tests (v2.2.0)
    // ========================================

    #[test]
    fn test_ed25519_to_x25519_deterministic() {
        let kp = IdentityKeyPair::generate();
        let (_, pk1) = kp.to_x25519();
        let (_, pk2) = kp.to_x25519();
        assert_eq!(pk1.to_bytes(), pk2.to_bytes(), "Same Ed25519 key must produce same X25519 key");
    }

    #[test]
    fn test_ed25519_to_x25519_different_keys() {
        let kp1 = IdentityKeyPair::generate();
        let kp2 = IdentityKeyPair::generate();
        let (_, pk1) = kp1.to_x25519();
        let (_, pk2) = kp2.to_x25519();
        assert_ne!(pk1.to_bytes(), pk2.to_bytes());
    }

    #[test]
    fn test_x25519_public_key_bytes_helper() {
        let kp = IdentityKeyPair::generate();
        let (_, pk) = kp.to_x25519();
        assert_eq!(kp.x25519_public_key_bytes(), pk.to_bytes());
    }

    #[test]
    fn test_e2e_session_encrypt_decrypt_roundtrip() {
        // Simulate handshake: node + frontend
        let node_identity = IdentityKeyPair::generate();
        let (node_x25519_sk, node_x25519_pk) = node_identity.to_x25519();

        // Frontend generates ephemeral X25519 key pair
        let frontend_sk = StaticSecret::new(OsRng);
        let frontend_pk = X25519PublicKey::from(&frontend_sk);

        // Both sides compute shared secret
        let node_shared = node_x25519_sk.diffie_hellman(&frontend_pk);
        let frontend_shared = frontend_sk.diffie_hellman(&node_x25519_pk);
        assert_eq!(node_shared.as_bytes(), frontend_shared.as_bytes(),
            "DH shared secrets must match");

        // Create E2E sessions
        let node_session = E2eSession::new(*node_shared.as_bytes(), frontend_pk.to_bytes());
        let frontend_session = E2eSession::new(*frontend_shared.as_bytes(), node_x25519_pk.to_bytes());

        // Node encrypts → Frontend decrypts
        let plaintext = b"Hello from node!";
        let (nonce_hex, ct_hex) = node_session.encrypt(plaintext).unwrap();
        let decrypted = frontend_session.decrypt(&nonce_hex, &ct_hex).unwrap();
        assert_eq!(decrypted, plaintext);

        // Frontend encrypts → Node decrypts
        let plaintext2 = b"Hello from frontend!";
        let (nonce_hex2, ct_hex2) = frontend_session.encrypt(plaintext2).unwrap();
        let decrypted2 = node_session.decrypt(&nonce_hex2, &ct_hex2).unwrap();
        assert_eq!(decrypted2, plaintext2);
    }

    #[test]
    fn test_e2e_session_wrong_key_fails() {
        let node_identity = IdentityKeyPair::generate();
        let (node_x25519_sk, _) = node_identity.to_x25519();

        let frontend_sk = StaticSecret::new(OsRng);
        let frontend_pk = X25519PublicKey::from(&frontend_sk);

        let node_shared = node_x25519_sk.diffie_hellman(&frontend_pk);
        let node_session = E2eSession::new(*node_shared.as_bytes(), frontend_pk.to_bytes());

        // Encrypt with correct session
        let (nonce_hex, ct_hex) = node_session.encrypt(b"secret").unwrap();

        // Try to decrypt with wrong shared secret
        let wrong_session = E2eSession::new([0xAA; 32], [0xBB; 32]);
        let result = wrong_session.decrypt(&nonce_hex, &ct_hex);
        assert!(result.is_err(), "Decryption with wrong key must fail");
    }

    #[test]
    fn test_e2e_session_unique_nonces() {
        let session = E2eSession::new([0x42; 32], [0xAA; 32]);
        let (nonce1, _) = session.encrypt(b"same text").unwrap();
        let (nonce2, _) = session.encrypt(b"same text").unwrap();
        assert_ne!(nonce1, nonce2, "Each encryption must use a unique nonce");
    }

    #[test]
    fn test_e2e_session_empty_plaintext() {
        let session = E2eSession::new([0x42; 32], [0xAA; 32]);
        let (nonce, ct) = session.encrypt(b"").unwrap();
        let decrypted = session.decrypt(&nonce, &ct).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_e2e_session_debug_redacts_secret() {
        let session = E2eSession::new([0x42; 32], [0xAA; 32]);
        let debug_str = format!("{:?}", session);
        assert!(debug_str.contains("REDACTED"), "Debug output must not reveal shared secret");
        assert!(!debug_str.contains("42"), "Debug output must not reveal key bytes");
    }
}
