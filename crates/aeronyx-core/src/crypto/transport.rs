// ============================================
// File: crates/aeronyx-core/src/crypto/transport.rs
// ============================================
//! # Transport Encryption
//!
//! ## Creation Reason
//! Provides authenticated encryption for data packets using
//! ChaCha20-Poly1305 AEAD cipher.
//!
//! ## Main Functionality
//! - `TransportCrypto`: Trait for transport encryption/decryption
//! - `DefaultTransportCrypto`: Production implementation
//! - Nonce construction from counters
//! - Associated data handling for authentication
//!
//! ## Packet Format
//! ```text
//! ┌────────────────────────────────────────────────────┐
//! │ Session ID (16 bytes)          │ ← AAD (authenticated) │
//! ├────────────────────────────────────────────────────┤
//! │ Counter (8 bytes)              │ ← Used for nonce      │
//! ├────────────────────────────────────────────────────┤
//! │ Encrypted Payload (variable)   │ ← ChaCha20 ciphertext │
//! │ ├─ IP packet data              │                       │
//! │ └─ Poly1305 Tag (16 bytes)     │ ← Authentication tag  │
//! └────────────────────────────────────────────────────┘
//! ```
//!
//! ## Nonce Construction
//! ```text
//! nonce (12 bytes) = counter (8 bytes LE) || 0x00000000 (4 bytes)
//! ```
//!
//! ## Security Properties
//! - **AEAD**: Authenticated Encryption with Associated Data
//! - **Replay Protection**: Counter must be monotonically increasing
//! - **Session Binding**: Session ID included in AAD
//!
//! ## ⚠️ Important Note for Next Developer
//! - Counter MUST be unique per packet per session key
//! - Never reuse (key, nonce) pair - catastrophic security failure
//! - Counter overflow should trigger session rekeying
//!
//! ## Last Modified
//! v0.1.0 - Initial transport crypto implementation

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};

use crate::crypto::keys::SessionKey;
use crate::error::{CoreError, Result};

use super::{CHACHA20_NONCE_SIZE, POLY1305_TAG_SIZE};

// ============================================
// Constants
// ============================================

/// Maximum plaintext size for a single packet.
/// This is limited by UDP MTU and protocol overhead.
pub const MAX_PLAINTEXT_SIZE: usize = 65535 - 24; // UDP payload - header

/// Overhead added by encryption (auth tag).
pub const ENCRYPTION_OVERHEAD: usize = POLY1305_TAG_SIZE;

// ============================================
// TransportCrypto Trait
// ============================================

/// Trait for transport-layer encryption operations.
///
/// # Purpose
/// Abstracts the encryption/decryption operations to allow:
/// - Testing with mock implementations
/// - Alternative cipher suites
/// - Hardware acceleration integration
pub trait TransportCrypto: Send + Sync {
    /// Encrypts a plaintext packet.
    ///
    /// # Arguments
    /// * `key` - 32-byte session key
    /// * `counter` - Monotonically increasing packet counter
    /// * `session_id` - 16-byte session identifier (used as AAD)
    /// * `plaintext` - Data to encrypt (typically an IP packet)
    /// * `output` - Buffer to write ciphertext (must be plaintext.len() + 16)
    ///
    /// # Returns
    /// Number of bytes written to output.
    ///
    /// # Errors
    /// - `Encryption`: If encryption fails (shouldn't happen with valid inputs)
    fn encrypt(
        &self,
        key: &SessionKey,
        counter: u64,
        session_id: &[u8; 16],
        plaintext: &[u8],
        output: &mut [u8],
    ) -> Result<usize>;

    /// Decrypts a ciphertext packet.
    ///
    /// # Arguments
    /// * `key` - 32-byte session key
    /// * `counter` - Expected packet counter (from header)
    /// * `session_id` - 16-byte session identifier (used as AAD)
    /// * `ciphertext` - Data to decrypt (includes auth tag)
    /// * `output` - Buffer to write plaintext (must be ciphertext.len() - 16)
    ///
    /// # Returns
    /// Number of bytes written to output.
    ///
    /// # Errors
    /// - `Decryption`: If authentication fails (tampered or wrong key)
    fn decrypt(
        &self,
        key: &SessionKey,
        counter: u64,
        session_id: &[u8; 16],
        ciphertext: &[u8],
        output: &mut [u8],
    ) -> Result<usize>;

    /// Returns the encryption overhead in bytes.
    fn overhead(&self) -> usize;
}

// ============================================
// DefaultTransportCrypto
// ============================================

/// Default implementation using ChaCha20-Poly1305.
#[derive(Debug, Default, Clone)]
pub struct DefaultTransportCrypto;

impl DefaultTransportCrypto {
    /// Creates a new instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Constructs a nonce from a counter value.
    ///
    /// # Format
    /// ```text
    /// nonce[0..8]  = counter (little-endian)
    /// nonce[8..12] = 0x00000000 (padding)
    /// ```
    fn make_nonce(counter: u64) -> Nonce {
        let mut nonce = [0u8; CHACHA20_NONCE_SIZE];
        nonce[..8].copy_from_slice(&counter.to_le_bytes());
        // nonce[8..12] remains zero (implicit padding)
        Nonce::from(nonce)
    }
}

impl TransportCrypto for DefaultTransportCrypto {
    fn encrypt(
        &self,
        key: &SessionKey,
        counter: u64,
        session_id: &[u8; 16],
        plaintext: &[u8],
        output: &mut [u8],
    ) -> Result<usize> {
        // Validate output buffer size
        let required_len = plaintext.len() + POLY1305_TAG_SIZE;
        if output.len() < required_len {
            return Err(CoreError::Encryption {
                context: format!(
                    "Output buffer too small: need {}, have {}",
                    required_len, output.len()
                ),
            });
        }

        // Create cipher and nonce
        let cipher = ChaCha20Poly1305::new_from_slice(key.as_bytes())
            .map_err(|_| CoreError::Encryption {
                context: "Failed to create cipher".into(),
            })?;
        let nonce = Self::make_nonce(counter);

        // Encrypt with session_id as associated data
        let ciphertext = cipher
            .encrypt(&nonce, chacha20poly1305::aead::Payload {
                msg: plaintext,
                aad: session_id,
            })
            .map_err(|_| CoreError::Encryption {
                context: "ChaCha20-Poly1305 encryption failed".into(),
            })?;

        // Copy to output buffer
        output[..ciphertext.len()].copy_from_slice(&ciphertext);

        Ok(ciphertext.len())
    }

    fn decrypt(
        &self,
        key: &SessionKey,
        counter: u64,
        session_id: &[u8; 16],
        ciphertext: &[u8],
        output: &mut [u8],
    ) -> Result<usize> {
        // Validate ciphertext has at least the auth tag
        if ciphertext.len() < POLY1305_TAG_SIZE {
            return Err(CoreError::Decryption);
        }

        // Validate output buffer size
        let plaintext_len = ciphertext.len() - POLY1305_TAG_SIZE;
        if output.len() < plaintext_len {
            return Err(CoreError::Decryption);
        }

        // Create cipher and nonce
        let cipher = ChaCha20Poly1305::new_from_slice(key.as_bytes())
            .map_err(|_| CoreError::Decryption)?;
        let nonce = Self::make_nonce(counter);

        // Decrypt with session_id as associated data
        let plaintext = cipher
            .decrypt(&nonce, chacha20poly1305::aead::Payload {
                msg: ciphertext,
                aad: session_id,
            })
            .map_err(|_| CoreError::Decryption)?;

        // Copy to output buffer
        output[..plaintext.len()].copy_from_slice(&plaintext);

        Ok(plaintext.len())
    }

    fn overhead(&self) -> usize {
        POLY1305_TAG_SIZE
    }
}

// ============================================
// Convenience Functions
// ============================================

/// Encrypts data using the default transport crypto.
///
/// # Arguments
/// * `key` - Session key
/// * `counter` - Packet counter
/// * `session_id` - Session identifier
/// * `plaintext` - Data to encrypt
///
/// # Returns
/// Encrypted data including authentication tag.
pub fn encrypt_packet(
    key: &SessionKey,
    counter: u64,
    session_id: &[u8; 16],
    plaintext: &[u8],
) -> Result<Vec<u8>> {
    let crypto = DefaultTransportCrypto::new();
    let mut output = vec![0u8; plaintext.len() + POLY1305_TAG_SIZE];
    let len = crypto.encrypt(key, counter, session_id, plaintext, &mut output)?;
    output.truncate(len);
    Ok(output)
}

/// Decrypts data using the default transport crypto.
///
/// # Arguments
/// * `key` - Session key
/// * `counter` - Packet counter
/// * `session_id` - Session identifier
/// * `ciphertext` - Data to decrypt
///
/// # Returns
/// Decrypted plaintext.
pub fn decrypt_packet(
    key: &SessionKey,
    counter: u64,
    session_id: &[u8; 16],
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    let crypto = DefaultTransportCrypto::new();
    let mut output = vec![0u8; ciphertext.len()];
    let len = crypto.decrypt(key, counter, session_id, ciphertext, &mut output)?;
    output.truncate(len);
    Ok(output)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> SessionKey {
        SessionKey::from_bytes([0x42u8; 32])
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = test_key();
        let session_id = [0x01u8; 16];
        let counter = 1u64;
        let plaintext = b"Hello, AeroNyx!";

        let ciphertext = encrypt_packet(&key, counter, &session_id, plaintext).unwrap();
        
        // Ciphertext should be larger due to auth tag
        assert_eq!(ciphertext.len(), plaintext.len() + POLY1305_TAG_SIZE);

        let decrypted = decrypt_packet(&key, counter, &session_id, &ciphertext).unwrap();
        
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_different_counters_produce_different_ciphertext() {
        let key = test_key();
        let session_id = [0x01u8; 16];
        let plaintext = b"Hello, AeroNyx!";

        let ct1 = encrypt_packet(&key, 1, &session_id, plaintext).unwrap();
        let ct2 = encrypt_packet(&key, 2, &session_id, plaintext).unwrap();

        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_wrong_key_fails_decryption() {
        let key1 = test_key();
        let key2 = SessionKey::from_bytes([0x43u8; 32]);
        let session_id = [0x01u8; 16];
        let counter = 1u64;
        let plaintext = b"Hello, AeroNyx!";

        let ciphertext = encrypt_packet(&key1, counter, &session_id, plaintext).unwrap();
        
        // Decryption with wrong key should fail
        let result = decrypt_packet(&key2, counter, &session_id, &ciphertext);
        assert!(matches!(result, Err(CoreError::Decryption)));
    }

    #[test]
    fn test_wrong_counter_fails_decryption() {
        let key = test_key();
        let session_id = [0x01u8; 16];
        let plaintext = b"Hello, AeroNyx!";

        let ciphertext = encrypt_packet(&key, 1, &session_id, plaintext).unwrap();
        
        // Decryption with wrong counter should fail
        let result = decrypt_packet(&key, 2, &session_id, &ciphertext);
        assert!(matches!(result, Err(CoreError::Decryption)));
    }

    #[test]
    fn test_wrong_session_id_fails_decryption() {
        let key = test_key();
        let session_id1 = [0x01u8; 16];
        let session_id2 = [0x02u8; 16];
        let counter = 1u64;
        let plaintext = b"Hello, AeroNyx!";

        let ciphertext = encrypt_packet(&key, counter, &session_id1, plaintext).unwrap();
        
        // Decryption with wrong session_id should fail
        let result = decrypt_packet(&key, counter, &session_id2, &ciphertext);
        assert!(matches!(result, Err(CoreError::Decryption)));
    }

    #[test]
    fn test_tampered_ciphertext_fails_decryption() {
        let key = test_key();
        let session_id = [0x01u8; 16];
        let counter = 1u64;
        let plaintext = b"Hello, AeroNyx!";

        let mut ciphertext = encrypt_packet(&key, counter, &session_id, plaintext).unwrap();
        
        // Tamper with ciphertext
        ciphertext[0] ^= 0xFF;
        
        // Decryption should fail
        let result = decrypt_packet(&key, counter, &session_id, &ciphertext);
        assert!(matches!(result, Err(CoreError::Decryption)));
    }

    #[test]
    fn test_empty_plaintext() {
        let key = test_key();
        let session_id = [0x01u8; 16];
        let counter = 1u64;
        let plaintext = b"";

        let ciphertext = encrypt_packet(&key, counter, &session_id, plaintext).unwrap();
        
        // Should just be the auth tag
        assert_eq!(ciphertext.len(), POLY1305_TAG_SIZE);

        let decrypted = decrypt_packet(&key, counter, &session_id, &ciphertext).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_large_plaintext() {
        let key = test_key();
        let session_id = [0x01u8; 16];
        let counter = 1u64;
        let plaintext = vec![0x42u8; 10000];

        let ciphertext = encrypt_packet(&key, counter, &session_id, &plaintext).unwrap();
        let decrypted = decrypt_packet(&key, counter, &session_id, &ciphertext).unwrap();
        
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_nonce_construction() {
        let nonce1 = DefaultTransportCrypto::make_nonce(1);
        let nonce2 = DefaultTransportCrypto::make_nonce(2);
        
        // Different counters should produce different nonces
        assert_ne!(nonce1.as_slice(), nonce2.as_slice());
        
        // Counter 1 should be at start of nonce
        let expected: [u8; 12] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(nonce1.as_slice(), &expected);
    }
}
