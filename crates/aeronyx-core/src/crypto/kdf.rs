// ============================================
// File: crates/aeronyx-core/src/crypto/kdf.rs
// ============================================
//! # Key Derivation Functions
//!
//! ## Creation Reason
//! Provides secure key derivation using HKDF-SHA256 for deriving
//! session keys from shared secrets.
//!
//! ## Main Functionality
//! - `derive_session_key`: Derives session key from X25519 shared secret
//! - Domain separation via salt and info parameters
//!
//! ## HKDF Construction
//! ```text
//! HKDF-SHA256(
//!     IKM:  X25519 shared secret (32 bytes)
//!     Salt: "aeronyx-v1" (protocol identifier)
//!     Info: client_public || server_public (key binding)
//!     L:    32 bytes (ChaCha20 key size)
//! )
//! ```
//!
//! ## Security Properties
//! - **Key Binding**: Info parameter binds key to specific endpoints
//! - **Domain Separation**: Salt ensures keys differ across protocols
//! - **Uniformity**: HKDF output is indistinguishable from random
//!
//! ## ⚠️ Important Note for Next Developer
//! - Never change HKDF_SALT without protocol version bump
//! - Info parameter order matters for key derivation
//! - Always include both public keys in info for key binding
//!
//! ## Last Modified
//! v0.1.0 - Initial KDF implementation

use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroize;
use tracing::{debug, info, trace};

use super::{CHACHA20_KEY_SIZE, ED25519_PUBLIC_KEY_SIZE, HKDF_INFO_PREFIX, HKDF_SALT};
use crate::crypto::SessionKey;
use crate::error::{CoreError, Result};

// ============================================
// Key Derivation
// ============================================

/// Derives a session key from the X25519 shared secret.
///
/// # Arguments
/// * `shared_secret` - 32-byte X25519 Diffie-Hellman output
/// * `client_public` - Client's Ed25519 public key (for binding)
/// * `server_public` - Server's Ed25519 public key (for binding)
///
/// # Returns
/// A 32-byte session key suitable for ChaCha20-Poly1305.
///
/// # Key Binding
/// Including both public keys in the info parameter ensures that:
/// 1. Keys are bound to the specific session participants
/// 2. Man-in-the-middle attacks are detectable
/// 3. Key reuse across different pairs is prevented
///
/// # Example
/// ```ignore
/// let shared = alice_ephemeral.exchange(&bob_ephemeral_public);
/// let session_key = derive_session_key(
///     &shared,
///     &alice_identity_public,
///     &bob_identity_public,
/// )?;
/// ```
pub fn derive_session_key(
    shared_secret: &[u8; 32],
    client_public: &[u8; ED25519_PUBLIC_KEY_SIZE],
    server_public: &[u8; ED25519_PUBLIC_KEY_SIZE],
) -> Result<SessionKey> {
    info!("[CRYPTO-DEBUG] ========== derive_session_key START ==========");
    
    // Log all inputs
    info!(
        "[CRYPTO-DEBUG] Input - Shared Secret: {}",
        hex::encode(shared_secret)
    );
    info!(
        "[CRYPTO-DEBUG] Input - Client Public Key: {}",
        hex::encode(client_public)
    );
    info!(
        "[CRYPTO-DEBUG] Input - Server Public Key: {}",
        hex::encode(server_public)
    );
    
    // Log HKDF parameters
    debug!(
        "[CRYPTO-DEBUG] HKDF Salt: {} (\"{}\")",
        hex::encode(HKDF_SALT),
        String::from_utf8_lossy(HKDF_SALT)
    );
    debug!(
        "[CRYPTO-DEBUG] HKDF Info Prefix: {} (\"{}\")",
        hex::encode(HKDF_INFO_PREFIX),
        String::from_utf8_lossy(HKDF_INFO_PREFIX)
    );

    // Build info parameter: prefix || client_public || server_public
    let mut info = Vec::with_capacity(
        HKDF_INFO_PREFIX.len() + ED25519_PUBLIC_KEY_SIZE * 2
    );
    info.extend_from_slice(HKDF_INFO_PREFIX);
    info.extend_from_slice(client_public);
    info.extend_from_slice(server_public);
    
    debug!(
        "[CRYPTO-DEBUG] HKDF Info (full): {} ({} bytes)",
        hex::encode(&info),
        info.len()
    );
    debug!(
        "[CRYPTO-DEBUG] HKDF Info breakdown:"
    );
    debug!(
        "[CRYPTO-DEBUG]   - Prefix ({} bytes): {}",
        HKDF_INFO_PREFIX.len(),
        hex::encode(HKDF_INFO_PREFIX)
    );
    debug!(
        "[CRYPTO-DEBUG]   - Client Public ({} bytes): {}",
        ED25519_PUBLIC_KEY_SIZE,
        hex::encode(client_public)
    );
    debug!(
        "[CRYPTO-DEBUG]   - Server Public ({} bytes): {}",
        ED25519_PUBLIC_KEY_SIZE,
        hex::encode(server_public)
    );

    // Perform HKDF-SHA256
    debug!("[CRYPTO-DEBUG] Performing HKDF-SHA256...");
    let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT), shared_secret);
    
    let mut key_bytes = [0u8; CHACHA20_KEY_SIZE];
    match hk.expand(&info, &mut key_bytes) {
        Ok(()) => {
            debug!("[CRYPTO-DEBUG] HKDF expansion successful");
        }
        Err(e) => {
            info!("[CRYPTO-DEBUG] HKDF expansion FAILED: {:?}", e);
            info.zeroize();
            return Err(CoreError::KeyDerivation {
                reason: "HKDF expansion failed".into(),
            });
        }
    }

    info!(
        "[CRYPTO-DEBUG] Output - Derived Session Key: {}",
        hex::encode(&key_bytes)
    );

    // Clear sensitive intermediate data
    info.zeroize();
    debug!("[CRYPTO-DEBUG] Cleared intermediate info buffer");

    info!("[CRYPTO-DEBUG] ========== derive_session_key SUCCESS ==========");
    
    // Summary for easy comparison
    info!("[CRYPTO-DEBUG] ====== KEY DERIVATION SUMMARY ======");
    info!(
        "[CRYPTO-DEBUG]   Shared Secret:     {}",
        hex::encode(shared_secret)
    );
    info!(
        "[CRYPTO-DEBUG]   Client Public:     {}",
        hex::encode(client_public)
    );
    info!(
        "[CRYPTO-DEBUG]   Server Public:     {}",
        hex::encode(server_public)
    );
    info!(
        "[CRYPTO-DEBUG]   Derived Key:       {}",
        hex::encode(&key_bytes)
    );
    info!("[CRYPTO-DEBUG] ====================================");

    Ok(SessionKey::from_bytes(key_bytes))
}

/// Derives multiple keys from a shared secret (for future use).
///
/// # Arguments
/// * `shared_secret` - Input keying material
/// * `salt` - Domain separation salt
/// * `info` - Context/application-specific info
/// * `output_len` - Desired output length in bytes
///
/// # Returns
/// Derived key material of the requested length.
///
/// # Panics
/// Panics if `output_len` exceeds HKDF-SHA256 maximum (255 * 32 bytes).
pub fn hkdf_expand(
    shared_secret: &[u8],
    salt: &[u8],
    info: &[u8],
    output_len: usize,
) -> Result<Vec<u8>> {
    debug!("[CRYPTO-DEBUG] ========== hkdf_expand START ==========");
    debug!(
        "[CRYPTO-DEBUG] Input - shared_secret: {} ({} bytes)",
        hex::encode(shared_secret),
        shared_secret.len()
    );
    debug!(
        "[CRYPTO-DEBUG] Input - salt: {} ({} bytes)",
        hex::encode(salt),
        salt.len()
    );
    debug!(
        "[CRYPTO-DEBUG] Input - info: {} ({} bytes)",
        hex::encode(info),
        info.len()
    );
    debug!("[CRYPTO-DEBUG] Input - output_len: {} bytes", output_len);

    let hk = Hkdf::<Sha256>::new(Some(salt), shared_secret);
    
    let mut output = vec![0u8; output_len];
    match hk.expand(info, &mut output) {
        Ok(()) => {
            debug!(
                "[CRYPTO-DEBUG] Output: {} ({} bytes)",
                hex::encode(&output),
                output.len()
            );
            debug!("[CRYPTO-DEBUG] ========== hkdf_expand SUCCESS ==========");
            Ok(output)
        }
        Err(_) => {
            debug!("[CRYPTO-DEBUG] ========== hkdf_expand FAILED ==========");
            Err(CoreError::KeyDerivation {
                reason: format!("HKDF expansion failed for {} bytes", output_len),
            })
        }
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_session_key() {
        let shared_secret = [0x42u8; 32];
        let client_public = [0x01u8; ED25519_PUBLIC_KEY_SIZE];
        let server_public = [0x02u8; ED25519_PUBLIC_KEY_SIZE];

        let key = derive_session_key(
            &shared_secret,
            &client_public,
            &server_public,
        ).unwrap();

        // Key should be 32 bytes
        assert_eq!(key.as_bytes().len(), 32);
        
        // Key should not be all zeros (would indicate failure)
        assert_ne!(key.as_bytes(), &[0u8; 32]);
    }

    #[test]
    fn test_derive_session_key_deterministic() {
        let shared_secret = [0x42u8; 32];
        let client_public = [0x01u8; ED25519_PUBLIC_KEY_SIZE];
        let server_public = [0x02u8; ED25519_PUBLIC_KEY_SIZE];

        let key1 = derive_session_key(
            &shared_secret,
            &client_public,
            &server_public,
        ).unwrap();

        let key2 = derive_session_key(
            &shared_secret,
            &client_public,
            &server_public,
        ).unwrap();

        // Same inputs should produce same key
        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_derive_session_key_different_inputs() {
        let shared_secret = [0x42u8; 32];
        let client_public = [0x01u8; ED25519_PUBLIC_KEY_SIZE];
        let server_public = [0x02u8; ED25519_PUBLIC_KEY_SIZE];
        let other_public = [0x03u8; ED25519_PUBLIC_KEY_SIZE];

        let key1 = derive_session_key(
            &shared_secret,
            &client_public,
            &server_public,
        ).unwrap();

        let key2 = derive_session_key(
            &shared_secret,
            &client_public,
            &other_public,
        ).unwrap();

        // Different server public should produce different key
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_derive_session_key_order_matters() {
        let shared_secret = [0x42u8; 32];
        let key_a = [0x01u8; ED25519_PUBLIC_KEY_SIZE];
        let key_b = [0x02u8; ED25519_PUBLIC_KEY_SIZE];

        let key1 = derive_session_key(
            &shared_secret,
            &key_a,  // client
            &key_b,  // server
        ).unwrap();

        let key2 = derive_session_key(
            &shared_secret,
            &key_b,  // client (swapped)
            &key_a,  // server (swapped)
        ).unwrap();

        // Order of public keys matters
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_hkdf_expand() {
        let ikm = [0x42u8; 32];
        let salt = b"test-salt";
        let info = b"test-info";

        let output = hkdf_expand(&ikm, salt, info, 64).unwrap();
        
        assert_eq!(output.len(), 64);
        assert_ne!(&output[..32], &[0u8; 32]);
    }
}
