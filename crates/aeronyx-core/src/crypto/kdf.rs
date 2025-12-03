// ============================================
// File: crates/aeronyx-core/src/crypto/kdf.rs
// ============================================
//! # Key Derivation Functions (Enhanced Debug Version)
//!
//! ## Modification Reason
//! Added extensive debugging to diagnose session key derivation issues
//!
//! ## Main Functionality
//! - `derive_session_key`: Derives session key from X25519 shared secret
//! - Domain separation via salt and info parameters
//! - **ENHANCED**: Detailed KDF debugging
//!
//! ## Last Modified
//! v0.1.1 - Enhanced KDF debugging for troubleshooting

use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroize;
use tracing::info;

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
pub fn derive_session_key(
    shared_secret: &[u8; 32],
    client_public: &[u8; ED25519_PUBLIC_KEY_SIZE],
    server_public: &[u8; ED25519_PUBLIC_KEY_SIZE],
) -> Result<SessionKey> {
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("[KDF] 🔑 DERIVING SESSION KEY");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    info!("[KDF] 📥 Input parameters:");
    info!("[KDF]   Shared Secret (hex): {}", hex::encode(shared_secret));
    info!("[KDF]   Client Public (hex): {}", hex::encode(client_public));
    info!("[KDF]   Server Public (hex): {}", hex::encode(server_public));
    
    info!("[KDF] 🧂 HKDF parameters:");
    info!("[KDF]   Salt: {:?}", std::str::from_utf8(HKDF_SALT).unwrap_or("<binary>"));
    info!("[KDF]   Info Prefix: {:?}", std::str::from_utf8(HKDF_INFO_PREFIX).unwrap_or("<binary>"));

    // Build info parameter: prefix || client_public || server_public
    let mut info = Vec::with_capacity(
        HKDF_INFO_PREFIX.len() + ED25519_PUBLIC_KEY_SIZE * 2
    );
    info.extend_from_slice(HKDF_INFO_PREFIX);
    info.extend_from_slice(client_public);
    info.extend_from_slice(server_public);

    info!("[KDF] 📋 Constructed Info parameter:");
    info!("[KDF]   Length: {} bytes", info.len());
    info!("[KDF]   Info (hex): {}", hex::encode(&info));
    info!("[KDF]   Info structure:");
    info!("[KDF]     - Prefix ({} bytes): {}", 
        HKDF_INFO_PREFIX.len(), 
        hex::encode(HKDF_INFO_PREFIX)
    );
    info!("[KDF]     - Client Public ({} bytes): {}", 
        ED25519_PUBLIC_KEY_SIZE, 
        hex::encode(client_public)
    );
    info!("[KDF]     - Server Public ({} bytes): {}", 
        ED25519_PUBLIC_KEY_SIZE, 
        hex::encode(server_public)
    );

    // Perform HKDF-SHA256
    info!("[KDF] 🔄 Performing HKDF-SHA256 expansion...");
    let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT), shared_secret);
    
    let mut key_bytes = [0u8; CHACHA20_KEY_SIZE];
    hk.expand(&info, &mut key_bytes)
        .map_err(|e| {
            info!("[KDF] ❌ HKDF expansion failed: {:?}", e);
            CoreError::KeyDerivation {
                reason: "HKDF expansion failed".into(),
            }
        })?;

    info!("[KDF] ✅ Session key derived successfully!");
    info!("[KDF] 📤 Output:");
    info!("[KDF]   Session Key (hex): {}", hex::encode(&key_bytes));
    info!("[KDF]   Session Key (base64): {}", base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD, 
        &key_bytes
    ));
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Clear sensitive intermediate data
    info.zeroize();

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
    let hk = Hkdf::<Sha256>::new(Some(salt), shared_secret);
    
    let mut output = vec![0u8; output_len];
    hk.expand(info, &mut output)
        .map_err(|_| CoreError::KeyDerivation {
            reason: format!("HKDF expansion failed for {} bytes", output_len),
        })?;

    Ok(output)
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
