// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_crypto.rs
// ============================================
//! # Storage Encryption — Record + RawLog Crypto Functions
//!
//! ## Creation Reason
//! Extracted from storage.rs to reduce file size and isolate cryptographic
//! logic. All encryption/decryption functions for record content and rawlog
//! content live here.
//!
//! ## Main Functionality
//! - `derive_record_key()` — HKDF from Ed25519 PRIVATE key, salt="memchain-records"
//! - `derive_rawlog_key()` — HKDF from Ed25519 PRIVATE key, salt="memchain-rawlog"
//! - `encrypt_record_content()` — Deterministic ChaCha20-Poly1305 (HMAC nonce)
//! - `decrypt_record_content()` — Corresponding decryption
//! - `encrypt_rawlog_content()` — Random-nonce ChaCha20-Poly1305
//! - `decrypt_rawlog_content()` — Corresponding decryption
//!
//! ## Dependencies
//! - Called by storage.rs (insert, row_to_record, has_active_content)
//! - Called by log_handler.rs (derive_rawlog_key for /log encryption)
//! - Called by reflection.rs (derive_rawlog_key for Miner rawlog decryption)
//!
//! ## Key Derivation Map
//! ```text
//! Ed25519 PRIVATE key (identity.to_bytes())
//! ├── derive_record_key() → salt="memchain-records" → deterministic encryption
//! └── derive_rawlog_key() → salt="memchain-rawlog"  → random-nonce encryption
//! ```
//!
//! ⚠️ Important Note for Next Developer:
//! - Both derive functions MUST use PRIVATE key, NOT public key
//! - derive_rawlog_key was fixed in v2.1.0+MVF+Encryption (was using public key)
//! - Record encryption is DETERMINISTIC (same plaintext → same ciphertext)
//!   This is required for content dedup (has_active_content) to work
//! - RawLog encryption uses RANDOM nonce (not deterministic)
//! - Format for both: nonce(12) || ciphertext(len + 16 tag)
//! - Minimum encrypted size = 28 bytes (12 nonce + 16 tag + 0 content)
//!
//! ## Last Modified
//! v2.1.0+MVF+Encryption - Extracted from storage.rs
//! v2.2.0 - 🌟 Split into dedicated file for maintainability

use sha2::Sha256;
use hkdf::Hkdf;
use tracing::warn;

// ============================================
// Key Derivation
// ============================================

/// Derive a record content encryption key from the owner's Ed25519 **private** key.
///
/// Uses HKDF-SHA256 with salt="memchain-records", info="v1".
/// Output: 32-byte key for deterministic ChaCha20-Poly1305 encryption.
///
/// ## Security
/// - Input MUST be the Ed25519 private key (32 bytes), NOT the public key.
/// - Anyone with only the public key or the database file cannot decrypt.
/// - The key is deterministic: same private key always produces the same record key.
pub fn derive_record_key(owner_private: &[u8; 32]) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(Some(b"memchain-records"), owner_private);
    let mut key = [0u8; 32];
    hk.expand(b"v1", &mut key).expect("HKDF expand should not fail for 32 bytes");
    key
}

/// Derive a rawlog encryption key from the owner's Ed25519 **private** key.
///
/// Uses HKDF-SHA256 with salt="memchain-rawlog", info="v1".
/// Output: 32-byte key for random-nonce ChaCha20-Poly1305 encryption.
///
/// ## Security
/// - Input MUST be the Ed25519 private key (32 bytes), NOT the public key.
/// - Fixed in v2.1.0+MVF+Encryption: previously used public key (insecure).
pub fn derive_rawlog_key(owner_private: &[u8; 32]) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(Some(b"memchain-rawlog"), owner_private);
    let mut key = [0u8; 32];
    hk.expand(b"v1", &mut key).expect("HKDF expand should not fail for 32 bytes");
    key
}

// ============================================
// Record Content Encryption (Deterministic)
// ============================================

/// Deterministic encryption for record content.
///
/// Uses HMAC-SHA256(key, plaintext) truncated to 12 bytes as a deterministic nonce,
/// then encrypts with ChaCha20-Poly1305. This ensures:
/// - Same plaintext + same key → same ciphertext (dedup compatible)
/// - Without the key, content cannot be decrypted
/// - Format: nonce(12) || ciphertext(len + 16 tag)
///
/// ## Why deterministic?
/// Record content dedup (`has_active_content`) and `record_id` hashing both
/// depend on `encrypted_content` being consistent for the same plaintext.
pub fn encrypt_record_content(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, String> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, NewAead};
    use hmac::{Hmac, Mac};

    // Deterministic nonce: HMAC-SHA256(key, plaintext)[0..12]
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|e| format!("HMAC init: {}", e))?;
    mac.update(plaintext);
    let hmac_result = mac.finalize().into_bytes();
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes.copy_from_slice(&hmac_result[..12]);

    let cipher_key = Key::from_slice(key);
    let cipher = ChaCha20Poly1305::new(cipher_key);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext)
        .map_err(|e| format!("ChaCha20 encrypt: {}", e))?;

    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypt record content.
/// Input format: nonce(12) || ciphertext(len + 16 tag)
pub fn decrypt_record_content(key: &[u8; 32], stored: &[u8]) -> Result<Vec<u8>, String> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, NewAead};

    if stored.len() < 12 + 16 {
        return Err("Record ciphertext too short".into());
    }

    let cipher_key = Key::from_slice(key);
    let cipher = ChaCha20Poly1305::new(cipher_key);
    let nonce = Nonce::from_slice(&stored[..12]);
    let ciphertext = &stored[12..];

    cipher.decrypt(nonce, ciphertext)
        .map_err(|e| format!("ChaCha20 decrypt: {}", e))
}

// ============================================
// RawLog Content Encryption (Random Nonce)
// ============================================

/// Encrypt content bytes for rawlog storage.
/// Uses random nonce (NOT deterministic — rawlogs don't need dedup).
/// Format: nonce(12) || ciphertext(len + 16 tag)
pub(crate) fn encrypt_rawlog_content(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, String> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, NewAead};

    let cipher_key = Key::from_slice(key);
    let cipher = ChaCha20Poly1305::new(cipher_key);

    let mut nonce_bytes = [0u8; 12];
    use rand::RngCore;
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext)
        .map_err(|e| format!("ChaCha20 encrypt: {}", e))?;

    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypt content bytes from rawlog storage.
/// Input format: nonce(12) || ciphertext(len + 16 tag)
pub(crate) fn decrypt_rawlog_content(key: &[u8; 32], stored: &[u8]) -> Result<Vec<u8>, String> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, NewAead};

    if stored.len() < 12 + 16 {
        return Err("Ciphertext too short".into());
    }

    let cipher_key = Key::from_slice(key);
    let cipher = ChaCha20Poly1305::new(cipher_key);
    let nonce = Nonce::from_slice(&stored[..12]);
    let ciphertext = &stored[12..];

    cipher.decrypt(nonce, ciphertext)
        .map_err(|e| format!("ChaCha20 decrypt: {}", e))
}

/// Public wrapper for rawlog content decryption (used by Miner).
pub fn decrypt_rawlog_content_pub(key: &[u8; 32], stored: &[u8]) -> Result<Vec<u8>, String> {
    decrypt_rawlog_content(key, stored)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; 32] {
        derive_record_key(&[0x42; 32])
    }

    #[test]
    fn test_record_encrypt_decrypt_roundtrip() {
        let key = test_key();
        let plaintext = b"User is allergic to nuts";
        let ct = encrypt_record_content(&key, plaintext).unwrap();
        assert!(ct.len() > plaintext.len());
        let decrypted = decrypt_record_content(&key, &ct).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_record_encryption_deterministic() {
        let key = test_key();
        let plaintext = b"Same content same ciphertext";
        let ct1 = encrypt_record_content(&key, plaintext).unwrap();
        let ct2 = encrypt_record_content(&key, plaintext).unwrap();
        assert_eq!(ct1, ct2);
    }

    #[test]
    fn test_record_different_plaintext_different_ciphertext() {
        let key = test_key();
        let ct1 = encrypt_record_content(&key, b"content A").unwrap();
        let ct2 = encrypt_record_content(&key, b"content B").unwrap();
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_rawlog_encrypt_decrypt_roundtrip() {
        let key = derive_rawlog_key(&[0x42; 32]);
        let plaintext = b"conversation turn content";
        let ct = encrypt_rawlog_content(&key, plaintext).unwrap();
        let decrypted = decrypt_rawlog_content(&key, &ct).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_rawlog_encryption_not_deterministic() {
        let key = derive_rawlog_key(&[0x42; 32]);
        let plaintext = b"same content";
        let ct1 = encrypt_rawlog_content(&key, plaintext).unwrap();
        let ct2 = encrypt_rawlog_content(&key, plaintext).unwrap();
        // Random nonce → different ciphertext each time
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_derive_record_key_deterministic() {
        let k1 = derive_record_key(&[0x01; 32]);
        let k2 = derive_record_key(&[0x01; 32]);
        assert_eq!(k1, k2);
        let k3 = derive_record_key(&[0x02; 32]);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_record_key_independent_from_rawlog_key() {
        let private_key = [0x42; 32];
        let rk = derive_record_key(&private_key);
        let rlk = derive_rawlog_key(&private_key);
        assert_ne!(rk, rlk);
    }

    #[test]
    fn test_decrypt_too_short() {
        let key = test_key();
        let result = decrypt_record_content(&key, &[0u8; 10]);
        assert!(result.is_err());
    }
}
