// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/sealed_local.rs
// ============================================
//! Shared identity-bound container for source-local workflow persistence.
//!
//! ## Creation Reason
//! Restart snapshots and private attempt journals require identical nonce,
//! AAD, key-derivation, size-bound, and sensitive-key cleanup semantics.
//! Keeping separate implementations would let those security rules diverge.
//!
//! ## Main Functionality
//! - Derives one domain-separated local key from the source identity.
//! - Authenticates magic, version, and a random XChaCha20 nonce as AAD.
//! - Encrypts/decrypts a caller-owned bounded plaintext body.
//! - Returns coarse private errors without logging key or payload material.
//!
//! ## Important Note For The Next Developer
//! - This is local persistence only; never expose it as a protocol codec.
//! - Every caller needs a unique key salt and info domain.
//! - Callers own plaintext cleanup before sealing and after opening.
//! - Do not add identity, work, node, lease, or sequence data to the header.
//!
//! Last Modified: v1.0.0-IdentitySealedLocal - Shared private container.
//! ============================================

use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hkdf::Hkdf;
use rand::{rngs::OsRng, RngCore};
use sha2::Sha256;
use zeroize::Zeroize;

use crate::crypto::keys::IdentityKeyPair;

const HEADER_BYTES: usize = 4 + 2 + 24;
const TAG_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IdentitySealedLocalError {
    TooLarge,
    Malformed,
    UnsupportedVersion,
    AuthenticationFailed,
}

/// Seals one already-encoded private body with an identity-derived key.
///
/// [BLIND-VAULT-IDENTITY-SEALED-LOCAL 2026-08-29 by Codex] The complete
/// header is AAD. Only format identifiers and a random nonce remain visible;
/// all workflow and attempt metadata stays in the encrypted body.
pub(super) fn seal_identity_bound(
    identity: &IdentityKeyPair,
    magic: [u8; 4],
    version: u16,
    key_salt: &[u8],
    key_info: &[u8],
    plaintext: &[u8],
    maximum_container_bytes: usize,
) -> Result<Vec<u8>, IdentitySealedLocalError> {
    if plaintext.len().saturating_add(HEADER_BYTES + TAG_BYTES) > maximum_container_bytes {
        return Err(IdentitySealedLocalError::TooLarge);
    }

    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut nonce);
    let header = identity_sealed_header(magic, version, nonce);
    let mut key = derive_identity_bound_key(identity, key_salt, key_info)?;
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
    let encrypted = cipher.encrypt(
        XNonce::from_slice(&nonce),
        Payload {
            msg: plaintext,
            aad: &header,
        },
    );
    key.zeroize();
    let ciphertext = encrypted.map_err(|_| IdentitySealedLocalError::AuthenticationFailed)?;

    let mut container = header;
    container.extend_from_slice(&ciphertext);
    Ok(container)
}

/// Opens one bounded private body after exact format and AEAD authentication.
pub(super) fn open_identity_bound(
    identity: &IdentityKeyPair,
    container: &[u8],
    magic: [u8; 4],
    version: u16,
    key_salt: &[u8],
    key_info: &[u8],
    maximum_container_bytes: usize,
) -> Result<Vec<u8>, IdentitySealedLocalError> {
    if container.len() > maximum_container_bytes {
        return Err(IdentitySealedLocalError::TooLarge);
    }
    if container.len() < HEADER_BYTES + TAG_BYTES || container[..4] != magic {
        return Err(IdentitySealedLocalError::Malformed);
    }
    if u16::from_be_bytes([container[4], container[5]]) != version {
        return Err(IdentitySealedLocalError::UnsupportedVersion);
    }

    let mut nonce = [0u8; 24];
    nonce.copy_from_slice(&container[6..HEADER_BYTES]);
    let header = &container[..HEADER_BYTES];
    let ciphertext = &container[HEADER_BYTES..];
    let mut key = derive_identity_bound_key(identity, key_salt, key_info)?;
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
    let decrypted = cipher.decrypt(
        XNonce::from_slice(&nonce),
        Payload {
            msg: ciphertext,
            aad: header,
        },
    );
    key.zeroize();
    decrypted.map_err(|_| IdentitySealedLocalError::AuthenticationFailed)
}

fn derive_identity_bound_key(
    identity: &IdentityKeyPair,
    key_salt: &[u8],
    key_info: &[u8],
) -> Result<[u8; 32], IdentitySealedLocalError> {
    let mut identity_secret = identity.to_bytes();
    let hkdf = Hkdf::<Sha256>::new(Some(key_salt), &identity_secret);
    identity_secret.zeroize();

    let mut key = [0u8; 32];
    let mut info = Vec::with_capacity(key_info.len() + 32);
    info.extend_from_slice(key_info);
    info.extend_from_slice(&identity.public_key_bytes());
    if hkdf.expand(&info, &mut key).is_err() {
        key.zeroize();
        info.zeroize();
        return Err(IdentitySealedLocalError::AuthenticationFailed);
    }
    info.zeroize();
    Ok(key)
}

fn identity_sealed_header(magic: [u8; 4], version: u16, nonce: [u8; 24]) -> Vec<u8> {
    let mut header = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(&magic);
    header.extend_from_slice(&version.to_be_bytes());
    header.extend_from_slice(&nonce);
    header
}
