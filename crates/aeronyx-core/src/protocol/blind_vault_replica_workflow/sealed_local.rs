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
//! - Exposes only a purpose-fixed source-reservation seal/open facade.
//!
//! ## Important Note For The Next Developer
//! - This is local persistence only; never expose it as a protocol codec.
//! - Every caller needs a unique key salt and info domain.
//! - Callers own plaintext cleanup before sealing and after opening.
//! - Do not add identity, work, node, lease, or sequence data to the header.
//!
//! Last Modified: v1.2.0-PullRestartSeal - Private purpose-fixed AXBP facade.
//! [BLIND-VAULT-PULL-RESTART 2026-10-04 by Codex] Reuses identity AEAD only.
//! v1.1.0-SourceReservationSeal - Narrow local persistence API.
//! [SOURCE-RESERVATION-SEAL 2026-10-04 by Codex] Fixed reservation domain
//! and bounds; no workflow readiness, persistence receipt, or dispatch authority.
//! v1.0.0-IdentitySealedLocal - Shared private container.
//! ============================================

use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hkdf::Hkdf;
use rand::{rngs::OsRng, RngCore};
use sha2::Sha256;
use thiserror::Error;
use zeroize::Zeroize;

use crate::crypto::keys::IdentityKeyPair;

const HEADER_BYTES: usize = 4 + 2 + 24;
const TAG_BYTES: usize = 16;

// [BLIND-VAULT-PULL-RESTART 2026-10-04 by Codex] Independent local-only
// domain. Neither the facade nor its plaintext is public outside this crate.
const PULL_RESTART_MAGIC: [u8; 4] = *b"AXBP";
const PULL_RESTART_VERSION: u16 = 1;
const PULL_RESTART_KEY_SALT: &[u8] = b"AeroNyx-BlindVault-OnionPull-Restart-Key-v1";
const PULL_RESTART_KEY_INFO: &[u8] = b"AeroNyx-BlindVault-OnionPull-Restart-State-v1";
pub(crate) const MAX_PULL_RESTART_SEALED_BYTES: usize = 512;
pub(crate) const MAX_PULL_RESTART_BODY_BYTES: usize =
    MAX_PULL_RESTART_SEALED_BYTES - HEADER_BYTES - TAG_BYTES;

pub(crate) fn seal_pull_restart(
    identity: &IdentityKeyPair,
    body: &[u8],
) -> Result<Vec<u8>, super::super::blind_vault::BlindVaultOnionPullRestartError> {
    seal_identity_bound(
        identity,
        PULL_RESTART_MAGIC,
        PULL_RESTART_VERSION,
        PULL_RESTART_KEY_SALT,
        PULL_RESTART_KEY_INFO,
        body,
        MAX_PULL_RESTART_SEALED_BYTES,
    )
    .map_err(pull_restart_error)
}

pub(crate) fn open_pull_restart(
    identity: &IdentityKeyPair,
    sealed: &[u8],
) -> Result<zeroize::Zeroizing<Vec<u8>>, super::super::blind_vault::BlindVaultOnionPullRestartError> {
    open_identity_bound(
        identity,
        sealed,
        PULL_RESTART_MAGIC,
        PULL_RESTART_VERSION,
        PULL_RESTART_KEY_SALT,
        PULL_RESTART_KEY_INFO,
        MAX_PULL_RESTART_SEALED_BYTES,
    )
    .map(zeroize::Zeroizing::new)
    .map_err(pull_restart_error)
}

fn pull_restart_error(
    error: IdentitySealedLocalError,
) -> super::super::blind_vault::BlindVaultOnionPullRestartError {
    use super::super::blind_vault::BlindVaultOnionPullRestartError as Error;
    match error {
        IdentitySealedLocalError::TooLarge => Error::TooLarge,
        IdentitySealedLocalError::Malformed => Error::Malformed,
        IdentitySealedLocalError::UnsupportedVersion => Error::UnsupportedVersion,
        IdentitySealedLocalError::AuthenticationFailed => Error::AuthenticationFailed,
    }
}

// [SOURCE-RESERVATION-SEAL 2026-10-04 by Codex] Source-local AXSR v1 is
// deliberately distinct from AXRJ journals, AXRS snapshots and AXBC bodies.
// Never let a public caller override these identifiers or key derivation domains.
const SOURCE_RESERVATION_MAGIC: [u8; 4] = *b"AXSR";
const SOURCE_RESERVATION_VERSION: u16 = 1;
const SOURCE_RESERVATION_KEY_SALT: &[u8] = b"AeroNyx-BlindVault-Source-Reservation-Key-v1";
const SOURCE_RESERVATION_KEY_INFO: &[u8] = b"AeroNyx-BlindVault-Source-Reservation-State-v1";

/// Fixed maximum private source-reservation body: 32 KiB, including metadata.
pub const MAX_BLIND_VAULT_SOURCE_RESERVATION_BODY_BYTES: usize = 32 * 1024;

/// Fixed local container maximum: 32 KiB + 30-byte header + 16-byte AEAD tag.
pub const MAX_BLIND_VAULT_SOURCE_RESERVATION_SEALED_BYTES: usize =
    MAX_BLIND_VAULT_SOURCE_RESERVATION_BODY_BYTES + HEADER_BYTES + TAG_BYTES;

/// Coarse local failures containing no identity, metadata, path or crypto detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BlindVaultSourceReservationSealError {
    #[error("source reservation exceeds local size limit")]
    TooLarge,
    #[error("malformed local source reservation")]
    Malformed,
    #[error("unsupported local source reservation version")]
    UnsupportedVersion,
    #[error("local source reservation authentication failed")]
    AuthenticationFailed,
}

/// Seals opaque source-local reservation bytes under the source node identity.
///
/// [SOURCE-RESERVATION-SEAL 2026-10-04 by Codex] AXSR v1 exposes only magic,
/// big-endian version and a random nonce; the complete header is authenticated.
/// Put all owner/work metadata in `private_body`, never a plaintext side header.
/// Callers own canonical body encoding, secret cleanup, durable storage and
/// rollback/replay policy. Re-sealing generates a new nonce; persist and replay
/// the returned bytes if exact container identity is required. Success does not
/// prove a durable reservation, Prepared state, or permission to dispatch.
///
/// # Errors
/// Rejects bodies above 32 KiB or a failure of the identity-bound seal.
pub fn seal_blind_vault_source_reservation(
    identity: &IdentityKeyPair,
    private_body: &[u8],
) -> Result<Vec<u8>, BlindVaultSourceReservationSealError> {
    seal_identity_bound(
        identity,
        SOURCE_RESERVATION_MAGIC,
        SOURCE_RESERVATION_VERSION,
        SOURCE_RESERVATION_KEY_SALT,
        SOURCE_RESERVATION_KEY_INFO,
        private_body,
        MAX_BLIND_VAULT_SOURCE_RESERVATION_SEALED_BYTES,
    )
    .map_err(source_reservation_error)
}

/// Opens only a bounded AXSR v1 reservation authenticated to this identity.
///
/// [SOURCE-RESERVATION-SEAL 2026-10-04 by Codex] Returns the opaque private
/// body only after authentication. Callers must validate its own version, exact
/// claims and restart policy, and zeroize it when finished; this function does
/// not interpret it as a workflow snapshot or authorize a network effect.
///
/// # Errors
/// Rejects oversize, malformed/truncated, wrong-purpose, wrong-version,
/// wrong-identity or tampered containers, including unauthenticated trailing bytes.
pub fn open_blind_vault_source_reservation(
    identity: &IdentityKeyPair,
    sealed: &[u8],
) -> Result<Vec<u8>, BlindVaultSourceReservationSealError> {
    open_identity_bound(
        identity,
        sealed,
        SOURCE_RESERVATION_MAGIC,
        SOURCE_RESERVATION_VERSION,
        SOURCE_RESERVATION_KEY_SALT,
        SOURCE_RESERVATION_KEY_INFO,
        MAX_BLIND_VAULT_SOURCE_RESERVATION_SEALED_BYTES,
    )
    .map_err(source_reservation_error)
}

fn source_reservation_error(error: IdentitySealedLocalError) -> BlindVaultSourceReservationSealError {
    match error {
        IdentitySealedLocalError::TooLarge => BlindVaultSourceReservationSealError::TooLarge,
        IdentitySealedLocalError::Malformed => BlindVaultSourceReservationSealError::Malformed,
        IdentitySealedLocalError::UnsupportedVersion => {
            BlindVaultSourceReservationSealError::UnsupportedVersion
        }
        IdentitySealedLocalError::AuthenticationFailed => {
            BlindVaultSourceReservationSealError::AuthenticationFailed
        }
    }
}

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
