// ============================================
// File: crates/aeronyx-core/src/protocol/anonymous_mailbox_recipient_seal.rs
// ============================================
//! Canonical recipient-sealed content for anonymous mailbox items.
//!
//! Nodes see only this bounded opaque byte string. Sender/receiver identities
//! remain inside the authenticated encryption boundary. Platform key storage is
//! deliberately outside this module and is represented by a narrow native key
//! handle trait.
//!
//! [ANONYMOUS-MAILBOX-RECIPIENT-SEAL 2026-09-08 by Codex] New effects require
//! a current AMDI v2 admission type; the historical context below can only open
//! already accepted ciphertext after restart or invitation expiry.

use std::fmt;

use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hkdf::Hkdf;
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};
use thiserror::Error;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::{Zeroize, Zeroizing};

use crate::crypto::IdentityPublicKey;
use crate::chat::{
    decode_envelope_strict_verified, encode_envelope, ChatEnvelope, MAX_CHAT_ENVELOPE_BYTES,
};

const RECIPIENT_SEAL_MAGIC: [u8; 4] = *b"AMSI";
const RECIPIENT_SEAL_KEY_SALT_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-RecipientSeal-Salt-v1";
const RECIPIENT_SEAL_KEY_INFO_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-RecipientSeal-Key-v1";
const RECIPIENT_SEAL_AAD_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-RecipientSeal-AAD-v1";
const RECIPIENT_SEAL_PREFIX_BYTES: usize = 4 + 2 + 4 + 1 + 16 + 32 + 24 + 1 + 4;
const RECIPIENT_SEAL_AEAD_TAG_BYTES: usize = 16;
const RECIPIENT_SEAL_ACTUAL_LENGTH_BYTES: usize = 4;

/// Frozen recipient-seal codec version.
pub const ANONYMOUS_MAILBOX_RECIPIENT_SEAL_VERSION_V1: u16 = 1;
/// Frozen X25519/HKDF-SHA256/XChaCha20-Poly1305 algorithm identifier.
pub const ANONYMOUS_MAILBOX_RECIPIENT_SEAL_ALGORITHM_V1: u8 = 1;
/// Largest canonical recipient-sealed item (108-byte framing plus 128 KiB slot).
#[allow(clippy::cast_possible_truncation)] // Frozen 128 KiB fits every supported usize.
pub const MAX_ANONYMOUS_MAILBOX_RECIPIENT_SEALED_ITEM_BYTES: usize = RECIPIENT_SEAL_PREFIX_BYTES
    + RECIPIENT_SEAL_ACTUAL_LENGTH_BYTES
    + MAX_CHAT_ENVELOPE_BYTES as usize
    + RECIPIENT_SEAL_AEAD_TAG_BYTES;

/// Coarse, privacy-safe recipient-seal failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AnonymousMailboxRecipientSealError {
    /// An input exceeds a frozen size class.
    #[error("anonymous mailbox recipient seal exceeds its limit")]
    TooLarge,
    /// A frame or authenticated plaintext is malformed or non-canonical.
    #[error("anonymous mailbox recipient seal is malformed")]
    Malformed,
    /// The codec version or crypto algorithm is unsupported.
    #[error("anonymous mailbox recipient seal version is unsupported")]
    UnsupportedVersion,
    /// The invitation, key handle, target, or receiver binding differs.
    #[error("anonymous mailbox recipient seal claims do not match")]
    ClaimsConflict,
    /// The native recipient key is absent or unusable.
    #[error("anonymous mailbox recipient key is unavailable")]
    KeyUnavailable,
    /// Authentication or sender signature verification failed.
    #[error("anonymous mailbox recipient seal was rejected")]
    Rejected,
}

/// Public recipient content-encryption capability carried by AMDI v2.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AnonymousMailboxRecipientSealPublicV1 {
    algorithm: u8,
    key_id: [u8; 16],
    public_key: [u8; 32],
}

impl AnonymousMailboxRecipientSealPublicV1 {
    /// Constructs one validated public capability.
    ///
    /// # Errors
    /// Returns a coarse seal error for an empty key id or invalid X25519 key.
    pub fn new(
        key_id: [u8; 16],
        public_key: [u8; 32],
    ) -> Result<Self, AnonymousMailboxRecipientSealError> {
        if key_id.iter().all(|byte| *byte == 0) || !x25519_public_key_is_valid(&public_key) {
            return Err(AnonymousMailboxRecipientSealError::Malformed);
        }
        Ok(Self {
            algorithm: ANONYMOUS_MAILBOX_RECIPIENT_SEAL_ALGORITHM_V1,
            key_id,
            public_key,
        })
    }

    /// Frozen crypto algorithm identifier.
    #[must_use]
    pub const fn algorithm(&self) -> u8 {
        self.algorithm
    }

    /// Opaque native key-handle identifier.
    #[must_use]
    pub const fn key_id(&self) -> [u8; 16] {
        self.key_id
    }

    /// Receiver X25519 public key.
    #[must_use]
    pub const fn public_key(&self) -> [u8; 32] {
        self.public_key
    }
}

impl fmt::Debug for AnonymousMailboxRecipientSealPublicV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxRecipientSealPublicV1")
            .field("algorithm", &self.algorithm)
            .finish_non_exhaustive()
    }
}

/// Native-only X25519 key operation used to reopen retained AMSI bytes.
///
/// Implementations own durable key persistence and must never expose the raw
/// private key through FFI, serde, logs, or this trait.
pub trait AnonymousMailboxRecipientSealKeyHandleV1: Send + Sync {
    /// Opaque key id bound by the signed invitation.
    fn key_id(&self) -> [u8; 16];
    /// Public key derived from the native private key.
    fn public_key(&self) -> [u8; 32];
    /// Derives one zeroizing shared secret or fails closed if the handle is gone.
    ///
    /// # Errors
    /// Returns a coarse seal error when the handle is unavailable or rejects
    /// the supplied peer key.
    fn derive_shared_secret(
        &self,
        peer_public_key: [u8; 32],
    ) -> Result<Zeroizing<[u8; 32]>, AnonymousMailboxRecipientSealError>;
}

/// Canonical opaque AMSI bytes created by an active AMDI v2 context.
pub struct AnonymousMailboxRecipientSealedItemV1 {
    bytes: Vec<u8>,
    origin_binding_commitment: [u8; 32],
}

impl AnonymousMailboxRecipientSealedItemV1 {
    pub(crate) const fn new(
        bytes: Vec<u8>,
        binding: &AnonymousMailboxRecipientSealBindingV1,
    ) -> Self {
        Self {
            bytes,
            origin_binding_commitment: binding.origin_commitment(),
        }
    }

    pub(crate) fn matches_binding(&self, binding: &AnonymousMailboxRecipientSealBindingV1) -> bool {
        self.origin_binding_commitment == binding.origin_commitment()
    }

    /// Exact bytes to persist for an exact Put retry.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the typed item and returns its canonical opaque bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl fmt::Debug for AnonymousMailboxRecipientSealedItemV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxRecipientSealedItemV1")
            .field("encoded_bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

/// Immutable authenticated context shared by sender sealing and recipient open.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct AnonymousMailboxRecipientSealBindingV1 {
    invitation_commitment: [u8; 32],
    target_node_id: [u8; 32],
    descriptor_sequence: u64,
    descriptor_commitment: [u8; 32],
    mailbox_id: [u8; 32],
    lease_claims_commitment: [u8; 32],
    receiver: [u8; 32],
    recipient: AnonymousMailboxRecipientSealPublicV1,
}

impl AnonymousMailboxRecipientSealBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        invitation_commitment: [u8; 32],
        target_node_id: [u8; 32],
        descriptor_sequence: u64,
        descriptor_commitment: [u8; 32],
        mailbox_id: [u8; 32],
        lease_claims_commitment: [u8; 32],
        receiver: [u8; 32],
        recipient: AnonymousMailboxRecipientSealPublicV1,
    ) -> Result<Self, AnonymousMailboxRecipientSealError> {
        if invitation_commitment.iter().all(|byte| *byte == 0)
            || descriptor_sequence == 0
            || descriptor_commitment.iter().all(|byte| *byte == 0)
            || mailbox_id.iter().all(|byte| *byte == 0)
            || lease_claims_commitment.iter().all(|byte| *byte == 0)
            || recipient.algorithm != ANONYMOUS_MAILBOX_RECIPIENT_SEAL_ALGORITHM_V1
        {
            return Err(AnonymousMailboxRecipientSealError::Malformed);
        }
        IdentityPublicKey::from_bytes(&target_node_id)
            .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?;
        IdentityPublicKey::from_bytes(&receiver)
            .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?;
        Ok(Self {
            invitation_commitment,
            target_node_id,
            descriptor_sequence,
            descriptor_commitment,
            mailbox_id,
            lease_claims_commitment,
            receiver,
            recipient,
        })
    }

    // [ANONYMOUS-MAILBOX-RECIPIENT-SEAL 2026-09-08 by Codex] Reuse the exact
    // canonical signed-invitation commitment as the private origin tag. It
    // already binds every field represented here, avoiding a second transcript
    // that could drift. The tag is never serialized.
    const fn origin_commitment(&self) -> [u8; 32] {
        self.invitation_commitment
    }
}

/// Historical recipient-only context returned after AMDI v2 authentication.
///
/// This type deliberately has no sealing or Put API and contains no deposit
/// seed. Its only authority is opening an already accepted AMSI item.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AnonymousMailboxRecipientOpenContextV1 {
    binding: AnonymousMailboxRecipientSealBindingV1,
    lease_expires_at: u64,
}

impl AnonymousMailboxRecipientOpenContextV1 {
    pub(crate) const fn new(
        binding: AnonymousMailboxRecipientSealBindingV1,
        lease_expires_at: u64,
    ) -> Self {
        Self {
            binding,
            lease_expires_at,
        }
    }

    /// Minimum native key-retention boundary. Native storage must additionally
    /// keep the handle while any locally retained sealed item is pending.
    #[must_use]
    pub const fn lease_expires_at(&self) -> u64 {
        self.lease_expires_at
    }

    /// Opens and validates one exact recipient-sealed `ChatEnvelope`.
    ///
    /// # Errors
    /// Returns a coarse seal error for unavailable keys, malformed or
    /// non-canonical frames, binding mismatch, or failed authentication.
    pub fn open_chat_envelope(
        &self,
        key_handle: &dyn AnonymousMailboxRecipientSealKeyHandleV1,
        sealed: &[u8],
    ) -> Result<ChatEnvelope, AnonymousMailboxRecipientSealError> {
        open_chat_envelope(&self.binding, key_handle, sealed)
    }
}

impl fmt::Debug for AnonymousMailboxRecipientOpenContextV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxRecipientOpenContextV1")
            .field("lease_expires_at", &self.lease_expires_at)
            .finish_non_exhaustive()
    }
}

pub(crate) fn seal_chat_envelope(
    binding: &AnonymousMailboxRecipientSealBindingV1,
    envelope: &ChatEnvelope,
) -> Result<Vec<u8>, AnonymousMailboxRecipientSealError> {
    let canonical =
        encode_envelope(envelope).map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?;
    decode_envelope_strict_verified(&canonical)
        .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?;
    if envelope.receiver != binding.receiver {
        return Err(AnonymousMailboxRecipientSealError::ClaimsConflict);
    }
    let slot = padding_slot(canonical.len()).ok_or(AnonymousMailboxRecipientSealError::TooLarge)?;
    let mut plaintext = Zeroizing::new(vec![0u8; RECIPIENT_SEAL_ACTUAL_LENGTH_BYTES + slot]);
    let actual_len =
        u32::try_from(canonical.len()).map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?;
    plaintext[..4].copy_from_slice(&actual_len.to_le_bytes());
    plaintext[4..4 + canonical.len()].copy_from_slice(&canonical);

    let mut ephemeral_secret_bytes = [0u8; 32];
    OsRng.fill_bytes(&mut ephemeral_secret_bytes);
    let ephemeral_secret = StaticSecret::from(ephemeral_secret_bytes);
    ephemeral_secret_bytes.zeroize();
    let ephemeral_public = X25519PublicKey::from(&ephemeral_secret).to_bytes();
    let shared = Zeroizing::new(
        *ephemeral_secret
            .diffie_hellman(&X25519PublicKey::from(binding.recipient.public_key))
            .as_bytes(),
    );
    if shared.iter().all(|byte| *byte == 0) {
        return Err(AnonymousMailboxRecipientSealError::Rejected);
    }
    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut nonce);
    seal_with_material(
        binding,
        &ephemeral_public,
        &nonce,
        slot,
        &shared,
        &plaintext,
    )
}

fn seal_with_material(
    binding: &AnonymousMailboxRecipientSealBindingV1,
    ephemeral_public: &[u8; 32],
    nonce: &[u8; 24],
    slot: usize,
    shared: &[u8; 32],
    plaintext: &[u8],
) -> Result<Vec<u8>, AnonymousMailboxRecipientSealError> {
    let ciphertext_len = plaintext
        .len()
        .checked_add(RECIPIENT_SEAL_AEAD_TAG_BYTES)
        .ok_or(AnonymousMailboxRecipientSealError::TooLarge)?;
    let ciphertext_len_u32 =
        u32::try_from(ciphertext_len).map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?;
    let class = padding_class(slot).ok_or(AnonymousMailboxRecipientSealError::TooLarge)?;
    let mut key = derive_key(binding, ephemeral_public, shared)?;
    let aad = aad(
        binding,
        ephemeral_public,
        nonce,
        class,
        slot,
        ciphertext_len_u32,
    )?;
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
    key.zeroize();
    let ciphertext = cipher
        .encrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad: &aad,
            },
        )
        .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?;
    let total_len = RECIPIENT_SEAL_PREFIX_BYTES
        .checked_add(ciphertext.len())
        .ok_or(AnonymousMailboxRecipientSealError::TooLarge)?;
    if total_len > MAX_ANONYMOUS_MAILBOX_RECIPIENT_SEALED_ITEM_BYTES {
        return Err(AnonymousMailboxRecipientSealError::TooLarge);
    }
    let mut encoded = Vec::with_capacity(total_len);
    encoded.extend_from_slice(&RECIPIENT_SEAL_MAGIC);
    encoded.extend_from_slice(&ANONYMOUS_MAILBOX_RECIPIENT_SEAL_VERSION_V1.to_le_bytes());
    encoded.extend_from_slice(
        &u32::try_from(total_len)
            .map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?
            .to_le_bytes(),
    );
    encoded.push(binding.recipient.algorithm);
    encoded.extend_from_slice(&binding.recipient.key_id);
    encoded.extend_from_slice(ephemeral_public);
    encoded.extend_from_slice(nonce);
    encoded.push(class);
    encoded.extend_from_slice(&ciphertext_len_u32.to_le_bytes());
    encoded.extend_from_slice(&ciphertext);
    Ok(encoded)
}

fn open_chat_envelope(
    binding: &AnonymousMailboxRecipientSealBindingV1,
    key_handle: &dyn AnonymousMailboxRecipientSealKeyHandleV1,
    encoded: &[u8],
) -> Result<ChatEnvelope, AnonymousMailboxRecipientSealError> {
    let parsed = parse(encoded)?;
    if parsed.algorithm != binding.recipient.algorithm
        || parsed.key_id != binding.recipient.key_id
        || key_handle.key_id() != binding.recipient.key_id
        || key_handle.public_key() != binding.recipient.public_key
    {
        return Err(AnonymousMailboxRecipientSealError::ClaimsConflict);
    }
    if !x25519_public_key_is_valid(&parsed.ephemeral_public) {
        return Err(AnonymousMailboxRecipientSealError::Rejected);
    }
    let shared = key_handle.derive_shared_secret(parsed.ephemeral_public)?;
    if shared.iter().all(|byte| *byte == 0) {
        return Err(AnonymousMailboxRecipientSealError::Rejected);
    }
    let slot = slot_for_class(parsed.padding_class)
        .ok_or(AnonymousMailboxRecipientSealError::Malformed)?;
    let expected_ciphertext = RECIPIENT_SEAL_ACTUAL_LENGTH_BYTES
        .checked_add(slot)
        .and_then(|value| value.checked_add(RECIPIENT_SEAL_AEAD_TAG_BYTES))
        .ok_or(AnonymousMailboxRecipientSealError::TooLarge)?;
    if parsed.ciphertext.len() != expected_ciphertext {
        return Err(AnonymousMailboxRecipientSealError::Malformed);
    }
    let mut key = derive_key(binding, &parsed.ephemeral_public, &shared)?;
    let aad = aad(
        binding,
        &parsed.ephemeral_public,
        &parsed.nonce,
        parsed.padding_class,
        slot,
        u32::try_from(parsed.ciphertext.len())
            .map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?,
    )?;
    let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
    key.zeroize();
    let plaintext = Zeroizing::new(
        cipher
            .decrypt(
                XNonce::from_slice(&parsed.nonce),
                Payload {
                    msg: parsed.ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?,
    );
    if plaintext.len() != RECIPIENT_SEAL_ACTUAL_LENGTH_BYTES + slot {
        return Err(AnonymousMailboxRecipientSealError::Malformed);
    }
    let actual = usize::try_from(u32::from_le_bytes(
        plaintext[..4]
            .try_into()
            .map_err(|_| AnonymousMailboxRecipientSealError::Malformed)?,
    ))
    .map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?;
    if actual == 0 || actual > slot || plaintext[4 + actual..].iter().any(|byte| *byte != 0) {
        return Err(AnonymousMailboxRecipientSealError::Malformed);
    }
    let envelope = decode_envelope_strict_verified(&plaintext[4..4 + actual])
        .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?;
    if envelope.receiver != binding.receiver {
        return Err(AnonymousMailboxRecipientSealError::ClaimsConflict);
    }
    Ok(envelope)
}

struct ParsedSeal<'a> {
    algorithm: u8,
    key_id: [u8; 16],
    ephemeral_public: [u8; 32],
    nonce: [u8; 24],
    padding_class: u8,
    ciphertext: &'a [u8],
}

fn parse(encoded: &[u8]) -> Result<ParsedSeal<'_>, AnonymousMailboxRecipientSealError> {
    if encoded.len() > MAX_ANONYMOUS_MAILBOX_RECIPIENT_SEALED_ITEM_BYTES {
        return Err(AnonymousMailboxRecipientSealError::TooLarge);
    }
    if encoded.len() < RECIPIENT_SEAL_PREFIX_BYTES + RECIPIENT_SEAL_AEAD_TAG_BYTES {
        return Err(AnonymousMailboxRecipientSealError::Malformed);
    }
    if encoded[..4] != RECIPIENT_SEAL_MAGIC {
        return Err(AnonymousMailboxRecipientSealError::Malformed);
    }
    let mut offset = 4;
    let version = take_u16(encoded, &mut offset)?;
    if version != ANONYMOUS_MAILBOX_RECIPIENT_SEAL_VERSION_V1 {
        return Err(AnonymousMailboxRecipientSealError::UnsupportedVersion);
    }
    let declared = usize::try_from(take_u32(encoded, &mut offset)?)
        .map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?;
    if declared != encoded.len() {
        return Err(AnonymousMailboxRecipientSealError::Malformed);
    }
    let algorithm = take::<1>(encoded, &mut offset)?[0];
    if algorithm != ANONYMOUS_MAILBOX_RECIPIENT_SEAL_ALGORITHM_V1 {
        return Err(AnonymousMailboxRecipientSealError::UnsupportedVersion);
    }
    let key_id = take::<16>(encoded, &mut offset)?;
    let ephemeral_public = take::<32>(encoded, &mut offset)?;
    let nonce = take::<24>(encoded, &mut offset)?;
    let padding_class = take::<1>(encoded, &mut offset)?[0];
    let ciphertext_len = usize::try_from(take_u32(encoded, &mut offset)?)
        .map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?;
    let end = offset
        .checked_add(ciphertext_len)
        .ok_or(AnonymousMailboxRecipientSealError::TooLarge)?;
    let ciphertext = encoded
        .get(offset..end)
        .ok_or(AnonymousMailboxRecipientSealError::Malformed)?;
    if end != encoded.len() {
        return Err(AnonymousMailboxRecipientSealError::Malformed);
    }
    Ok(ParsedSeal {
        algorithm,
        key_id,
        ephemeral_public,
        nonce,
        padding_class,
        ciphertext,
    })
}

fn derive_key(
    binding: &AnonymousMailboxRecipientSealBindingV1,
    ephemeral_public: &[u8; 32],
    shared: &[u8; 32],
) -> Result<[u8; 32], AnonymousMailboxRecipientSealError> {
    let mut salt_hasher = Sha256::new();
    salt_hasher.update(RECIPIENT_SEAL_KEY_SALT_DOMAIN);
    salt_hasher.update(binding.invitation_commitment);
    salt_hasher.update(binding.mailbox_id);
    salt_hasher.update(binding.lease_claims_commitment);
    let salt = salt_hasher.finalize();
    let mut info = Vec::with_capacity(RECIPIENT_SEAL_KEY_INFO_DOMAIN.len() + 2 + 1 + 16 + 32 + 32);
    info.extend_from_slice(RECIPIENT_SEAL_KEY_INFO_DOMAIN);
    info.extend_from_slice(&ANONYMOUS_MAILBOX_RECIPIENT_SEAL_VERSION_V1.to_le_bytes());
    info.push(binding.recipient.algorithm);
    info.extend_from_slice(&binding.recipient.key_id);
    info.extend_from_slice(&binding.recipient.public_key);
    info.extend_from_slice(ephemeral_public);
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), shared);
    let mut key = [0u8; 32];
    hkdf.expand(&info, &mut key)
        .map_err(|_| AnonymousMailboxRecipientSealError::Malformed)?;
    info.zeroize();
    Ok(key)
}

fn aad(
    binding: &AnonymousMailboxRecipientSealBindingV1,
    ephemeral_public: &[u8; 32],
    nonce: &[u8; 24],
    padding_class: u8,
    slot: usize,
    ciphertext_len: u32,
) -> Result<Vec<u8>, AnonymousMailboxRecipientSealError> {
    let slot = u32::try_from(slot).map_err(|_| AnonymousMailboxRecipientSealError::TooLarge)?;
    let mut aad =
        Vec::with_capacity(RECIPIENT_SEAL_AAD_DOMAIN.len() + 2 + 32 * 6 + 8 + 16 + 24 + 10);
    aad.extend_from_slice(RECIPIENT_SEAL_AAD_DOMAIN);
    aad.extend_from_slice(&ANONYMOUS_MAILBOX_RECIPIENT_SEAL_VERSION_V1.to_le_bytes());
    aad.extend_from_slice(&binding.invitation_commitment);
    aad.extend_from_slice(&binding.target_node_id);
    aad.extend_from_slice(&binding.descriptor_sequence.to_le_bytes());
    aad.extend_from_slice(&binding.descriptor_commitment);
    aad.extend_from_slice(&binding.mailbox_id);
    aad.extend_from_slice(&binding.lease_claims_commitment);
    aad.extend_from_slice(&binding.receiver);
    aad.push(binding.recipient.algorithm);
    aad.extend_from_slice(&binding.recipient.key_id);
    aad.extend_from_slice(&binding.recipient.public_key);
    aad.extend_from_slice(ephemeral_public);
    aad.extend_from_slice(nonce);
    aad.push(padding_class);
    aad.extend_from_slice(&slot.to_le_bytes());
    aad.extend_from_slice(&ciphertext_len.to_le_bytes());
    Ok(aad)
}

fn padding_slot(length: usize) -> Option<usize> {
    [4 * 1024, 16 * 1024, 64 * 1024, 128 * 1024]
        .into_iter()
        .find(|slot| length <= *slot)
}

const fn padding_class(slot: usize) -> Option<u8> {
    match slot {
        4_096 => Some(1),
        16_384 => Some(2),
        65_536 => Some(3),
        131_072 => Some(4),
        _ => None,
    }
}

const fn slot_for_class(class: u8) -> Option<usize> {
    match class {
        1 => Some(4_096),
        2 => Some(16_384),
        3 => Some(65_536),
        4 => Some(131_072),
        _ => None,
    }
}

fn x25519_public_key_is_valid(public_key: &[u8; 32]) -> bool {
    if public_key.iter().all(|byte| *byte == 0) {
        return false;
    }
    let probe = StaticSecret::from([0x6d; 32]);
    let shared = probe.diffie_hellman(&X25519PublicKey::from(*public_key));
    !shared.as_bytes().iter().all(|byte| *byte == 0)
}

fn take<const N: usize>(
    encoded: &[u8],
    offset: &mut usize,
) -> Result<[u8; N], AnonymousMailboxRecipientSealError> {
    let end = offset
        .checked_add(N)
        .ok_or(AnonymousMailboxRecipientSealError::TooLarge)?;
    let value = encoded
        .get(*offset..end)
        .ok_or(AnonymousMailboxRecipientSealError::Malformed)?
        .try_into()
        .map_err(|_| AnonymousMailboxRecipientSealError::Malformed)?;
    *offset = end;
    Ok(value)
}

fn take_u16(encoded: &[u8], offset: &mut usize) -> Result<u16, AnonymousMailboxRecipientSealError> {
    Ok(u16::from_le_bytes(take(encoded, offset)?))
}

fn take_u32(encoded: &[u8], offset: &mut usize) -> Result<u32, AnonymousMailboxRecipientSealError> {
    Ok(u32::from_le_bytes(take(encoded, offset)?))
}

