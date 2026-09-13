// ============================================
// File: crates/aeronyx-core/src/protocol/anonymous_mailbox_deposit_invitation.rs
// ============================================
//! Canonical receiver-shared deposit-only anonymous mailbox invitation.
//!
//! This module composes the existing anonymous-mailbox terminal codec. It does
//! not add a terminal operation or a server-visible wire message. The encoded
//! invitation is a bearer capability intended for private client-to-client
//! exchange: it contains the deposit seed, but never the read seed. The read
//! capability signs the exact target pin, lease proof, policy window, and
//! deposit material so none can be substituted independently.
//!
//! [M13J 2026-09-05 by Codex] AMDI v1 is a client compliance boundary, not a
//! server-side revocation primitive. A holder that copied the deposit seed can
//! continue attempting Put until the target enforces lease expiry and quota.
//!
//! [ANONYMOUS-MAILBOX-DEPOSIT-SUMMARY 2026-09-13 by Codex] A successfully
//! admitted V2 capability exposes only its already-verified redacted summary,
//! allowing a client to persist the exact target pin without exposing bearer
//! or recipient private material.

use std::fmt;

use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::{Zeroize, Zeroizing};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxLeaseCreateV1, AnonymousMailboxOperationV1, AnonymousMailboxOutcomeV1,
    AnonymousMailboxProtocolError, AnonymousMailboxPutV1, AnonymousMailboxTerminalFrameV1,
    AnonymousMailboxTerminalResponseV1, MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS,
    MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS,
};
use crate::protocol::anonymous_mailbox_recipient_seal::{
    seal_chat_envelope, AnonymousMailboxRecipientOpenContextV1,
    AnonymousMailboxRecipientSealBindingV1, AnonymousMailboxRecipientSealError,
    AnonymousMailboxRecipientSealPublicV1, AnonymousMailboxRecipientSealedItemV1,
};
use crate::protocol::chat::ChatEnvelope;

const INVITATION_MAGIC: [u8; 4] = *b"AMDI";
const INVITATION_SIGNATURE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-DepositInvitation-v1";
const INVITATION_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-AnonymousMailbox-DepositInvitationCommitment-v1";
const INVITATION_V2_SIGNATURE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-DepositInvitation-v2";
const INVITATION_V2_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-AnonymousMailbox-DepositInvitationCommitment-v2";
const HEADER_BYTES: usize = 4 + 2 + 4;
const FIXED_PREFIX_BYTES: usize = HEADER_BYTES + 16 + 8 + 8 + 4 + 32 + 8 + 32 + 32 + 32;
const FIXED_PREFIX_BYTES_V2: usize = FIXED_PREFIX_BYTES + 32 + 1 + 16 + 32;
const LENGTH_BYTES: usize = 2;
const SIGNATURE_BYTES: usize = 64;
const MAX_LEASE_CREATE_FRAME_BYTES: usize = 512;
const MAX_LEASE_RESPONSE_FRAME_BYTES: usize = 256;

/// Frozen AMDI codec version.
pub const ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V1: u16 = 1;
/// Recipient-sealed AMDI codec version. V1 remains deposit-only.
pub const ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V2: u16 = 2;
/// Hard upper bound for one canonical deposit invitation.
pub const MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES: usize = 1024;
/// Longest client-policy invitation lifetime.
pub const MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_TTL_SECS: u64 = 7 * 24 * 60 * 60;
/// Smallest signed lease runway accepted by AMDI v1.
pub const MIN_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS: u32 = 120;
/// Largest signed lease runway accepted by AMDI v1.
pub const MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS: u32 = 24 * 60 * 60;
/// Default signed lease runway for clients that do not select a stricter policy.
pub const DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS: u32 =
    MIN_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS;

/// Privacy-safe AMDI validation failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AnonymousMailboxDepositInvitationError {
    /// The complete invitation or a nested frame exceeds its frozen bound.
    #[error("anonymous mailbox deposit invitation exceeds its limit")]
    TooLarge,
    /// The framing or a nested value is not canonical.
    #[error("anonymous mailbox deposit invitation is malformed")]
    Malformed,
    /// The invitation codec version is not supported.
    #[error("anonymous mailbox deposit invitation version is unsupported")]
    UnsupportedVersion,
    /// The invitation or proven lease is outside its usable time window.
    #[error("anonymous mailbox deposit invitation is expired")]
    Expired,
    /// Two independently supplied claims do not describe one capability.
    #[error("anonymous mailbox deposit invitation claims do not match")]
    ClaimsConflict,
    /// A target, lease, or invitation signature was rejected.
    #[error("anonymous mailbox deposit invitation signature was rejected")]
    SignatureRejected,
}

impl From<AnonymousMailboxProtocolError> for AnonymousMailboxDepositInvitationError {
    fn from(value: AnonymousMailboxProtocolError) -> Self {
        match value {
            AnonymousMailboxProtocolError::TooLarge => Self::TooLarge,
            AnonymousMailboxProtocolError::UnsupportedVersion => Self::UnsupportedVersion,
            AnonymousMailboxProtocolError::Expired => Self::Expired,
            AnonymousMailboxProtocolError::ClaimsConflict => Self::ClaimsConflict,
            AnonymousMailboxProtocolError::SignatureRejected => Self::SignatureRejected,
            AnonymousMailboxProtocolError::Malformed
            | AnonymousMailboxProtocolError::UnsupportedOperation
            | AnonymousMailboxProtocolError::ProofRejected => Self::Malformed,
        }
    }
}

impl From<AnonymousMailboxRecipientSealError> for AnonymousMailboxDepositInvitationError {
    fn from(value: AnonymousMailboxRecipientSealError) -> Self {
        match value {
            AnonymousMailboxRecipientSealError::TooLarge => Self::TooLarge,
            AnonymousMailboxRecipientSealError::UnsupportedVersion => Self::UnsupportedVersion,
            AnonymousMailboxRecipientSealError::ClaimsConflict => Self::ClaimsConflict,
            AnonymousMailboxRecipientSealError::Malformed
            | AnonymousMailboxRecipientSealError::KeyUnavailable
            | AnonymousMailboxRecipientSealError::Rejected => Self::Malformed,
        }
    }
}

/// Receiver-selected, read-signed exact custody target.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AnonymousMailboxDepositTargetPinV1 {
    node_id: [u8; 32],
    descriptor_sequence: u64,
    descriptor_commitment: [u8; 32],
}

impl AnonymousMailboxDepositTargetPinV1 {
    /// Creates one exact target pin. Descriptor authenticity/freshness remains
    /// the source router's existing responsibility.
    ///
    /// # Errors
    ///
    /// Returns a coarse error if the node key or descriptor pin is malformed.
    pub fn new(
        node_id: [u8; 32],
        descriptor_sequence: u64,
        descriptor_commitment: [u8; 32],
    ) -> Result<Self, AnonymousMailboxDepositInvitationError> {
        IdentityPublicKey::from_bytes(&node_id)
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
        if descriptor_sequence == 0 || descriptor_commitment.iter().all(|byte| *byte == 0) {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        Ok(Self {
            node_id,
            descriptor_sequence,
            descriptor_commitment,
        })
    }

    /// Exact target node id.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Exact signed descriptor sequence.
    #[must_use]
    pub const fn descriptor_sequence(&self) -> u64 {
        self.descriptor_sequence
    }

    /// Exact signed descriptor commitment.
    #[must_use]
    pub const fn descriptor_commitment(&self) -> [u8; 32] {
        self.descriptor_commitment
    }
}

impl fmt::Debug for AnonymousMailboxDepositTargetPinV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxDepositTargetPinV1")
            .field("descriptor_sequence", &self.descriptor_sequence)
            .finish_non_exhaustive()
    }
}

/// Redacted metadata returned after full invitation verification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AnonymousMailboxDepositInvitationSummaryV1 {
    invitation_id: [u8; 16],
    target: AnonymousMailboxDepositTargetPinV1,
    mailbox_id: [u8; 32],
    lease_claims_commitment: [u8; 32],
    invitation_expires_at: u64,
    lease_expires_at: u64,
    min_remaining_lease_secs: u32,
}

impl AnonymousMailboxDepositInvitationSummaryV1 {
    /// Random exact-replay identifier of this invitation.
    #[must_use]
    pub const fn invitation_id(&self) -> [u8; 16] {
        self.invitation_id
    }

    /// Receiver-selected exact custody target pin.
    #[must_use]
    pub const fn target(&self) -> AnonymousMailboxDepositTargetPinV1 {
        self.target
    }

    /// Unlinkable mailbox id proven by the accepted lease.
    #[must_use]
    pub const fn mailbox_id(&self) -> [u8; 32] {
        self.mailbox_id
    }

    /// Commitment to every immutable lease claim.
    #[must_use]
    pub const fn lease_claims_commitment(&self) -> [u8; 32] {
        self.lease_claims_commitment
    }

    /// Client-policy expiry of this invitation.
    #[must_use]
    pub const fn invitation_expires_at(&self) -> u64 {
        self.invitation_expires_at
    }

    /// Terminal-enforced expiry of the proven lease.
    #[must_use]
    pub const fn lease_expires_at(&self) -> u64 {
        self.lease_expires_at
    }

    /// Signed minimum remaining lease time required before a new Put.
    /// Required lease runway for new effects.
    #[must_use]
    pub const fn min_remaining_lease_secs(&self) -> u32 {
        self.min_remaining_lease_secs
    }
}

impl fmt::Debug for AnonymousMailboxDepositInvitationSummaryV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxDepositInvitationSummaryV1")
            .field("invitation_expires_at", &self.invitation_expires_at)
            .field("lease_expires_at", &self.lease_expires_at)
            .field("min_remaining_lease_secs", &self.min_remaining_lease_secs)
            .finish_non_exhaustive()
    }
}

/// Canonical receiver-shared Put-only bearer capability.
///
/// Private fields prevent callers from treating the embedded seed or proof as
/// independently authoritative. Decode, verify, and Put preparation stay one
/// composition boundary.
pub struct AnonymousMailboxDepositInvitationV1 {
    version: u16,
    invitation_id: [u8; 16],
    issued_at: u64,
    expires_at: u64,
    min_remaining_lease_secs: u32,
    target: AnonymousMailboxDepositTargetPinV1,
    lease_claims_commitment: [u8; 32],
    deposit_seed: [u8; 32],
    lease_create_frame: Vec<u8>,
    lease_response_frame: Vec<u8>,
    signature: [u8; 64],
}

impl AnonymousMailboxDepositInvitationV1 {
    /// Creates and read-signs one exact invitation.
    ///
    /// # Errors
    ///
    /// Returns a coarse error if any policy, nested proof, capability, or
    /// signature invariant is invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        invitation_id: [u8; 16],
        issued_at: u64,
        expires_at: u64,
        min_remaining_lease_secs: u32,
        target: AnonymousMailboxDepositTargetPinV1,
        deposit_seed: [u8; 32],
        lease_create_frame: Vec<u8>,
        lease_response_frame: Vec<u8>,
        reader: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxDepositInvitationError> {
        let (lease, responded_at) =
            decode_accepted_lease(&target, &lease_create_frame, &lease_response_frame)?;
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V1,
            invitation_id,
            issued_at,
            expires_at,
            min_remaining_lease_secs,
            target,
            lease_claims_commitment: lease.claims_commitment(),
            deposit_seed,
            lease_create_frame,
            lease_response_frame,
            signature: [0; 64],
        };
        value.validate_unsigned_at(issued_at)?;
        if responded_at > issued_at.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS) {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        if reader.public_key_bytes() != lease.read_verifier {
            return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
        }
        let mut transcript = value.signing_bytes()?;
        value.signature = reader.sign(&transcript);
        transcript.zeroize();
        value.verify_at(issued_at)?;
        Ok(value)
    }

    /// Decodes exactly one canonical invitation and verifies it for `now`.
    ///
    /// # Errors
    ///
    /// Returns a coarse error for invalid bounds, framing, policy, proof, or
    /// signatures.
    pub fn decode_at(
        encoded: &[u8],
        now: u64,
    ) -> Result<Self, AnonymousMailboxDepositInvitationError> {
        if encoded.len() > MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        if encoded.len() < FIXED_PREFIX_BYTES + (2 * LENGTH_BYTES) + SIGNATURE_BYTES {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        if encoded[..4] != INVITATION_MAGIC {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        let mut offset = 4;
        let version = take_u16(encoded, &mut offset)?;
        if version != ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V1 {
            return Err(AnonymousMailboxDepositInvitationError::UnsupportedVersion);
        }
        let declared = usize::try_from(take_u32(encoded, &mut offset)?)
            .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?;
        if declared != encoded.len() {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        let invitation_id = take::<16>(encoded, &mut offset)?;
        let issued_at = take_u64(encoded, &mut offset)?;
        let expires_at = take_u64(encoded, &mut offset)?;
        let min_remaining_lease_secs = take_u32(encoded, &mut offset)?;
        let node_id = take::<32>(encoded, &mut offset)?;
        let descriptor_sequence = take_u64(encoded, &mut offset)?;
        let descriptor_commitment = take::<32>(encoded, &mut offset)?;
        let target = AnonymousMailboxDepositTargetPinV1::new(
            node_id,
            descriptor_sequence,
            descriptor_commitment,
        )?;
        let lease_claims_commitment = take::<32>(encoded, &mut offset)?;
        let deposit_seed = take::<32>(encoded, &mut offset)?;
        let lease_create_len = usize::from(take_u16(encoded, &mut offset)?);
        if lease_create_len == 0 || lease_create_len > MAX_LEASE_CREATE_FRAME_BYTES {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        let lease_create_frame = take_vec(encoded, &mut offset, lease_create_len)?;
        let lease_response_len = usize::from(take_u16(encoded, &mut offset)?);
        if lease_response_len == 0 || lease_response_len > MAX_LEASE_RESPONSE_FRAME_BYTES {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        let lease_response_frame = take_vec(encoded, &mut offset, lease_response_len)?;
        let signature = take::<64>(encoded, &mut offset)?;
        if offset != encoded.len() {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        let value = Self {
            version,
            invitation_id,
            issued_at,
            expires_at,
            min_remaining_lease_secs,
            target,
            lease_claims_commitment,
            deposit_seed,
            lease_create_frame,
            lease_response_frame,
            signature,
        };
        value.verify_at(now)?;
        let mut canonical = value.encode()?;
        if canonical != encoded {
            canonical.zeroize();
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        canonical.zeroize();
        Ok(value)
    }

    /// Verifies the invitation, its accepted historical lease proof, and the
    /// current client-policy time/runway boundary.
    ///
    /// # Errors
    ///
    /// Returns a coarse error if the invitation, nested lease proof, or signed
    /// client policy is invalid at `now`.
    pub fn verify_at(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxDepositInvitationSummaryV1, AnonymousMailboxDepositInvitationError>
    {
        let lease = self.validate_unsigned_at(now)?;
        let mut transcript = self.signing_bytes()?;
        let result = IdentityPublicKey::from_bytes(&lease.read_verifier)
            .and_then(|key| key.verify(&transcript, &self.signature))
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected);
        transcript.zeroize();
        result?;
        Ok(AnonymousMailboxDepositInvitationSummaryV1 {
            invitation_id: self.invitation_id,
            target: self.target,
            mailbox_id: lease.mailbox_id,
            lease_claims_commitment: self.lease_claims_commitment,
            invitation_expires_at: self.expires_at,
            lease_expires_at: lease.expires_at,
            min_remaining_lease_secs: self.min_remaining_lease_secs,
        })
    }

    /// Returns the exact canonical bearer bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if a nested frame or the complete invitation exceeds
    /// its frozen bound.
    pub fn encode(&self) -> Result<Vec<u8>, AnonymousMailboxDepositInvitationError> {
        let mut encoded = self.unsigned_wire_bytes()?;
        encoded.extend_from_slice(&self.signature);
        if encoded.len() > MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES {
            encoded.zeroize();
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        Ok(encoded)
    }

    /// Returns a domain-separated exact-replay commitment without exposing the
    /// invitation id, mailbox id, target id, or deposit seed.
    ///
    /// # Errors
    ///
    /// Returns an error if the invitation cannot be encoded canonically.
    pub fn commitment(&self) -> Result<[u8; 32], AnonymousMailboxDepositInvitationError> {
        let mut encoded = self.encode()?;
        let mut hasher = Sha256::new();
        hasher.update(INVITATION_COMMITMENT_DOMAIN);
        hasher.update(
            u32::try_from(encoded.len())
                .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?
                .to_le_bytes(),
        );
        hasher.update(&encoded);
        encoded.zeroize();
        Ok(hasher.finalize().into())
    }

    /// Creates one Put using only the embedded deposit capability. There is no
    /// corresponding Pull or Ack API on this type.
    ///
    /// # Errors
    ///
    /// Returns a coarse error if the invitation is unusable, the Put window is
    /// stale or exceeds the lease, or the sealed item violates protocol bounds.
    pub fn prepare_put(
        &self,
        item_id: [u8; 16],
        sealed_envelope: Vec<u8>,
        issued_at: u64,
        expires_at: u64,
        now: u64,
    ) -> Result<AnonymousMailboxPutV1, AnonymousMailboxDepositInvitationError> {
        let summary = self.verify_at(now)?;
        if issued_at > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
            || expires_at > summary.lease_expires_at
        {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        let depositor = IdentityKeyPair::from_bytes(&self.deposit_seed)
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
        let deposit_verifier = depositor.public_key_bytes();
        let put = AnonymousMailboxPutV1::new(
            summary.mailbox_id,
            item_id,
            sealed_envelope,
            issued_at,
            expires_at,
            &depositor,
        )?;
        put.verify_at(&deposit_verifier, now)?;
        Ok(put)
    }

    fn validate_unsigned_at(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxLeaseCreateV1, AnonymousMailboxDepositInvitationError> {
        if self.version != ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V1 {
            return Err(AnonymousMailboxDepositInvitationError::UnsupportedVersion);
        }
        if self.invitation_id.iter().all(|byte| *byte == 0) {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        validate_invitation_window(
            self.issued_at,
            self.expires_at,
            self.min_remaining_lease_secs,
            now,
        )?;
        let (lease, responded_at) = decode_accepted_lease(
            &self.target,
            &self.lease_create_frame,
            &self.lease_response_frame,
        )?;
        if responded_at
            > self
                .issued_at
                .saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
        {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        if self.lease_claims_commitment != lease.claims_commitment() {
            return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
        }
        let derived = IdentityKeyPair::from_bytes(&self.deposit_seed)
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
        if derived.public_key_bytes() != lease.deposit_verifier {
            return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
        }
        let runway = u64::from(self.min_remaining_lease_secs);
        let latest_invitation_expiry = lease
            .expires_at
            .checked_sub(runway)
            .ok_or(AnonymousMailboxDepositInvitationError::Expired)?;
        if self.expires_at > latest_invitation_expiry
            || now
                .checked_add(runway)
                .ok_or(AnonymousMailboxDepositInvitationError::Expired)?
                > lease.expires_at
            || lease.issued_at > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
        {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        Ok(lease)
    }

    fn signing_bytes(&self) -> Result<Zeroizing<Vec<u8>>, AnonymousMailboxDepositInvitationError> {
        let mut unsigned = self.unsigned_wire_bytes()?;
        let mut transcript = Zeroizing::new(Vec::with_capacity(
            INVITATION_SIGNATURE_DOMAIN.len() + unsigned.len(),
        ));
        transcript.extend_from_slice(INVITATION_SIGNATURE_DOMAIN);
        transcript.extend_from_slice(&unsigned);
        unsigned.zeroize();
        Ok(transcript)
    }

    fn encoded_len(&self) -> Result<usize, AnonymousMailboxDepositInvitationError> {
        FIXED_PREFIX_BYTES
            .checked_add(LENGTH_BYTES)
            .and_then(|value| value.checked_add(self.lease_create_frame.len()))
            .and_then(|value| value.checked_add(LENGTH_BYTES))
            .and_then(|value| value.checked_add(self.lease_response_frame.len()))
            .and_then(|value| value.checked_add(SIGNATURE_BYTES))
            .filter(|value| *value <= MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES)
            .ok_or(AnonymousMailboxDepositInvitationError::TooLarge)
    }

    fn unsigned_wire_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxDepositInvitationError> {
        if self.lease_create_frame.is_empty()
            || self.lease_create_frame.len() > MAX_LEASE_CREATE_FRAME_BYTES
            || self.lease_response_frame.is_empty()
            || self.lease_response_frame.len() > MAX_LEASE_RESPONSE_FRAME_BYTES
        {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        let lease_create_len = u16::try_from(self.lease_create_frame.len())
            .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?;
        let lease_response_len = u16::try_from(self.lease_response_frame.len())
            .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?;
        let total_len = self.encoded_len()?;
        let total_len = u32::try_from(total_len)
            .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?;
        let mut encoded = Vec::with_capacity(total_len as usize - SIGNATURE_BYTES);
        encoded.extend_from_slice(&INVITATION_MAGIC);
        encoded.extend_from_slice(&self.version.to_le_bytes());
        encoded.extend_from_slice(&total_len.to_le_bytes());
        encoded.extend_from_slice(&self.invitation_id);
        encoded.extend_from_slice(&self.issued_at.to_le_bytes());
        encoded.extend_from_slice(&self.expires_at.to_le_bytes());
        encoded.extend_from_slice(&self.min_remaining_lease_secs.to_le_bytes());
        encoded.extend_from_slice(&self.target.node_id);
        encoded.extend_from_slice(&self.target.descriptor_sequence.to_le_bytes());
        encoded.extend_from_slice(&self.target.descriptor_commitment);
        encoded.extend_from_slice(&self.lease_claims_commitment);
        encoded.extend_from_slice(&self.deposit_seed);
        encoded.extend_from_slice(&lease_create_len.to_le_bytes());
        encoded.extend_from_slice(&self.lease_create_frame);
        encoded.extend_from_slice(&lease_response_len.to_le_bytes());
        encoded.extend_from_slice(&self.lease_response_frame);
        debug_assert_eq!(encoded.len() + SIGNATURE_BYTES, total_len as usize);
        Ok(encoded)
    }
}

impl fmt::Debug for AnonymousMailboxDepositInvitationV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxDepositInvitationV1")
            .field("version", &self.version)
            .field("encoded_bytes", &self.encoded_len())
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .field("min_remaining_lease_secs", &self.min_remaining_lease_secs)
            .finish_non_exhaustive()
    }
}

impl Drop for AnonymousMailboxDepositInvitationV1 {
    fn drop(&mut self) {
        self.deposit_seed.zeroize();
    }
}

/// Redacted metadata returned after current AMDI v2 admission verification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AnonymousMailboxDepositInvitationSummaryV2 {
    invitation_id: [u8; 16],
    target: AnonymousMailboxDepositTargetPinV1,
    mailbox_id: [u8; 32],
    lease_claims_commitment: [u8; 32],
    invitation_expires_at: u64,
    lease_expires_at: u64,
    min_remaining_lease_secs: u32,
    recipient_seal: AnonymousMailboxRecipientSealPublicV1,
}

impl AnonymousMailboxDepositInvitationSummaryV2 {
    /// Random exact-replay identifier.
    #[must_use]
    pub const fn invitation_id(&self) -> [u8; 16] {
        self.invitation_id
    }

    /// Receiver-selected exact custody target.
    #[must_use]
    pub const fn target(&self) -> AnonymousMailboxDepositTargetPinV1 {
        self.target
    }

    /// Unlinkable mailbox id from the accepted lease.
    #[must_use]
    pub const fn mailbox_id(&self) -> [u8; 32] {
        self.mailbox_id
    }

    /// Commitment to every immutable lease claim.
    #[must_use]
    pub const fn lease_claims_commitment(&self) -> [u8; 32] {
        self.lease_claims_commitment
    }

    /// Client-policy deadline for new effects.
    #[must_use]
    pub const fn invitation_expires_at(&self) -> u64 {
        self.invitation_expires_at
    }

    /// Terminal-enforced lease expiry.
    #[must_use]
    pub const fn lease_expires_at(&self) -> u64 {
        self.lease_expires_at
    }

    /// Required lease runway for new effects.
    #[must_use]
    pub const fn min_remaining_lease_secs(&self) -> u32 {
        self.min_remaining_lease_secs
    }

    /// Signed public recipient-seal capability.
    #[must_use]
    pub const fn recipient_seal(&self) -> AnonymousMailboxRecipientSealPublicV1 {
        self.recipient_seal
    }
}

impl fmt::Debug for AnonymousMailboxDepositInvitationSummaryV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxDepositInvitationSummaryV2")
            .field("invitation_expires_at", &self.invitation_expires_at)
            .field("lease_expires_at", &self.lease_expires_at)
            .field("min_remaining_lease_secs", &self.min_remaining_lease_secs)
            .finish_non_exhaustive()
    }
}

/// Canonical receiver-shared recipient-sealed deposit capability.
///
/// [ANONYMOUS-MAILBOX-RECIPIENT-SEAL 2026-09-08 by Codex] V2 binds a native
/// recipient encryption key to the exact signed V1 lease proof. It neither
/// exports nor persists the recipient private key.
pub struct AnonymousMailboxDepositInvitationV2 {
    version: u16,
    invitation_id: [u8; 16],
    issued_at: u64,
    expires_at: u64,
    min_remaining_lease_secs: u32,
    target: AnonymousMailboxDepositTargetPinV1,
    lease_claims_commitment: [u8; 32],
    deposit_seed: [u8; 32],
    chat_receiver: [u8; 32],
    recipient_seal: AnonymousMailboxRecipientSealPublicV1,
    lease_create_frame: Vec<u8>,
    lease_response_frame: Vec<u8>,
    signature: [u8; 64],
}

/// Short-lived typed authority for creating one new recipient-sealed Put.
pub struct AnonymousMailboxDepositInvitationActiveV2<'a> {
    invitation: &'a AnonymousMailboxDepositInvitationV2,
    summary: AnonymousMailboxDepositInvitationSummaryV2,
}

impl AnonymousMailboxDepositInvitationV2 {
    /// Creates and read-signs one exact recipient-sealed invitation.
    ///
    /// `chat_receiver` must come from authenticated local/contact context.
    /// The `reader` signature authenticates mailbox authority and is not proof
    /// that the same party controls the chat identity.
    ///
    /// # Errors
    /// Returns a coarse invitation error when nested frames, signatures,
    /// claims, keys, bounds, or current time policy are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        invitation_id: [u8; 16],
        issued_at: u64,
        expires_at: u64,
        min_remaining_lease_secs: u32,
        target: AnonymousMailboxDepositTargetPinV1,
        deposit_seed: [u8; 32],
        chat_receiver: [u8; 32],
        recipient_seal: AnonymousMailboxRecipientSealPublicV1,
        lease_create_frame: Vec<u8>,
        lease_response_frame: Vec<u8>,
        reader: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxDepositInvitationError> {
        let (lease, responded_at) =
            decode_accepted_lease(&target, &lease_create_frame, &lease_response_frame)?;
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V2,
            invitation_id,
            issued_at,
            expires_at,
            min_remaining_lease_secs,
            target,
            lease_claims_commitment: lease.claims_commitment(),
            deposit_seed,
            chat_receiver,
            recipient_seal,
            lease_create_frame,
            lease_response_frame,
            signature: [0; 64],
        };
        if responded_at > issued_at.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS) {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        if reader.public_key_bytes() != lease.read_verifier {
            return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
        }
        // [ANONYMOUS-MAILBOX-RECIPIENT-SEAL 2026-09-08 by Codex] The caller
        // supplies this identity from authenticated local/contact context;
        // the pseudonymous mailbox reader is not a chat-identity proof.
        IdentityPublicKey::from_bytes(&value.chat_receiver)
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
        let mut transcript = value.signing_bytes()?;
        value.signature = reader.sign(&transcript);
        transcript.zeroize();
        value.verify_for_new_put_at(issued_at)?;
        Ok(value)
    }

    /// Decodes canonical V2 bytes and verifies current new-effect authority.
    ///
    /// # Errors
    /// Returns a coarse invitation error for non-canonical, unauthenticated,
    /// mismatched, oversized, unsupported, or expired input.
    pub fn decode_at(
        encoded: &[u8],
        now: u64,
    ) -> Result<Self, AnonymousMailboxDepositInvitationError> {
        let value = Self::decode_static(encoded)?;
        value.verify_for_new_put_at(now)?;
        Ok(value)
    }

    /// Decodes a historical invitation into an open-only context.
    ///
    /// Current invitation expiry is intentionally not consulted. The decoded
    /// bearer and its deposit seed are dropped and zeroized before return.
    ///
    /// # Errors
    /// Returns a coarse invitation error when immutable history is invalid.
    pub fn decode_historical_recipient_context(
        encoded: &[u8],
    ) -> Result<AnonymousMailboxRecipientOpenContextV1, AnonymousMailboxDepositInvitationError>
    {
        let value = Self::decode_static(encoded)?;
        value.verify_historical_recipient_context()
    }

    /// Verifies current expiry/skew/runway and returns the sole new-effect type.
    ///
    /// # Errors
    /// Returns a coarse invitation error when authentication, claims, or the
    /// current effect window fail validation.
    pub fn verify_for_new_put_at(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxDepositInvitationActiveV2<'_>, AnonymousMailboxDepositInvitationError>
    {
        let summary = self.summary_at(now)?;
        Ok(AnonymousMailboxDepositInvitationActiveV2 {
            invitation: self,
            summary,
        })
    }

    /// Revalidates immutable signed history and returns open-only metadata.
    ///
    /// # Errors
    /// Returns a coarse invitation error when immutable history is invalid.
    pub fn verify_historical_recipient_context(
        &self,
    ) -> Result<AnonymousMailboxRecipientOpenContextV1, AnonymousMailboxDepositInvitationError>
    {
        let lease = self.validate_static()?;
        let binding = self.recipient_binding(lease.mailbox_id)?;
        Ok(AnonymousMailboxRecipientOpenContextV1::new(
            binding,
            lease.expires_at,
        ))
    }

    /// Returns exact signed V2 bytes.
    ///
    /// # Errors
    /// Returns [`AnonymousMailboxDepositInvitationError::TooLarge`] when the
    /// canonical frame exceeds its frozen bound.
    pub fn encode(&self) -> Result<Vec<u8>, AnonymousMailboxDepositInvitationError> {
        let mut encoded = self.unsigned_wire_bytes()?;
        encoded.extend_from_slice(&self.signature);
        if encoded.len() > MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES {
            encoded.zeroize();
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        Ok(encoded)
    }

    /// Domain-separated commitment to the exact signed V2 bytes.
    ///
    /// # Errors
    /// Returns a coarse invitation error when the canonical frame cannot be
    /// represented within its frozen bound.
    pub fn commitment(&self) -> Result<[u8; 32], AnonymousMailboxDepositInvitationError> {
        let mut encoded = self.encode()?;
        let mut hasher = Sha256::new();
        hasher.update(INVITATION_V2_COMMITMENT_DOMAIN);
        hasher.update(
            u32::try_from(encoded.len())
                .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?
                .to_le_bytes(),
        );
        hasher.update(&encoded);
        encoded.zeroize();
        Ok(hasher.finalize().into())
    }

    fn decode_static(encoded: &[u8]) -> Result<Self, AnonymousMailboxDepositInvitationError> {
        if encoded.len() > MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        if encoded.len() < FIXED_PREFIX_BYTES_V2 + (2 * LENGTH_BYTES) + SIGNATURE_BYTES
            || encoded[..4] != INVITATION_MAGIC
        {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        let mut offset = 4;
        let version = take_u16(encoded, &mut offset)?;
        if version != ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V2 {
            return Err(AnonymousMailboxDepositInvitationError::UnsupportedVersion);
        }
        let declared = usize::try_from(take_u32(encoded, &mut offset)?)
            .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?;
        if declared != encoded.len() {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        let invitation_id = take::<16>(encoded, &mut offset)?;
        let issued_at = take_u64(encoded, &mut offset)?;
        let expires_at = take_u64(encoded, &mut offset)?;
        let min_remaining_lease_secs = take_u32(encoded, &mut offset)?;
        let target = AnonymousMailboxDepositTargetPinV1::new(
            take::<32>(encoded, &mut offset)?,
            take_u64(encoded, &mut offset)?,
            take::<32>(encoded, &mut offset)?,
        )?;
        let lease_claims_commitment = take::<32>(encoded, &mut offset)?;
        let deposit_seed = Zeroizing::new(take::<32>(encoded, &mut offset)?);
        let chat_receiver = take::<32>(encoded, &mut offset)?;
        let recipient_algorithm = take::<1>(encoded, &mut offset)?[0];
        if recipient_algorithm
            != crate::protocol::anonymous_mailbox_recipient_seal::ANONYMOUS_MAILBOX_RECIPIENT_SEAL_ALGORITHM_V1
        {
            return Err(AnonymousMailboxDepositInvitationError::UnsupportedVersion);
        }
        let recipient_seal = AnonymousMailboxRecipientSealPublicV1::new(
            take::<16>(encoded, &mut offset)?,
            take::<32>(encoded, &mut offset)?,
        )?;
        let lease_create_len = usize::from(take_u16(encoded, &mut offset)?);
        if lease_create_len == 0 || lease_create_len > MAX_LEASE_CREATE_FRAME_BYTES {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        let lease_create_frame = take_vec(encoded, &mut offset, lease_create_len)?;
        let lease_response_len = usize::from(take_u16(encoded, &mut offset)?);
        if lease_response_len == 0 || lease_response_len > MAX_LEASE_RESPONSE_FRAME_BYTES {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        let lease_response_frame = take_vec(encoded, &mut offset, lease_response_len)?;
        let signature = take::<64>(encoded, &mut offset)?;
        if offset != encoded.len() {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        let value = Self {
            version,
            invitation_id,
            issued_at,
            expires_at,
            min_remaining_lease_secs,
            target,
            lease_claims_commitment,
            deposit_seed: *deposit_seed,
            chat_receiver,
            recipient_seal,
            lease_create_frame,
            lease_response_frame,
            signature,
        };
        value.validate_static()?;
        let mut canonical = value.encode()?;
        if canonical != encoded {
            canonical.zeroize();
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        canonical.zeroize();
        Ok(value)
    }

    fn validate_static(
        &self,
    ) -> Result<AnonymousMailboxLeaseCreateV1, AnonymousMailboxDepositInvitationError> {
        if self.version != ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_VERSION_V2
            || self.invitation_id.iter().all(|byte| *byte == 0)
        {
            return Err(AnonymousMailboxDepositInvitationError::Malformed);
        }
        let ttl = self
            .expires_at
            .checked_sub(self.issued_at)
            .ok_or(AnonymousMailboxDepositInvitationError::Expired)?;
        if ttl == 0
            || ttl > MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_TTL_SECS
            || !(MIN_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS
                ..=MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS)
                .contains(&self.min_remaining_lease_secs)
        {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        let (lease, responded_at) = decode_accepted_lease(
            &self.target,
            &self.lease_create_frame,
            &self.lease_response_frame,
        )?;
        if self.lease_claims_commitment != lease.claims_commitment() {
            return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
        }
        let derived = IdentityKeyPair::from_bytes(&self.deposit_seed)
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
        if derived.public_key_bytes() != lease.deposit_verifier {
            return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
        }
        IdentityPublicKey::from_bytes(&self.chat_receiver)
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
        let runway = u64::from(self.min_remaining_lease_secs);
        let latest_expiry = lease
            .expires_at
            .checked_sub(runway)
            .ok_or(AnonymousMailboxDepositInvitationError::Expired)?;
        if self.expires_at > latest_expiry
            || lease.issued_at
                > self
                    .issued_at
                    .saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
            || responded_at
                > self
                    .issued_at
                    .saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
        {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        let mut transcript = self.signing_bytes()?;
        let verification = IdentityPublicKey::from_bytes(&lease.read_verifier)
            .and_then(|key| key.verify(&transcript, &self.signature))
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected);
        transcript.zeroize();
        verification?;
        Ok(lease)
    }

    // [ANONYMOUS-MAILBOX-RECIPIENT-SEAL 2026-09-08 by Codex] Keep static
    // authentication and current-effect time admission in one reusable gate.
    fn active_lease_at(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxLeaseCreateV1, AnonymousMailboxDepositInvitationError> {
        let lease = self.validate_static()?;
        if now > self.expires_at
            || self.issued_at > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
            || lease.issued_at > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
            || now
                .checked_add(u64::from(self.min_remaining_lease_secs))
                .ok_or(AnonymousMailboxDepositInvitationError::Expired)?
                > lease.expires_at
        {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        Ok(lease)
    }

    fn recipient_binding(
        &self,
        mailbox_id: [u8; 32],
    ) -> Result<AnonymousMailboxRecipientSealBindingV1, AnonymousMailboxDepositInvitationError>
    {
        Ok(AnonymousMailboxRecipientSealBindingV1::new(
            self.commitment()?,
            self.target.node_id,
            self.target.descriptor_sequence,
            self.target.descriptor_commitment,
            mailbox_id,
            self.lease_claims_commitment,
            self.chat_receiver,
            self.recipient_seal,
        )?)
    }

    fn signing_bytes(&self) -> Result<Zeroizing<Vec<u8>>, AnonymousMailboxDepositInvitationError> {
        let mut unsigned = self.unsigned_wire_bytes()?;
        let mut transcript = Zeroizing::new(Vec::with_capacity(
            INVITATION_V2_SIGNATURE_DOMAIN.len() + unsigned.len(),
        ));
        transcript.extend_from_slice(INVITATION_V2_SIGNATURE_DOMAIN);
        transcript.extend_from_slice(&unsigned);
        unsigned.zeroize();
        Ok(transcript)
    }

    fn encoded_len(&self) -> Result<usize, AnonymousMailboxDepositInvitationError> {
        FIXED_PREFIX_BYTES_V2
            .checked_add(LENGTH_BYTES)
            .and_then(|value| value.checked_add(self.lease_create_frame.len()))
            .and_then(|value| value.checked_add(LENGTH_BYTES))
            .and_then(|value| value.checked_add(self.lease_response_frame.len()))
            .and_then(|value| value.checked_add(SIGNATURE_BYTES))
            .filter(|value| *value <= MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES)
            .ok_or(AnonymousMailboxDepositInvitationError::TooLarge)
    }

    fn unsigned_wire_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxDepositInvitationError> {
        if self.lease_create_frame.is_empty()
            || self.lease_create_frame.len() > MAX_LEASE_CREATE_FRAME_BYTES
            || self.lease_response_frame.is_empty()
            || self.lease_response_frame.len() > MAX_LEASE_RESPONSE_FRAME_BYTES
        {
            return Err(AnonymousMailboxDepositInvitationError::TooLarge);
        }
        let total_len = self.encoded_len()?;
        let mut encoded = Vec::with_capacity(total_len - SIGNATURE_BYTES);
        encoded.extend_from_slice(&INVITATION_MAGIC);
        encoded.extend_from_slice(&self.version.to_le_bytes());
        encoded.extend_from_slice(
            &u32::try_from(total_len)
                .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?
                .to_le_bytes(),
        );
        encoded.extend_from_slice(&self.invitation_id);
        encoded.extend_from_slice(&self.issued_at.to_le_bytes());
        encoded.extend_from_slice(&self.expires_at.to_le_bytes());
        encoded.extend_from_slice(&self.min_remaining_lease_secs.to_le_bytes());
        encoded.extend_from_slice(&self.target.node_id);
        encoded.extend_from_slice(&self.target.descriptor_sequence.to_le_bytes());
        encoded.extend_from_slice(&self.target.descriptor_commitment);
        encoded.extend_from_slice(&self.lease_claims_commitment);
        encoded.extend_from_slice(&self.deposit_seed);
        encoded.extend_from_slice(&self.chat_receiver);
        encoded.push(self.recipient_seal.algorithm());
        encoded.extend_from_slice(&self.recipient_seal.key_id());
        encoded.extend_from_slice(&self.recipient_seal.public_key());
        encoded.extend_from_slice(
            &u16::try_from(self.lease_create_frame.len())
                .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?
                .to_le_bytes(),
        );
        encoded.extend_from_slice(&self.lease_create_frame);
        encoded.extend_from_slice(
            &u16::try_from(self.lease_response_frame.len())
                .map_err(|_| AnonymousMailboxDepositInvitationError::TooLarge)?
                .to_le_bytes(),
        );
        encoded.extend_from_slice(&self.lease_response_frame);
        Ok(encoded)
    }
}

impl AnonymousMailboxDepositInvitationActiveV2<'_> {
    /// Returns the redacted projection verified when this authority was admitted.
    ///
    /// This is metadata only: it does not extend the admission window or expose
    /// the deposit seed, recipient private key, or raw lease verifier.
    #[must_use]
    pub const fn summary(&self) -> AnonymousMailboxDepositInvitationSummaryV2 {
        self.summary
    }

    /// Creates canonical AMSI bytes after revalidating current authority.
    ///
    /// # Errors
    /// Returns a coarse seal error when authority is no longer current or the
    /// envelope, receiver binding, key, or ciphertext is invalid.
    pub fn seal_chat_envelope_at(
        &self,
        envelope: &ChatEnvelope,
        now: u64,
    ) -> Result<AnonymousMailboxRecipientSealedItemV1, AnonymousMailboxRecipientSealError> {
        let lease = self
            .invitation
            .active_lease_at(now)
            .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?;
        let binding = self
            .invitation
            .recipient_binding(lease.mailbox_id)
            .map_err(|_| AnonymousMailboxRecipientSealError::Rejected)?;
        Ok(AnonymousMailboxRecipientSealedItemV1::new(
            seal_chat_envelope(&binding, envelope)?,
            &binding,
        ))
    }

    /// Consumes AMSI bytes created by this typed active context into one Put.
    ///
    /// # Errors
    /// Returns a coarse invitation error when time, lease, size, or signing
    /// constraints reject the new mutation.
    pub fn prepare_put(
        &self,
        item_id: [u8; 16],
        sealed_envelope: AnonymousMailboxRecipientSealedItemV1,
        issued_at: u64,
        expires_at: u64,
        now: u64,
    ) -> Result<AnonymousMailboxPutV1, AnonymousMailboxDepositInvitationError> {
        let summary = self.invitation.summary_at(now)?;
        let binding = self.invitation.recipient_binding(summary.mailbox_id)?;
        if !sealed_envelope.matches_binding(&binding) {
            return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
        }
        if issued_at > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
            || expires_at > summary.lease_expires_at
        {
            return Err(AnonymousMailboxDepositInvitationError::Expired);
        }
        let depositor = IdentityKeyPair::from_bytes(&self.invitation.deposit_seed)
            .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
        let put = AnonymousMailboxPutV1::new(
            summary.mailbox_id,
            item_id,
            sealed_envelope.into_bytes(),
            issued_at,
            expires_at,
            &depositor,
        )?;
        put.verify_at(&depositor.public_key_bytes(), now)?;
        Ok(put)
    }
}

impl AnonymousMailboxDepositInvitationV2 {
    fn summary_at(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxDepositInvitationSummaryV2, AnonymousMailboxDepositInvitationError>
    {
        let lease = self.active_lease_at(now)?;
        Ok(AnonymousMailboxDepositInvitationSummaryV2 {
            invitation_id: self.invitation_id,
            target: self.target,
            mailbox_id: lease.mailbox_id,
            lease_claims_commitment: self.lease_claims_commitment,
            invitation_expires_at: self.expires_at,
            lease_expires_at: lease.expires_at,
            min_remaining_lease_secs: self.min_remaining_lease_secs,
            recipient_seal: self.recipient_seal,
        })
    }
}

impl fmt::Debug for AnonymousMailboxDepositInvitationV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxDepositInvitationV2")
            .field("version", &self.version)
            .field("encoded_bytes", &self.encoded_len())
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .field("min_remaining_lease_secs", &self.min_remaining_lease_secs)
            .finish_non_exhaustive()
    }
}

impl Drop for AnonymousMailboxDepositInvitationV2 {
    fn drop(&mut self) {
        self.deposit_seed.zeroize();
    }
}

fn validate_invitation_window(
    issued_at: u64,
    expires_at: u64,
    min_remaining_lease_secs: u32,
    now: u64,
) -> Result<(), AnonymousMailboxDepositInvitationError> {
    let ttl = expires_at
        .checked_sub(issued_at)
        .ok_or(AnonymousMailboxDepositInvitationError::Expired)?;
    if ttl == 0
        || ttl > MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_TTL_SECS
        || !(MIN_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS
            ..=MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS)
            .contains(&min_remaining_lease_secs)
        || now > expires_at
        || issued_at > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
    {
        return Err(AnonymousMailboxDepositInvitationError::Expired);
    }
    Ok(())
}

fn decode_accepted_lease(
    target: &AnonymousMailboxDepositTargetPinV1,
    lease_create_frame: &[u8],
    lease_response_frame: &[u8],
) -> Result<(AnonymousMailboxLeaseCreateV1, u64), AnonymousMailboxDepositInvitationError> {
    if lease_create_frame.len() > MAX_LEASE_CREATE_FRAME_BYTES
        || lease_response_frame.len() > MAX_LEASE_RESPONSE_FRAME_BYTES
    {
        return Err(AnonymousMailboxDepositInvitationError::TooLarge);
    }
    let AnonymousMailboxTerminalFrameV1::LeaseCreate(lease) =
        decode_anonymous_mailbox_terminal_frame(lease_create_frame)?
    else {
        return Err(AnonymousMailboxDepositInvitationError::Malformed);
    };
    let AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response) =
        decode_anonymous_mailbox_terminal_frame(lease_response_frame)?
    else {
        return Err(AnonymousMailboxDepositInvitationError::Malformed);
    };
    if encode_anonymous_mailbox_terminal_frame(&AnonymousMailboxTerminalFrameV1::LeaseCreate(
        lease.clone(),
    ))? != lease_create_frame
        || encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response.clone()),
        )? != lease_response_frame
    {
        return Err(AnonymousMailboxDepositInvitationError::Malformed);
    }
    verify_historical_lease_acceptance(target, &lease, &response)?;
    Ok((lease, response.responded_at))
}

fn verify_historical_lease_acceptance(
    target: &AnonymousMailboxDepositTargetPinV1,
    lease: &AnonymousMailboxLeaseCreateV1,
    response: &AnonymousMailboxTerminalResponseV1,
) -> Result<(), AnonymousMailboxDepositInvitationError> {
    if lease.mailbox_id.iter().all(|byte| *byte == 0)
        || lease.admission.ticket_id.iter().all(|byte| *byte == 0)
        || lease.admission.target_node_id != target.node_id
        || lease.admission.lease_claims_commitment != lease.claims_commitment()
    {
        return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
    }
    let target_key = IdentityPublicKey::from_bytes(&target.node_id)
        .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
    target_key
        .verify(
            &lease.admission.signing_bytes()?,
            &lease.admission.signature,
        )
        .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
    let lease_signing_bytes = lease.signing_bytes()?;
    IdentityPublicKey::from_bytes(&lease.read_verifier)
        .and_then(|key| key.verify(&lease_signing_bytes, &lease.signature))
        .map_err(|_| AnonymousMailboxDepositInvitationError::SignatureRejected)?;
    if IdentityPublicKey::from_bytes(&lease.deposit_verifier).is_err() {
        return Err(AnonymousMailboxDepositInvitationError::SignatureRejected);
    }

    let ticket_ttl = lease
        .admission
        .expires_at
        .checked_sub(lease.admission.issued_at)
        .ok_or(AnonymousMailboxDepositInvitationError::Expired)?;
    if ticket_ttl == 0 || ticket_ttl > MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS {
        return Err(AnonymousMailboxDepositInvitationError::Expired);
    }
    // A target may freshly re-attest a durable exact Existing lease after the
    // one-use admission ticket expires. `responded_at` is therefore evidence
    // time, not necessarily the original row-creation time.
    let responded_at = response.responded_at;
    if lease.admission.issued_at
        > responded_at.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
        || lease.issued_at > responded_at.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
        || responded_at > lease.expires_at
    {
        return Err(AnonymousMailboxDepositInvitationError::Expired);
    }
    if response.outcome != AnonymousMailboxOutcomeV1::Accepted
        || !response.sealed_payload.is_empty()
    {
        return Err(AnonymousMailboxDepositInvitationError::ClaimsConflict);
    }
    response
        .verify_for_request(
            AnonymousMailboxOperationV1::LeaseCreate,
            &lease.admission.ticket_id,
            &lease.request_commitment()?,
            &target.node_id,
        )
        .map_err(Into::into)
}

fn take<const N: usize>(
    encoded: &[u8],
    offset: &mut usize,
) -> Result<[u8; N], AnonymousMailboxDepositInvitationError> {
    let end = offset
        .checked_add(N)
        .ok_or(AnonymousMailboxDepositInvitationError::TooLarge)?;
    let value = encoded
        .get(*offset..end)
        .ok_or(AnonymousMailboxDepositInvitationError::Malformed)?
        .try_into()
        .map_err(|_| AnonymousMailboxDepositInvitationError::Malformed)?;
    *offset = end;
    Ok(value)
}

fn take_vec(
    encoded: &[u8],
    offset: &mut usize,
    length: usize,
) -> Result<Vec<u8>, AnonymousMailboxDepositInvitationError> {
    let end = offset
        .checked_add(length)
        .ok_or(AnonymousMailboxDepositInvitationError::TooLarge)?;
    let value = encoded
        .get(*offset..end)
        .ok_or(AnonymousMailboxDepositInvitationError::Malformed)?
        .to_vec();
    *offset = end;
    Ok(value)
}

fn take_u16(
    encoded: &[u8],
    offset: &mut usize,
) -> Result<u16, AnonymousMailboxDepositInvitationError> {
    Ok(u16::from_le_bytes(take(encoded, offset)?))
}

fn take_u32(
    encoded: &[u8],
    offset: &mut usize,
) -> Result<u32, AnonymousMailboxDepositInvitationError> {
    Ok(u32::from_le_bytes(take(encoded, offset)?))
}

fn take_u64(
    encoded: &[u8],
    offset: &mut usize,
) -> Result<u64, AnonymousMailboxDepositInvitationError> {
    Ok(u64::from_le_bytes(take(encoded, offset)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::anonymous_mailbox::{
        decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
        AnonymousMailboxAdmissionTicketV1, AnonymousMailboxTerminalFrameV1,
    };

    const LEASE_ISSUED: u64 = 1_800_000_000;
    const LEASE_EXPIRES: u64 = LEASE_ISSUED + (10 * 24 * 60 * 60);
    const TICKET_EXPIRES: u64 = LEASE_ISSUED + MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS;
    const ACCEPTED_AT: u64 = LEASE_ISSUED + 10;
    const INVITATION_ISSUED: u64 = TICKET_EXPIRES + 1;
    const INVITATION_EXPIRES: u64 = INVITATION_ISSUED + (7 * 24 * 60 * 60);

    struct Fixture {
        target: IdentityKeyPair,
        deposit: IdentityKeyPair,
        reader: IdentityKeyPair,
        pin: AnonymousMailboxDepositTargetPinV1,
        lease_frame: Vec<u8>,
        response_frame: Vec<u8>,
    }

    fn fixture() -> Fixture {
        let target = IdentityKeyPair::from_bytes(&[0x11; 32]).expect("target");
        let deposit = IdentityKeyPair::from_bytes(&[0x22; 32]).expect("deposit");
        let reader = IdentityKeyPair::from_bytes(&[0x33; 32]).expect("reader");
        let mailbox_id = [0x44; 32];
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &mailbox_id,
            &deposit.public_key_bytes(),
            &reader.public_key_bytes(),
            64,
            4 * 1024 * 1024,
            LEASE_ISSUED,
            LEASE_EXPIRES,
        );
        let ticket = AnonymousMailboxAdmissionTicketV1::issue(
            [0x55; 16],
            claims,
            LEASE_ISSUED,
            TICKET_EXPIRES,
            &target,
        )
        .expect("ticket");
        let lease = AnonymousMailboxLeaseCreateV1::new(
            mailbox_id,
            deposit.public_key_bytes(),
            64,
            4 * 1024 * 1024,
            LEASE_ISSUED,
            LEASE_EXPIRES,
            ticket,
            &reader,
        )
        .expect("lease");
        let response = AnonymousMailboxTerminalResponseV1::signed(
            AnonymousMailboxOperationV1::LeaseCreate,
            lease.admission.ticket_id,
            lease.request_commitment().expect("commitment"),
            AnonymousMailboxOutcomeV1::Accepted,
            Vec::new(),
            ACCEPTED_AT,
            &target,
        )
        .expect("response");
        let pin =
            AnonymousMailboxDepositTargetPinV1::new(target.public_key_bytes(), 42, [0x66; 32])
                .expect("pin");
        Fixture {
            target,
            deposit,
            reader,
            pin,
            lease_frame: encode_anonymous_mailbox_terminal_frame(
                &AnonymousMailboxTerminalFrameV1::LeaseCreate(lease),
            )
            .expect("lease frame"),
            response_frame: encode_anonymous_mailbox_terminal_frame(
                &AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response),
            )
            .expect("response frame"),
        }
    }

    fn invitation() -> AnonymousMailboxDepositInvitationV1 {
        let fixture = fixture();
        AnonymousMailboxDepositInvitationV1::issue(
            [0x77; 16],
            INVITATION_ISSUED,
            INVITATION_EXPIRES,
            DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
            fixture.pin,
            fixture.deposit.to_bytes(),
            fixture.lease_frame,
            fixture.response_frame,
            &fixture.reader,
        )
        .expect("invitation")
    }

    #[test]
    fn historical_accepted_lease_survives_ticket_expiry_and_prepares_put_only() {
        let invitation = invitation();
        let summary = invitation
            .verify_at(INVITATION_ISSUED)
            .expect("historical proof");
        let put = invitation
            .prepare_put(
                [0x88; 16],
                b"opaque".to_vec(),
                INVITATION_ISSUED,
                INVITATION_ISSUED + 60,
                INVITATION_ISSUED,
            )
            .expect("deposit-only Put");
        let lease = match decode_anonymous_mailbox_terminal_frame(&invitation.lease_create_frame)
            .expect("lease")
        {
            AnonymousMailboxTerminalFrameV1::LeaseCreate(value) => value,
            _ => panic!("lease kind"),
        };
        put.verify_at(&lease.deposit_verifier, INVITATION_ISSUED)
            .expect("deposit verifier");
        assert_eq!(put.mailbox_id, summary.mailbox_id());
    }

    #[test]
    fn codec_rejects_length_version_trailing_and_oversize() {
        let encoded = invitation().encode().expect("encode");
        assert!(AnonymousMailboxDepositInvitationV1::decode_at(
            &encoded[..encoded.len() - 1],
            INVITATION_ISSUED
        )
        .is_err());
        let mut unknown = encoded.clone();
        unknown[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            AnonymousMailboxDepositInvitationV1::decode_at(&unknown, INVITATION_ISSUED)
                .expect_err("unknown"),
            AnonymousMailboxDepositInvitationError::UnsupportedVersion
        );
        for declared in [encoded.len() - 1, encoded.len() + 1] {
            let mut wrong_length = encoded.clone();
            wrong_length[6..10].copy_from_slice(
                &u32::try_from(declared)
                    .expect("fixture length fits u32")
                    .to_le_bytes(),
            );
            assert_eq!(
                AnonymousMailboxDepositInvitationV1::decode_at(&wrong_length, INVITATION_ISSUED)
                    .expect_err("declared length"),
                AnonymousMailboxDepositInvitationError::Malformed
            );
        }
        let mut trailing = encoded;
        trailing.push(0);
        assert_eq!(
            AnonymousMailboxDepositInvitationV1::decode_at(&trailing, INVITATION_ISSUED)
                .expect_err("trailing"),
            AnonymousMailboxDepositInvitationError::Malformed
        );
        assert_eq!(
            AnonymousMailboxDepositInvitationV1::decode_at(
                &vec![0; MAX_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_BYTES + 1],
                INVITATION_ISSUED,
            )
            .expect_err("oversize"),
            AnonymousMailboxDepositInvitationError::TooLarge
        );
    }

    #[test]
    fn signed_fields_seed_verifier_and_target_response_fail_closed() {
        let encoded = invitation().encode().expect("encode");
        for offset in [
            10usize, 26, 34, 42, 46, 78, 86, 118, 150, 182, 184, 240, 540, 542, 600, 719,
        ] {
            let mut changed = encoded.clone();
            changed[offset] ^= 1;
            assert!(
                AnonymousMailboxDepositInvitationV1::decode_at(&changed, INVITATION_ISSUED)
                    .is_err()
            );
        }

        let fixture = fixture();
        assert_eq!(
            AnonymousMailboxDepositInvitationV1::issue(
                [0x77; 16],
                INVITATION_ISSUED,
                INVITATION_EXPIRES,
                DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
                fixture.pin,
                [0x99; 32],
                fixture.lease_frame.clone(),
                fixture.response_frame.clone(),
                &fixture.reader,
            )
            .expect_err("seed mismatch"),
            AnonymousMailboxDepositInvitationError::ClaimsConflict
        );

        let other_target = IdentityKeyPair::from_bytes(&[0xaa; 32]).expect("other");
        let wrong_pin = AnonymousMailboxDepositTargetPinV1::new(
            other_target.public_key_bytes(),
            42,
            [0x66; 32],
        )
        .expect("pin");
        assert!(AnonymousMailboxDepositInvitationV1::issue(
            [0x77; 16],
            INVITATION_ISSUED,
            INVITATION_EXPIRES,
            DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
            wrong_pin,
            fixture.deposit.to_bytes(),
            fixture.lease_frame,
            fixture.response_frame,
            &fixture.reader,
        )
        .is_err());
    }

    #[test]
    fn response_operation_outcome_payload_signature_and_time_are_bound() {
        let fixture = fixture();
        let lease =
            match decode_anonymous_mailbox_terminal_frame(&fixture.lease_frame).expect("lease") {
                AnonymousMailboxTerminalFrameV1::LeaseCreate(value) => value,
                _ => panic!("lease kind"),
            };
        let cases = [
            (
                AnonymousMailboxOperationV1::Put,
                AnonymousMailboxOutcomeV1::Accepted,
                Vec::new(),
                ACCEPTED_AT,
            ),
            (
                AnonymousMailboxOperationV1::LeaseCreate,
                AnonymousMailboxOutcomeV1::Rejected,
                Vec::new(),
                ACCEPTED_AT,
            ),
            (
                AnonymousMailboxOperationV1::LeaseCreate,
                AnonymousMailboxOutcomeV1::Accepted,
                vec![1],
                ACCEPTED_AT,
            ),
            (
                AnonymousMailboxOperationV1::LeaseCreate,
                AnonymousMailboxOutcomeV1::Accepted,
                Vec::new(),
                LEASE_EXPIRES + 1,
            ),
        ];
        for (operation, outcome, payload, responded_at) in cases {
            let response = AnonymousMailboxTerminalResponseV1::signed(
                operation,
                lease.admission.ticket_id,
                lease.request_commitment().expect("commitment"),
                outcome,
                payload,
                responded_at,
                &fixture.target,
            )
            .expect("response");
            let response = encode_anonymous_mailbox_terminal_frame(
                &AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response),
            );
            if let Ok(response) = response {
                assert!(AnonymousMailboxDepositInvitationV1::issue(
                    [0x77; 16],
                    INVITATION_ISSUED,
                    INVITATION_EXPIRES,
                    DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
                    fixture.pin,
                    fixture.deposit.to_bytes(),
                    fixture.lease_frame.clone(),
                    response,
                    &fixture.reader,
                )
                .is_err());
            }
        }

        let wrong_target = IdentityKeyPair::from_bytes(&[0xab; 32]).expect("wrong target");
        let wrong_responses = [
            AnonymousMailboxTerminalResponseV1::signed(
                AnonymousMailboxOperationV1::LeaseCreate,
                [0xac; 16],
                lease.request_commitment().expect("commitment"),
                AnonymousMailboxOutcomeV1::Accepted,
                Vec::new(),
                ACCEPTED_AT,
                &fixture.target,
            )
            .expect("wrong id"),
            AnonymousMailboxTerminalResponseV1::signed(
                AnonymousMailboxOperationV1::LeaseCreate,
                lease.admission.ticket_id,
                [0xad; 32],
                AnonymousMailboxOutcomeV1::Accepted,
                Vec::new(),
                ACCEPTED_AT,
                &fixture.target,
            )
            .expect("wrong commitment"),
            AnonymousMailboxTerminalResponseV1::signed(
                AnonymousMailboxOperationV1::LeaseCreate,
                lease.admission.ticket_id,
                lease.request_commitment().expect("commitment"),
                AnonymousMailboxOutcomeV1::Accepted,
                Vec::new(),
                ACCEPTED_AT,
                &wrong_target,
            )
            .expect("wrong responder"),
        ];
        for response in wrong_responses {
            let response = encode_anonymous_mailbox_terminal_frame(
                &AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response),
            )
            .expect("response frame");
            assert!(AnonymousMailboxDepositInvitationV1::issue(
                [0x77; 16],
                INVITATION_ISSUED,
                INVITATION_EXPIRES,
                DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
                fixture.pin,
                fixture.deposit.to_bytes(),
                fixture.lease_frame.clone(),
                response,
                &fixture.reader,
            )
            .is_err());
        }

        let mut bad_signature = fixture.response_frame;
        let last = bad_signature.len() - 1;
        bad_signature[last] ^= 1;
        assert!(AnonymousMailboxDepositInvitationV1::issue(
            [0x77; 16],
            INVITATION_ISSUED,
            INVITATION_EXPIRES,
            DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
            fixture.pin,
            fixture.deposit.to_bytes(),
            fixture.lease_frame,
            bad_signature,
            &fixture.reader,
        )
        .is_err());
    }

    #[test]
    fn invitation_and_runway_expiry_are_client_policy_only() {
        let invitation = invitation();
        assert_eq!(
            invitation.verify_at(INVITATION_EXPIRES + 1),
            Err(AnonymousMailboxDepositInvitationError::Expired)
        );
        assert!(matches!(
            invitation.prepare_put(
                [0x90; 16],
                b"opaque".to_vec(),
                INVITATION_ISSUED,
                INVITATION_ISSUED + 60,
                INVITATION_ISSUED + MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS + 1,
            ),
            Err(AnonymousMailboxDepositInvitationError::Expired)
        ));
        let fixture = fixture();
        assert_eq!(
            AnonymousMailboxDepositInvitationV1::issue(
                [0x77; 16],
                INVITATION_ISSUED,
                LEASE_EXPIRES,
                DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
                fixture.pin,
                fixture.deposit.to_bytes(),
                fixture.lease_frame,
                fixture.response_frame,
                &fixture.reader,
            )
            .expect_err("no runway"),
            AnonymousMailboxDepositInvitationError::Expired
        );
    }

    #[test]
    fn debug_output_is_redacted() {
        let invitation = invitation();
        let debug = format!("{invitation:?}");
        assert!(!debug.contains(&hex::encode([0x22; 32])));
        assert!(!debug.contains(&hex::encode([0x44; 32])));
        assert!(!debug.contains(&hex::encode([0x66; 32])));
        assert!(!debug.contains("lease_create_frame"));
        assert!(!debug.contains("signature"));
    }

    #[test]
    fn golden_wire_and_offsets_are_frozen() {
        let invitation = invitation();
        let encoded = invitation.encode().expect("encode");
        assert_eq!(encoded.len(), 783);
        assert_eq!(&encoded[0..4], b"AMDI");
        assert_eq!(&encoded[4..6], &1u16.to_le_bytes());
        assert_eq!(&encoded[6..10], &783u32.to_le_bytes());
        assert_eq!(&encoded[10..26], &[0x77; 16]);
        assert_eq!(&encoded[26..34], &INVITATION_ISSUED.to_le_bytes());
        assert_eq!(&encoded[34..42], &INVITATION_EXPIRES.to_le_bytes());
        assert_eq!(
            &encoded[42..46],
            &DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS.to_le_bytes()
        );
        assert_eq!(&encoded[150..182], &[0x22; 32]);
        assert_eq!(
            u16::from_le_bytes(encoded[182..184].try_into().unwrap()),
            356
        );
        assert_eq!(
            u16::from_le_bytes(encoded[540..542].try_into().unwrap()),
            177
        );
        assert_eq!(
            hex::encode(Sha256::digest(&encoded)),
            "af36586386fd09abb5b8924df4f59a30e75c44b4c248e17038c2df3652c4c524"
        );
        assert_eq!(
            hex::encode(&encoded),
            "414d444901000f030000777777777777777777777777777777772dd3496b00000000ad0d536b0000000078000000d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c97787372a0000000000000066666666666666666666666666666666666666666666666666666666666666663c11c4daf854f8ed57a760f84253f66928a47698e032a82faafbb48db822bfe822222222222222222222222222222222222222222222222222222222222222226401414d01015c010000014444444444444444444444444444444444444444444444444444444444444444a09aa5f47a6759802ff955f8dc2d2a14a5c99d23be97f864127ff9383455a4f017cb79fb2b4120f2b1ec65e4198d6e08b28e813feb01e4a400839b85e18080ce4000000040000000000000d2496b000000000001576b000000000155555555555555555555555555555555d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c97787373c11c4daf854f8ed57a760f84253f66928a47698e032a82faafbb48db822bfe800d2496b000000002cd3496b00000000d8f6c3792f497972956298c2d582254357548155268ece4339e5ad8f687bcc49d1245383b1f9e2f7d9e049446885bf7a458817bd58928a98b5e8ccf4e8481f04366675f5d0d3d57d0e55527114e7ddcc53632b5e5cc2741d3c96f1d4cd915bf8610d9130a4bd9ab135045997563fa00d5cb70f6677a3419a7bef693ed8724707b100414d0181a90000000100000000555555555555555555555555555555557bfeec48297740aff27cc2595ee56a261bb03ee0548f89ccba6a5f4f3874b0870000000000000000000000000ad2496b00000000d04ab232742bb4ab3a1368bd4615e4e6d0224ab71a016baf8520a332c9778737707cc91999bbdcfc7cb6b0d24963d2e770a2868309a5a37da3902bbf8bb4c22d6fed853c64f18f96544181e7b0d1871c9e1152e88d1f1f523fd0ee7b7fb0620058b601a968755a3f094c2a821ebf438f952922afafe38169f4c0886a7e2b181efcfe635d1249ac50f8eded7c175e772ba9a1222d8bbcfd3ea0345c08d8549907"
        );
    }

    struct RecipientKeyHandle {
        key_id: [u8; 16],
        secret: x25519_dalek::StaticSecret,
        public: [u8; 32],
        available: bool,
    }

    impl RecipientKeyHandle {
        fn new(key_id: [u8; 16], secret_bytes: [u8; 32]) -> Self {
            let secret = x25519_dalek::StaticSecret::from(secret_bytes);
            let public = x25519_dalek::PublicKey::from(&secret).to_bytes();
            Self {
                key_id,
                secret,
                public,
                available: true,
            }
        }
    }

    impl crate::protocol::anonymous_mailbox_recipient_seal::AnonymousMailboxRecipientSealKeyHandleV1
        for RecipientKeyHandle
    {
        fn key_id(&self) -> [u8; 16] {
            self.key_id
        }

        fn public_key(&self) -> [u8; 32] {
            self.public
        }

        fn derive_shared_secret(
            &self,
            peer_public_key: [u8; 32],
        ) -> Result<
            Zeroizing<[u8; 32]>,
            crate::protocol::anonymous_mailbox_recipient_seal::AnonymousMailboxRecipientSealError,
        > {
            if !self.available {
                return Err(crate::protocol::anonymous_mailbox_recipient_seal::AnonymousMailboxRecipientSealError::KeyUnavailable);
            }
            Ok(Zeroizing::new(
                *self
                    .secret
                    .diffie_hellman(&x25519_dalek::PublicKey::from(peer_public_key))
                    .as_bytes(),
            ))
        }
    }

    fn invitation_v2_with_id(
        invitation_id: [u8; 16],
    ) -> (AnonymousMailboxDepositInvitationV2, [u8; 32], [u8; 32]) {
        let fixture = fixture();
        let secret_bytes = [0xc1; 32];
        let handle = RecipientKeyHandle::new([0xc2; 16], secret_bytes);
        let chat_receiver = IdentityKeyPair::from_bytes(&[0xd1; 32]).expect("chat receiver");
        let recipient = AnonymousMailboxRecipientSealPublicV1::new(handle.key_id, handle.public)
            .expect("recipient seal");
        (
            AnonymousMailboxDepositInvitationV2::issue(
                invitation_id,
                INVITATION_ISSUED,
                INVITATION_EXPIRES,
                DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS,
                fixture.pin,
                fixture.deposit.to_bytes(),
                chat_receiver.public_key_bytes(),
                recipient,
                fixture.lease_frame,
                fixture.response_frame,
                &fixture.reader,
            )
            .expect("v2 invitation"),
            secret_bytes,
            chat_receiver.public_key_bytes(),
        )
    }

    fn invitation_v2() -> (AnonymousMailboxDepositInvitationV2, [u8; 32], [u8; 32]) {
        invitation_v2_with_id([0xc3; 16])
    }

    fn signed_chat(receiver: [u8; 32]) -> ChatEnvelope {
        let sender = IdentityKeyPair::from_bytes(&[0xc4; 32]).expect("sender");
        let mut envelope = ChatEnvelope {
            message_id: [0xc5; 16],
            sender: sender.public_key_bytes(),
            receiver,
            timestamp: INVITATION_ISSUED,
            ciphertext: b"opaque inner chat ciphertext".to_vec(),
            nonce: [0xc6; 24],
            content_type: crate::protocol::chat::ChatContentType::Text,
            signature: [0; 64],
        };
        envelope.signature = sender.sign(&envelope.sign_data());
        envelope
    }

    #[test]
    fn v2_active_put_and_expired_historical_restart_open_are_separate() {
        let (invitation, secret_bytes, chat_receiver) = invitation_v2();
        let encoded = invitation.encode().expect("encode v2");
        let active = invitation
            .verify_for_new_put_at(INVITATION_ISSUED)
            .expect("active");
        let summary = active.summary();
        assert_ne!(fixture().reader.public_key_bytes(), chat_receiver);
        assert_ne!(
            RecipientKeyHandle::new([0xc2; 16], secret_bytes).public,
            chat_receiver
        );
        let envelope = signed_chat(chat_receiver);
        let sealed = active
            .seal_chat_envelope_at(&envelope, INVITATION_ISSUED)
            .expect("seal");
        let sealed_bytes = sealed.as_bytes().to_vec();
        let put = active
            .prepare_put(
                [0xc7; 16],
                sealed,
                INVITATION_ISSUED,
                INVITATION_ISSUED + 60,
                INVITATION_ISSUED,
            )
            .expect("put");
        assert_eq!(put.sealed_envelope, sealed_bytes);
        assert_eq!(put.mailbox_id, summary.mailbox_id());

        // The target-visible terminal projection is structurally still the
        // legacy Put: no chat receiver was added to its decoded field set.
        let terminal = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::Put(put.clone()),
        )
        .expect("terminal");
        let AnonymousMailboxTerminalFrameV1::Put(target_visible) =
            decode_anonymous_mailbox_terminal_frame(&terminal).expect("target-visible Put")
        else {
            panic!("expected Put");
        };
        assert_eq!(target_visible.mailbox_id, summary.mailbox_id());
        assert_eq!(target_visible.item_id, [0xc7; 16]);
        assert_eq!(target_visible.sealed_envelope, sealed_bytes);

        assert_eq!(
            AnonymousMailboxDepositInvitationV2::decode_at(&encoded, INVITATION_EXPIRES + 1)
                .expect_err("expired new effect"),
            AnonymousMailboxDepositInvitationError::Expired
        );
        assert!(active
            .seal_chat_envelope_at(&envelope, INVITATION_EXPIRES + 1)
            .is_err());

        let expired_put_item = active
            .seal_chat_envelope_at(&envelope, INVITATION_ISSUED)
            .expect("seal before expiry");
        assert!(matches!(
            active.prepare_put(
                [0xc8; 16],
                expired_put_item,
                INVITATION_ISSUED,
                INVITATION_ISSUED + 60,
                INVITATION_EXPIRES + 1,
            ),
            Err(AnonymousMailboxDepositInvitationError::Expired)
        ));

        let wrong_receiver = signed_chat(fixture().reader.public_key_bytes());
        assert_eq!(
            active
                .seal_chat_envelope_at(&wrong_receiver, INVITATION_ISSUED)
                .expect_err("mailbox reader is not the chat receiver"),
            AnonymousMailboxRecipientSealError::ClaimsConflict
        );

        // Simulate a process restart: reconstruct both the authenticated
        // historical context and the native key handle from durable material.
        let historical =
            AnonymousMailboxDepositInvitationV2::decode_historical_recipient_context(&encoded)
                .expect("historical context");
        let handle = RecipientKeyHandle::new([0xc2; 16], secret_bytes);
        let opened = historical
            .open_chat_envelope(&handle, &sealed_bytes)
            .expect("open retained item after invitation expiry");
        assert_eq!(opened.message_id, envelope.message_id);
        assert_eq!(historical.lease_expires_at(), LEASE_EXPIRES);

        let unavailable = RecipientKeyHandle {
            available: false,
            ..RecipientKeyHandle::new([0xc2; 16], secret_bytes)
        };
        assert_eq!(
            historical
                .open_chat_envelope(&unavailable, &sealed_bytes)
                .expect_err("missing durable key"),
            AnonymousMailboxRecipientSealError::KeyUnavailable
        );
    }

    #[test]
    fn decoded_v2_active_exposes_only_the_exact_redacted_summary() {
        let (invitation, _, _) = invitation_v2();
        let encoded = invitation.encode().expect("encode v2");
        let expected_recipient = invitation.recipient_seal;
        let expected_lease_commitment = invitation.lease_claims_commitment;

        let decoded = AnonymousMailboxDepositInvitationV2::decode_at(&encoded, INVITATION_ISSUED)
            .expect("decode admitted v2");
        let active = decoded
            .verify_for_new_put_at(INVITATION_ISSUED)
            .expect("active v2");
        let summary = active.summary();

        assert_eq!(summary.invitation_id(), [0xc3; 16]);
        assert_eq!(summary.target(), fixture().pin);
        assert_eq!(summary.mailbox_id(), [0x44; 32]);
        assert_eq!(summary.lease_claims_commitment(), expected_lease_commitment);
        assert_eq!(summary.invitation_expires_at(), INVITATION_EXPIRES);
        assert_eq!(summary.lease_expires_at(), LEASE_EXPIRES);
        assert_eq!(
            summary.min_remaining_lease_secs(),
            DEFAULT_ANONYMOUS_MAILBOX_DEPOSIT_INVITATION_RUNWAY_SECS
        );
        assert_eq!(summary.recipient_seal(), expected_recipient);

        assert!(matches!(
            decoded.verify_for_new_put_at(INVITATION_EXPIRES + 1),
            Err(AnonymousMailboxDepositInvitationError::Expired)
        ));
        let historical =
            AnonymousMailboxDepositInvitationV2::decode_historical_recipient_context(&encoded)
                .expect("historical open-only context");
        assert_eq!(historical.lease_expires_at(), LEASE_EXPIRES);

        let debug = format!("{summary:?}");
        assert!(debug.contains("invitation_expires_at"));
        for secret_or_identifier in [
            "invitation_id",
            "target",
            "mailbox_id",
            "lease_claims_commitment",
            "recipient_seal",
            "deposit_seed",
            "private_key",
            "read_verifier",
        ] {
            assert!(
                !debug.contains(secret_or_identifier),
                "redacted summary Debug leaked {secret_or_identifier}"
            );
        }
    }

    #[test]
    fn typed_sealed_item_cannot_cross_invitation_bindings() {
        let (invitation_a, _, chat_receiver) = invitation_v2_with_id([0xc3; 16]);
        let (invitation_b, _, _) = invitation_v2_with_id([0xd3; 16]);
        let active_a = invitation_a
            .verify_for_new_put_at(INVITATION_ISSUED)
            .expect("active A");
        let active_b = invitation_b
            .verify_for_new_put_at(INVITATION_ISSUED)
            .expect("active B");
        let envelope = signed_chat(chat_receiver);

        let wrong_origin = active_a
            .seal_chat_envelope_at(&envelope, INVITATION_ISSUED)
            .expect("A seal");
        assert!(matches!(
            active_b.prepare_put(
                [0xd4; 16],
                wrong_origin,
                INVITATION_ISSUED,
                INVITATION_ISSUED + 60,
                INVITATION_ISSUED,
            ),
            Err(AnonymousMailboxDepositInvitationError::ClaimsConflict)
        ));

        let same_origin = active_a
            .seal_chat_envelope_at(&envelope, INVITATION_ISSUED)
            .expect("second A seal");
        assert!(active_a
            .prepare_put(
                [0xd5; 16],
                same_origin,
                INVITATION_ISSUED,
                INVITATION_ISSUED + 60,
                INVITATION_ISSUED,
            )
            .is_ok());
    }

    #[test]
    fn v2_wire_domains_and_recipient_fields_are_frozen() {
        let (invitation, _, chat_receiver) = invitation_v2();
        let encoded = invitation.encode().expect("encode");
        assert_eq!(encoded.len(), 864);
        assert_eq!(&encoded[..4], b"AMDI");
        assert_eq!(&encoded[4..6], &2u16.to_le_bytes());
        assert_eq!(&encoded[6..10], &864u32.to_le_bytes());
        assert_eq!(&encoded[182..214], &chat_receiver);
        assert_eq!(encoded[214], 1);
        assert_eq!(&encoded[215..231], &[0xc2; 16]);
        assert_eq!(
            u16::from_le_bytes(encoded[263..265].try_into().expect("lease length")),
            356
        );
        assert_eq!(
            hex::encode(Sha256::digest(&encoded)),
            "1146ce09af689a77e2b7a3143707ecb76fe86d2e1c7b90cfef0267fe7a86de10"
        );
        assert_eq!(
            hex::encode(invitation.commitment().expect("commitment")),
            "4ea25de63bef0be8e4ec4763231fc4ac138a3b43c3603718464f5df0ffe14e55"
        );

        let mut receiver_tamper = encoded.clone();
        receiver_tamper[182] ^= 1;
        assert_eq!(
            AnonymousMailboxDepositInvitationV2::decode_at(&receiver_tamper, INVITATION_ISSUED,)
                .expect_err("chat receiver is signed"),
            AnonymousMailboxDepositInvitationError::SignatureRejected
        );

        for offset in [
            4usize, 10, 46, 86, 118, 150, 182, 214, 215, 231, 263, 621, 800,
        ] {
            let mut changed = encoded.clone();
            changed[offset] ^= 1;
            assert!(
                AnonymousMailboxDepositInvitationV2::decode_at(&changed, INVITATION_ISSUED)
                    .is_err()
            );
        }
        let mut trailing = encoded;
        trailing.push(0);
        assert!(
            AnonymousMailboxDepositInvitationV2::decode_at(&trailing, INVITATION_ISSUED).is_err()
        );
    }
}
