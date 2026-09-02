// ============================================
// File: crates/aeronyx-core/src/protocol/anonymous_mailbox.rs
// ============================================
//! Anonymous mailbox v1 core protocol.
//!
//! This module defines only bounded, node-blind wire/domain primitives. A full
//! [`crate::protocol::chat::ChatEnvelope`] is sealed by the client and treated
//! here as opaque bytes; no outer type contains chat sender or receiver fields.
//! Storage, custody-set selection, routing, retries, and HTTP composition are
//! deliberately outside the core slice.
//!
//! [ANONYMOUS-MAILBOX-V1 2026-09-02 by Codex] All transcript domains, terminal
//! kinds, and limits below are public compatibility contracts. Unknown version,
//! kind, trailing bytes, malformed claims, and oversized input fail closed.

use std::fmt;

use serde::de::{DeserializeOwned, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

const TICKET_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-AdmissionTicket-v1";
const LEASE_CLAIMS_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-LeaseClaims-v1";
const LEASE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-LeaseCreate-v1";
const PUT_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-Put-v1";
const PULL_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-PullOne-v1";
const ACK_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-Ack-v1";
const RESPONSE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-Response-v1";
const ROUTE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-RouteRequest-v1";
const ROUTE_RESPONSE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-RouteResponse-v1";
const EXACT_REQUEST_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-ExactRequest-v1";

/// Initial anonymous mailbox protocol version.
pub const ANONYMOUS_MAILBOX_VERSION_V1: u8 = 1;
/// Maximum sealed ChatEnvelope bytes in one Put.
pub const MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES: usize = 160 * 1024;
/// Maximum complete terminal frame including its explicit header.
pub const MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES: usize = 176 * 1024;
/// Maximum sealed terminal bytes in a routed request or response.
///
/// The missing KiB below the legacy 192-KiB blob class reserves the exact
/// MemChain and three-layer onion overhead. `onion.rs` locks this with an
/// encoded-size fixture; legacy caps are not widened.
pub const MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES: usize = 191 * 1024;
/// Maximum opaque pull cursor.
pub const MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES: usize = 256;
/// Maximum admitted items per lease.
pub const MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE: u16 = 1_024;
/// Maximum admitted bytes per lease.
pub const MAX_ANONYMOUS_MAILBOX_BYTES_PER_LEASE: u64 = 128 * 1024 * 1024;
/// Maximum admission-ticket lifetime.
pub const MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS: u64 = 5 * 60;
/// Maximum lease lifetime.
pub const MAX_ANONYMOUS_MAILBOX_LEASE_TTL_SECS: u64 = 30 * 24 * 60 * 60;
/// Maximum sealed-item lifetime.
pub const MAX_ANONYMOUS_MAILBOX_ITEM_TTL_SECS: u64 = 7 * 24 * 60 * 60;
/// Maximum accepted future request skew.
pub const MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS: u64 = 120;

const MAGIC: [u8; 2] = [0x41, 0x4d];
const HEADER_BYTES: usize = 8;
const BODY_BYTES: u64 = (MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES - HEADER_BYTES) as u64;

/// Privacy-safe, coarse mailbox protocol failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AnonymousMailboxProtocolError {
    /// A complete frame or opaque member exceeds its fixed class.
    #[error("anonymous mailbox frame exceeds its protocol limit")]
    TooLarge,
    /// A frame is truncated, inconsistent, or non-canonical.
    #[error("anonymous mailbox frame is malformed")]
    Malformed,
    /// The version is not supported.
    #[error("anonymous mailbox version is unsupported")]
    UnsupportedVersion,
    /// The terminal operation kind is not supported.
    #[error("anonymous mailbox operation is unsupported")]
    UnsupportedOperation,
    /// A signed request is outside its accepted time window.
    #[error("anonymous mailbox request is outside its time window")]
    Expired,
    /// An exact capability or retry binding differs.
    #[error("anonymous mailbox claims do not match")]
    ClaimsConflict,
    /// A signing key or signature is invalid.
    #[error("anonymous mailbox signature was rejected")]
    SignatureRejected,
}

/// Frozen response operation identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AnonymousMailboxOperationV1 {
    /// Lease creation.
    LeaseCreate = 1,
    /// Sealed append.
    Put = 2,
    /// Pull at most one item.
    PullOne = 3,
    /// Acknowledge one item.
    Ack = 4,
}

impl AnonymousMailboxOperationV1 {
    const fn code(self) -> u8 {
        self as u8
    }
}

/// Coarse terminal outcome; sensitive failure detail remains source-sealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AnonymousMailboxOutcomeV1 {
    /// Exact request durably accepted.
    Accepted = 0,
    /// Authentication or policy rejected.
    Rejected = 1,
    /// Selected terminal temporarily unavailable.
    Unavailable = 2,
    /// Fixed lease or node capacity exhausted.
    AtCapacity = 3,
    /// Idempotency key reused for different claims.
    Conflict = 4,
    /// Ticket, lease, or item expired.
    Expired = 5,
}

impl AnonymousMailboxOutcomeV1 {
    const fn code(self) -> u8 {
        self as u8
    }
}

mod bytes64 {
    use super::*;

    pub fn serialize<S: Serializer>(value: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error> {
        let lo: [u8; 32] = value[..32].try_into().expect("32-byte half");
        let hi: [u8; 32] = value[32..].try_into().expect("32-byte half");
        (lo, hi).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 64], D::Error> {
        let (lo, hi): ([u8; 32], [u8; 32]) = Deserialize::deserialize(deserializer)?;
        let mut value = [0; 64];
        value[..32].copy_from_slice(&lo);
        value[32..].copy_from_slice(&hi);
        Ok(value)
    }
}

fn bounded_bytes<'de, D: Deserializer<'de>, const MAX: usize>(
    deserializer: D,
) -> Result<Vec<u8>, D::Error> {
    struct Bounded<const MAX: usize>;
    impl<'de, const MAX: usize> Visitor<'de> for Bounded<MAX> {
        type Value = Vec<u8>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "at most {MAX} opaque bytes")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Vec<u8>, A::Error> {
            let hint = sequence.size_hint().unwrap_or(0);
            if hint > MAX {
                return Err(serde::de::Error::invalid_length(hint, &self));
            }
            let mut value = Vec::with_capacity(hint.min(MAX));
            while let Some(byte) = sequence.next_element()? {
                if value.len() == MAX {
                    return Err(serde::de::Error::invalid_length(MAX + 1, &self));
                }
                value.push(byte);
            }
            Ok(value)
        }
    }
    deserializer.deserialize_seq(Bounded::<MAX>)
}

fn item_bytes<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    bounded_bytes::<D, MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES>(deserializer)
}
fn cursor_bytes<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    bounded_bytes::<D, MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES>(deserializer)
}
fn route_bytes<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    bounded_bytes::<D, MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES>(deserializer)
}

fn version(value: u8) -> Result<(), AnonymousMailboxProtocolError> {
    (value == ANONYMOUS_MAILBOX_VERSION_V1)
        .then_some(())
        .ok_or(AnonymousMailboxProtocolError::UnsupportedVersion)
}

fn window(
    issued: u64,
    expires: u64,
    now: u64,
    max_ttl: u64,
) -> Result<(), AnonymousMailboxProtocolError> {
    let ttl = expires
        .checked_sub(issued)
        .ok_or(AnonymousMailboxProtocolError::Expired)?;
    if ttl == 0
        || ttl > max_ttl
        || now > expires
        || issued > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
    {
        return Err(AnonymousMailboxProtocolError::Expired);
    }
    Ok(())
}

fn verify(
    key: &[u8; 32],
    bytes: &[u8],
    signature: &[u8; 64],
) -> Result<(), AnonymousMailboxProtocolError> {
    IdentityPublicKey::from_bytes(key)
        .and_then(|key| key.verify(bytes, signature))
        .map_err(|_| AnonymousMailboxProtocolError::SignatureRejected)
}

fn exact(bytes: &[u8], signature: &[u8; 64]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(EXACT_REQUEST_DOMAIN);
    hash.update((bytes.len() as u32).to_le_bytes());
    hash.update(bytes);
    hash.update(signature);
    hash.finalize().into()
}

fn opaque(
    data: &mut Vec<u8>,
    bytes: &[u8],
    max: usize,
) -> Result<(), AnonymousMailboxProtocolError> {
    if bytes.len() > max || bytes.len() > u32::MAX as usize {
        return Err(AnonymousMailboxProtocolError::TooLarge);
    }
    data.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    data.extend_from_slice(&Sha256::digest(bytes));
    Ok(())
}

/// Short-lived target-node authority over one exact lease claim set.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxAdmissionTicketV1 {
    /// Schema version.
    pub version: u8,
    /// Random idempotency identifier.
    pub ticket_id: [u8; 16],
    /// Exact issuing/serving node.
    pub target_node_id: [u8; 32],
    /// Commitment to every lease claim.
    pub lease_claims_commitment: [u8; 32],
    /// Issue time in Unix seconds.
    pub issued_at: u64,
    /// Expiry time in Unix seconds.
    pub expires_at: u64,
    /// Target-node Ed25519 signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxAdmissionTicketV1 {
    /// Issues one target-bound admission ticket.
    pub fn issue(
        ticket_id: [u8; 16],
        claims: [u8; 32],
        issued_at: u64,
        expires_at: u64,
        target: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        window(
            issued_at,
            expires_at,
            issued_at,
            MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS,
        )?;
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            ticket_id,
            target_node_id: target.public_key_bytes(),
            lease_claims_commitment: claims,
            issued_at,
            expires_at,
            signature: [0; 64],
        };
        value.signature = target.sign(&value.signing_bytes()?);
        Ok(value)
    }

    /// Returns the frozen ticket signing transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let mut data = Vec::with_capacity(TICKET_DOMAIN.len() + 97);
        data.extend_from_slice(TICKET_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.ticket_id);
        data.extend_from_slice(&self.target_node_id);
        data.extend_from_slice(&self.lease_claims_commitment);
        data.extend_from_slice(&self.issued_at.to_le_bytes());
        data.extend_from_slice(&self.expires_at.to_le_bytes());
        Ok(data)
    }

    /// Verifies time, target, claims, and target signature.
    pub fn verify_at(
        &self,
        target: &[u8; 32],
        claims: &[u8; 32],
        now: u64,
    ) -> Result<(), AnonymousMailboxProtocolError> {
        window(
            self.issued_at,
            self.expires_at,
            now,
            MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS,
        )?;
        if &self.target_node_id != target || &self.lease_claims_commitment != claims {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        verify(
            &self.target_node_id,
            &self.signing_bytes()?,
            &self.signature,
        )
    }

    /// Returns the exact signed ticket retry commitment.
    pub fn request_commitment(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        Ok(exact(&self.signing_bytes()?, &self.signature))
    }
}

/// Capability-key lease creation request.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxLeaseCreateV1 {
    /// Schema version.
    pub version: u8,
    /// Client-generated unlinkable mailbox id.
    pub mailbox_id: [u8; 32],
    /// Deposit capability public key.
    pub deposit_verifier: [u8; 32],
    /// Pull/ack capability public key.
    pub read_verifier: [u8; 32],
    /// Fixed item budget.
    pub max_items: u16,
    /// Fixed byte budget.
    pub max_bytes: u64,
    /// Issue time.
    pub issued_at: u64,
    /// Lease expiry.
    pub expires_at: u64,
    /// Target-issued exact-claims authority.
    pub admission: AnonymousMailboxAdmissionTicketV1,
    /// Read-capability signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxLeaseCreateV1 {
    /// Computes the exact claims commitment required before ticket issuance.
    #[must_use]
    pub fn lease_claims_commitment(
        mailbox_id: &[u8; 32],
        deposit: &[u8; 32],
        reader: &[u8; 32],
        max_items: u16,
        max_bytes: u64,
        issued_at: u64,
        expires_at: u64,
    ) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(LEASE_CLAIMS_DOMAIN);
        hash.update([ANONYMOUS_MAILBOX_VERSION_V1]);
        hash.update(mailbox_id);
        hash.update(deposit);
        hash.update(reader);
        hash.update(max_items.to_le_bytes());
        hash.update(max_bytes.to_le_bytes());
        hash.update(issued_at.to_le_bytes());
        hash.update(expires_at.to_le_bytes());
        hash.finalize().into()
    }

    /// Creates and signs a lease request after checking its admission binding.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mailbox_id: [u8; 32],
        deposit_verifier: [u8; 32],
        max_items: u16,
        max_bytes: u64,
        issued_at: u64,
        expires_at: u64,
        admission: AnonymousMailboxAdmissionTicketV1,
        reader: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            mailbox_id,
            deposit_verifier,
            read_verifier: reader.public_key_bytes(),
            max_items,
            max_bytes,
            issued_at,
            expires_at,
            admission,
            signature: [0; 64],
        };
        value.shape()?;
        if value.admission.lease_claims_commitment != value.claims_commitment() {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        value.signature = reader.sign(&value.signing_bytes()?);
        Ok(value)
    }

    /// Recomputes all lease claims.
    #[must_use]
    pub fn claims_commitment(&self) -> [u8; 32] {
        Self::lease_claims_commitment(
            &self.mailbox_id,
            &self.deposit_verifier,
            &self.read_verifier,
            self.max_items,
            self.max_bytes,
            self.issued_at,
            self.expires_at,
        )
    }

    /// Returns the domain-separated lease signature transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let mut data = Vec::with_capacity(LEASE_DOMAIN.len() + 65);
        data.extend_from_slice(LEASE_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.claims_commitment());
        data.extend_from_slice(&self.admission.request_commitment()?);
        Ok(data)
    }

    /// Returns the exact signed lease retry commitment.
    pub fn request_commitment(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        Ok(exact(&self.signing_bytes()?, &self.signature))
    }

    /// Verifies lease, ticket, target, time, and reader authorization.
    pub fn verify_for_target(
        &self,
        target: &[u8; 32],
        now: u64,
    ) -> Result<(), AnonymousMailboxProtocolError> {
        self.shape()?;
        window(
            self.issued_at,
            self.expires_at,
            now,
            MAX_ANONYMOUS_MAILBOX_LEASE_TTL_SECS,
        )?;
        self.admission
            .verify_at(target, &self.claims_commitment(), now)?;
        verify(&self.read_verifier, &self.signing_bytes()?, &self.signature)
    }

    fn shape(&self) -> Result<(), AnonymousMailboxProtocolError> {
        version(self.version)?;
        if self.max_items == 0
            || self.max_items > MAX_ANONYMOUS_MAILBOX_ITEMS_PER_LEASE
            || self.max_bytes == 0
            || self.max_bytes > MAX_ANONYMOUS_MAILBOX_BYTES_PER_LEASE
        {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        window(
            self.issued_at,
            self.expires_at,
            self.issued_at,
            MAX_ANONYMOUS_MAILBOX_LEASE_TTL_SECS,
        )
    }
}

/// Append-only request containing one complete sealed ChatEnvelope.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxPutV1 {
    /// Schema version.
    pub version: u8,
    /// Unlinkable mailbox id.
    pub mailbox_id: [u8; 32],
    /// Random idempotency id.
    pub item_id: [u8; 16],
    /// Complete opaque sealed ChatEnvelope.
    #[serde(deserialize_with = "item_bytes")]
    pub sealed_envelope: Vec<u8>,
    /// Issue time.
    pub issued_at: u64,
    /// Item expiry.
    pub expires_at: u64,
    /// Deposit-capability signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxPutV1 {
    /// Creates one signed immutable Put.
    pub fn new(
        mailbox_id: [u8; 32],
        item_id: [u8; 16],
        sealed_envelope: Vec<u8>,
        issued_at: u64,
        expires_at: u64,
        depositor: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            mailbox_id,
            item_id,
            sealed_envelope,
            issued_at,
            expires_at,
            signature: [0; 64],
        };
        value.shape(issued_at)?;
        value.signature = depositor.sign(&value.signing_bytes()?);
        Ok(value)
    }

    /// Returns the domain-separated Put signature transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let mut data = Vec::with_capacity(PUT_DOMAIN.len() + 101);
        data.extend_from_slice(PUT_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.mailbox_id);
        data.extend_from_slice(&self.item_id);
        data.extend_from_slice(&self.issued_at.to_le_bytes());
        data.extend_from_slice(&self.expires_at.to_le_bytes());
        opaque(
            &mut data,
            &self.sealed_envelope,
            MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES,
        )?;
        Ok(data)
    }

    /// SHA-256 commitment to the complete sealed envelope.
    #[must_use]
    pub fn sealed_commitment(&self) -> [u8; 32] {
        Sha256::digest(&self.sealed_envelope).into()
    }
    /// Exact signed Put retry commitment.
    pub fn request_commitment(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        Ok(exact(&self.signing_bytes()?, &self.signature))
    }
    /// Verifies size, time, and deposit capability.
    pub fn verify_at(
        &self,
        deposit: &[u8; 32],
        now: u64,
    ) -> Result<(), AnonymousMailboxProtocolError> {
        self.shape(now)?;
        verify(deposit, &self.signing_bytes()?, &self.signature)
    }
    fn shape(&self, now: u64) -> Result<(), AnonymousMailboxProtocolError> {
        version(self.version)?;
        if self.sealed_envelope.is_empty()
            || self.sealed_envelope.len() > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
        {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        window(
            self.issued_at,
            self.expires_at,
            now,
            MAX_ANONYMOUS_MAILBOX_ITEM_TTL_SECS,
        )
    }
}

/// Read-capability request for at most one sealed item.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxPullOneV1 {
    /// Schema version.
    pub version: u8,
    /// Unlinkable mailbox id.
    pub mailbox_id: [u8; 32],
    /// Random retry id.
    pub request_id: [u8; 16],
    /// Opaque bounded continuation cursor.
    #[serde(deserialize_with = "cursor_bytes")]
    pub cursor: Vec<u8>,
    /// Request time.
    pub requested_at: u64,
    /// Read-capability signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxPullOneV1 {
    /// Creates one signed pull request.
    pub fn new(
        mailbox_id: [u8; 32],
        request_id: [u8; 16],
        cursor: Vec<u8>,
        requested_at: u64,
        reader: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            mailbox_id,
            request_id,
            cursor,
            requested_at,
            signature: [0; 64],
        };
        value.signature = reader.sign(&value.signing_bytes()?);
        Ok(value)
    }
    /// Returns the domain-separated pull transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let mut data = Vec::with_capacity(PULL_DOMAIN.len() + 93);
        data.extend_from_slice(PULL_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.mailbox_id);
        data.extend_from_slice(&self.request_id);
        data.extend_from_slice(&self.requested_at.to_le_bytes());
        opaque(&mut data, &self.cursor, MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES)?;
        Ok(data)
    }
    /// Exact signed pull retry commitment.
    pub fn request_commitment(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        Ok(exact(&self.signing_bytes()?, &self.signature))
    }
    /// Verifies freshness and read capability.
    pub fn verify_at(
        &self,
        reader: &[u8; 32],
        now: u64,
    ) -> Result<(), AnonymousMailboxProtocolError> {
        fresh(self.requested_at, now)?;
        verify(reader, &self.signing_bytes()?, &self.signature)
    }
}

/// Read-capability acknowledgement of one exact sealed item commitment.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxAckV1 {
    /// Schema version.
    pub version: u8,
    /// Unlinkable mailbox id.
    pub mailbox_id: [u8; 32],
    /// Random retry id.
    pub request_id: [u8; 16],
    /// Exact item id returned by PullOne.
    pub item_id: [u8; 16],
    /// Commitment to the returned sealed bytes.
    pub sealed_commitment: [u8; 32],
    /// Acknowledgement time.
    pub acknowledged_at: u64,
    /// Read-capability signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxAckV1 {
    /// Creates one signed acknowledgement.
    pub fn new(
        mailbox_id: [u8; 32],
        request_id: [u8; 16],
        item_id: [u8; 16],
        sealed_commitment: [u8; 32],
        acknowledged_at: u64,
        reader: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            mailbox_id,
            request_id,
            item_id,
            sealed_commitment,
            acknowledged_at,
            signature: [0; 64],
        };
        value.signature = reader.sign(&value.signing_bytes()?);
        Ok(value)
    }
    /// Returns the domain-separated acknowledgement transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let mut data = Vec::with_capacity(ACK_DOMAIN.len() + 105);
        data.extend_from_slice(ACK_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.mailbox_id);
        data.extend_from_slice(&self.request_id);
        data.extend_from_slice(&self.item_id);
        data.extend_from_slice(&self.sealed_commitment);
        data.extend_from_slice(&self.acknowledged_at.to_le_bytes());
        Ok(data)
    }
    /// Exact signed acknowledgement retry commitment.
    pub fn request_commitment(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        Ok(exact(&self.signing_bytes()?, &self.signature))
    }
    /// Verifies freshness and read capability.
    pub fn verify_at(
        &self,
        reader: &[u8; 32],
        now: u64,
    ) -> Result<(), AnonymousMailboxProtocolError> {
        fresh(self.acknowledged_at, now)?;
        verify(reader, &self.signing_bytes()?, &self.signature)
    }
}

fn fresh(timestamp: u64, now: u64) -> Result<(), AnonymousMailboxProtocolError> {
    if timestamp > now.saturating_add(MAX_ANONYMOUS_MAILBOX_REQUEST_SKEW_SECS)
        || now.saturating_sub(timestamp) > MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS
    {
        Err(AnonymousMailboxProtocolError::Expired)
    } else {
        Ok(())
    }
}

/// Common signed body used under four distinct response frame kinds.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxTerminalResponseV1 {
    /// Schema version.
    pub version: u8,
    /// Request operation.
    pub operation: AnonymousMailboxOperationV1,
    /// Exact request retry id.
    pub request_id: [u8; 16],
    /// Commitment to the complete signed request.
    pub request_commitment: [u8; 32],
    /// Coarse disposition.
    pub outcome: AnonymousMailboxOutcomeV1,
    /// Optional source-sealed result; PullOne carries its item here.
    #[serde(deserialize_with = "item_bytes")]
    pub sealed_payload: Vec<u8>,
    /// Response time.
    pub responded_at: u64,
    /// Signing terminal node.
    pub responder_node_id: [u8; 32],
    /// Terminal-node signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxTerminalResponseV1 {
    /// Creates one request-bound signed terminal response.
    #[allow(clippy::too_many_arguments)]
    pub fn signed(
        operation: AnonymousMailboxOperationV1,
        request_id: [u8; 16],
        request_commitment: [u8; 32],
        outcome: AnonymousMailboxOutcomeV1,
        sealed_payload: Vec<u8>,
        responded_at: u64,
        responder: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            operation,
            request_id,
            request_commitment,
            outcome,
            sealed_payload,
            responded_at,
            responder_node_id: responder.public_key_bytes(),
            signature: [0; 64],
        };
        value.signature = responder.sign(&value.signing_bytes()?);
        Ok(value)
    }
    /// Returns the domain-separated terminal-response transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let mut data = Vec::with_capacity(RESPONSE_DOMAIN.len() + 129);
        data.extend_from_slice(RESPONSE_DOMAIN);
        data.push(self.version);
        data.push(self.operation.code());
        data.extend_from_slice(&self.request_id);
        data.extend_from_slice(&self.request_commitment);
        data.push(self.outcome.code());
        opaque(
            &mut data,
            &self.sealed_payload,
            MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES,
        )?;
        data.extend_from_slice(&self.responded_at.to_le_bytes());
        data.extend_from_slice(&self.responder_node_id);
        Ok(data)
    }
    /// Verifies exact request/operation/responder binding and signature.
    pub fn verify_for_request(
        &self,
        operation: AnonymousMailboxOperationV1,
        request_id: &[u8; 16],
        commitment: &[u8; 32],
        responder: &[u8; 32],
    ) -> Result<(), AnonymousMailboxProtocolError> {
        if self.operation != operation
            || &self.request_id != request_id
            || &self.request_commitment != commitment
            || &self.responder_node_id != responder
        {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        verify(
            &self.responder_node_id,
            &self.signing_bytes()?,
            &self.signature,
        )
    }
}

/// One-target routed opaque request. It contains no chat participant fields.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxRouteRequestV1 {
    /// Schema version.
    pub version: u8,
    /// Random route retry id.
    pub request_id: [u8; 16],
    /// Exact selected custody node.
    pub target_node_id: [u8; 32],
    /// Source-sealed terminal frame.
    #[serde(deserialize_with = "route_bytes")]
    pub sealed_terminal_frame: Vec<u8>,
    /// Request time.
    pub requested_at: u64,
    /// Authenticated previous/source-node signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxRouteRequestV1 {
    /// Creates one source-signed, one-target route request.
    pub fn signed(
        request_id: [u8; 16],
        target_node_id: [u8; 32],
        sealed_terminal_frame: Vec<u8>,
        requested_at: u64,
        source: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            request_id,
            target_node_id,
            sealed_terminal_frame,
            requested_at,
            signature: [0; 64],
        };
        value.signature = source.sign(&value.signing_bytes()?);
        Ok(value)
    }
    /// Returns the domain-separated route request transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        if self.sealed_terminal_frame.is_empty() {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let mut data = Vec::with_capacity(ROUTE_DOMAIN.len() + 93);
        data.extend_from_slice(ROUTE_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.request_id);
        data.extend_from_slice(&self.target_node_id);
        data.extend_from_slice(&self.requested_at.to_le_bytes());
        opaque(
            &mut data,
            &self.sealed_terminal_frame,
            MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES,
        )?;
        Ok(data)
    }
    /// Returns the exact signed route retry commitment.
    pub fn request_commitment(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        Ok(exact(&self.signing_bytes()?, &self.signature))
    }
    /// Verifies source, target, freshness, size, and signature.
    pub fn verify_from(
        &self,
        source: &[u8; 32],
        target: &[u8; 32],
        now: u64,
    ) -> Result<(), AnonymousMailboxProtocolError> {
        if &self.target_node_id != target {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        fresh(self.requested_at, now)?;
        verify(source, &self.signing_bytes()?, &self.signature)
    }
}

impl fmt::Debug for AnonymousMailboxRouteRequestV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnonymousMailboxRouteRequestV1")
            .field("version", &self.version)
            .field("sealed_terminal_bytes", &self.sealed_terminal_frame.len())
            .field("requested_at", &self.requested_at)
            .finish_non_exhaustive()
    }
}

/// One-target routed opaque response bound to the exact request.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxRouteResponseV1 {
    /// Schema version.
    pub version: u8,
    /// Exact route retry id.
    pub request_id: [u8; 16],
    /// Exact signed request commitment.
    pub request_commitment: [u8; 32],
    /// Coarse route outcome.
    pub outcome: AnonymousMailboxOutcomeV1,
    /// Source-sealed terminal response.
    #[serde(deserialize_with = "route_bytes")]
    pub sealed_terminal_response: Vec<u8>,
    /// Response time.
    pub responded_at: u64,
    /// Exact target/responder node.
    pub responder_node_id: [u8; 32],
    /// Target-node signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxRouteResponseV1 {
    /// Creates one target-signed response bound to a request.
    pub fn signed(
        request: &AnonymousMailboxRouteRequestV1,
        outcome: AnonymousMailboxOutcomeV1,
        sealed_terminal_response: Vec<u8>,
        responded_at: u64,
        responder: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        if responder.public_key_bytes() != request.target_node_id {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        let mut value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            request_id: request.request_id,
            request_commitment: request.request_commitment()?,
            outcome,
            sealed_terminal_response,
            responded_at,
            responder_node_id: responder.public_key_bytes(),
            signature: [0; 64],
        };
        value.signature = responder.sign(&value.signing_bytes()?);
        Ok(value)
    }
    /// Returns the domain-separated route-response transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let mut data = Vec::with_capacity(ROUTE_RESPONSE_DOMAIN.len() + 130);
        data.extend_from_slice(ROUTE_RESPONSE_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.request_id);
        data.extend_from_slice(&self.request_commitment);
        data.push(self.outcome.code());
        opaque(
            &mut data,
            &self.sealed_terminal_response,
            MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES,
        )?;
        data.extend_from_slice(&self.responded_at.to_le_bytes());
        data.extend_from_slice(&self.responder_node_id);
        Ok(data)
    }
    /// Verifies exact request, target, and target signature.
    pub fn verify_for_request(
        &self,
        request: &AnonymousMailboxRouteRequestV1,
        responder: &[u8; 32],
    ) -> Result<(), AnonymousMailboxProtocolError> {
        if self.request_id != request.request_id
            || self.request_commitment != request.request_commitment()?
            || &self.responder_node_id != responder
            || &request.target_node_id != responder
        {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        verify(
            &self.responder_node_id,
            &self.signing_bytes()?,
            &self.signature,
        )
    }
}

impl fmt::Debug for AnonymousMailboxRouteResponseV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnonymousMailboxRouteResponseV1")
            .field("version", &self.version)
            .field("outcome", &self.outcome)
            .field(
                "sealed_terminal_bytes",
                &self.sealed_terminal_response.len(),
            )
            .field("responded_at", &self.responded_at)
            .finish_non_exhaustive()
    }
}

/// Explicit terminal frame variants with frozen outer kind values.
#[derive(Clone, PartialEq, Eq)]
pub enum AnonymousMailboxTerminalFrameV1 {
    /// Kind 1.
    LeaseCreate(AnonymousMailboxLeaseCreateV1),
    /// Kind 2.
    Put(AnonymousMailboxPutV1),
    /// Kind 3.
    PullOne(AnonymousMailboxPullOneV1),
    /// Kind 4.
    Ack(AnonymousMailboxAckV1),
    /// Kind 129.
    LeaseCreateResponse(AnonymousMailboxTerminalResponseV1),
    /// Kind 130.
    PutResponse(AnonymousMailboxTerminalResponseV1),
    /// Kind 131.
    PullOneResponse(AnonymousMailboxTerminalResponseV1),
    /// Kind 132.
    AckResponse(AnonymousMailboxTerminalResponseV1),
}

impl AnonymousMailboxTerminalFrameV1 {
    /// Frozen terminal kind byte.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::LeaseCreate(_) => 1,
            Self::Put(_) => 2,
            Self::PullOne(_) => 3,
            Self::Ack(_) => 4,
            Self::LeaseCreateResponse(_) => 129,
            Self::PutResponse(_) => 130,
            Self::PullOneResponse(_) => 131,
            Self::AckResponse(_) => 132,
        }
    }
}

impl fmt::Debug for AnonymousMailboxTerminalFrameV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnonymousMailboxTerminalFrameV1")
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}

fn response_operation(
    frame: &AnonymousMailboxTerminalFrameV1,
) -> Option<AnonymousMailboxOperationV1> {
    match frame {
        AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(_) => {
            Some(AnonymousMailboxOperationV1::LeaseCreate)
        }
        AnonymousMailboxTerminalFrameV1::PutResponse(_) => Some(AnonymousMailboxOperationV1::Put),
        AnonymousMailboxTerminalFrameV1::PullOneResponse(_) => {
            Some(AnonymousMailboxOperationV1::PullOne)
        }
        AnonymousMailboxTerminalFrameV1::AckResponse(_) => Some(AnonymousMailboxOperationV1::Ack),
        _ => None,
    }
}

/// Encodes one terminal value under magic/version/kind/body-length framing.
pub fn encode_anonymous_mailbox_terminal_frame(
    frame: &AnonymousMailboxTerminalFrameV1,
) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
    let body = match frame {
        AnonymousMailboxTerminalFrameV1::LeaseCreate(value) => {
            value.shape()?;
            encode_bincode_bounded(value, BODY_BYTES)
        }
        AnonymousMailboxTerminalFrameV1::Put(value) => {
            value.shape(value.issued_at)?;
            encode_bincode_bounded(value, BODY_BYTES)
        }
        AnonymousMailboxTerminalFrameV1::PullOne(value) => {
            value.signing_bytes()?;
            encode_bincode_bounded(value, BODY_BYTES)
        }
        AnonymousMailboxTerminalFrameV1::Ack(value) => {
            value.signing_bytes()?;
            encode_bincode_bounded(value, BODY_BYTES)
        }
        AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(value)
        | AnonymousMailboxTerminalFrameV1::PutResponse(value)
        | AnonymousMailboxTerminalFrameV1::PullOneResponse(value)
        | AnonymousMailboxTerminalFrameV1::AckResponse(value) => {
            if response_operation(frame) != Some(value.operation) {
                return Err(AnonymousMailboxProtocolError::ClaimsConflict);
            }
            value.signing_bytes()?;
            encode_bincode_bounded(value, BODY_BYTES)
        }
    }
    .map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
    let body_len =
        u32::try_from(body.len()).map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
    let mut encoded = Vec::with_capacity(HEADER_BYTES + body.len());
    encoded.extend_from_slice(&MAGIC);
    encoded.push(ANONYMOUS_MAILBOX_VERSION_V1);
    encoded.push(frame.kind());
    encoded.extend_from_slice(&body_len.to_le_bytes());
    encoded.extend_from_slice(&body);
    if encoded.len() > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxProtocolError::TooLarge);
    }
    Ok(encoded)
}

/// Decodes exactly one canonical terminal frame, rejecting trailing bytes.
pub fn decode_anonymous_mailbox_terminal_frame(
    encoded: &[u8],
) -> Result<AnonymousMailboxTerminalFrameV1, AnonymousMailboxProtocolError> {
    if encoded.len() > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxProtocolError::TooLarge);
    }
    if encoded.len() < HEADER_BYTES || encoded[..2] != MAGIC {
        return Err(AnonymousMailboxProtocolError::Malformed);
    }
    version(encoded[2])?;
    let kind = encoded[3];
    let length = u32::from_le_bytes(
        encoded[4..8]
            .try_into()
            .map_err(|_| AnonymousMailboxProtocolError::Malformed)?,
    ) as usize;
    if length != encoded.len() - HEADER_BYTES {
        return Err(AnonymousMailboxProtocolError::Malformed);
    }
    let body = &encoded[HEADER_BYTES..];
    let frame = match kind {
        1 => AnonymousMailboxTerminalFrameV1::LeaseCreate(decode_body(body)?),
        2 => AnonymousMailboxTerminalFrameV1::Put(decode_body(body)?),
        3 => AnonymousMailboxTerminalFrameV1::PullOne(decode_body(body)?),
        4 => AnonymousMailboxTerminalFrameV1::Ack(decode_body(body)?),
        129 => AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(decode_body(body)?),
        130 => AnonymousMailboxTerminalFrameV1::PutResponse(decode_body(body)?),
        131 => AnonymousMailboxTerminalFrameV1::PullOneResponse(decode_body(body)?),
        132 => AnonymousMailboxTerminalFrameV1::AckResponse(decode_body(body)?),
        _ => return Err(AnonymousMailboxProtocolError::UnsupportedOperation),
    };
    if encode_anonymous_mailbox_terminal_frame(&frame)? != encoded {
        return Err(AnonymousMailboxProtocolError::Malformed);
    }
    Ok(frame)
}

fn decode_body<T: DeserializeOwned>(body: &[u8]) -> Result<T, AnonymousMailboxProtocolError> {
    decode_bincode_bounded(body, BODY_BYTES, TrailingBytesPolicy::Reject)
        .map_err(|_| AnonymousMailboxProtocolError::Malformed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease_fixture() -> (
        AnonymousMailboxLeaseCreateV1,
        IdentityKeyPair,
        IdentityKeyPair,
        IdentityKeyPair,
    ) {
        let target = IdentityKeyPair::from_bytes(&[0x31; 32]).expect("target");
        let deposit = IdentityKeyPair::from_bytes(&[0x32; 32]).expect("deposit");
        let reader = IdentityKeyPair::from_bytes(&[0x33; 32]).expect("reader");
        let issued = 1_800_000_000;
        let expires = issued + 86_400;
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &[0x41; 32],
            &deposit.public_key_bytes(),
            &reader.public_key_bytes(),
            64,
            4 * 1024 * 1024,
            issued,
            expires,
        );
        let ticket = AnonymousMailboxAdmissionTicketV1::issue(
            [0x51; 16],
            claims,
            issued,
            issued + 300,
            &target,
        )
        .expect("ticket");
        let lease = AnonymousMailboxLeaseCreateV1::new(
            [0x41; 32],
            deposit.public_key_bytes(),
            64,
            4 * 1024 * 1024,
            issued,
            expires,
            ticket,
            &reader,
        )
        .expect("lease");
        (lease, target, deposit, reader)
    }

    #[test]
    fn golden_transcripts_and_exact_retry_claims_are_frozen() {
        let (lease, target, _, _) = lease_fixture();
        lease
            .verify_for_target(&target.public_key_bytes(), 1_800_000_100)
            .expect("verify");
        assert_eq!(
            hex::encode(lease.claims_commitment()),
            "2466686eda3f9ba30e967eb835571771cc5746642dc23d58a3b73659f5261dc7"
        );
        assert_eq!(
            hex::encode(lease.request_commitment().expect("commitment")),
            "3d517bf93195c08633f5a61001ca42d2bf5300a6fad9fde75cfc2d87b6fcdf25"
        );
        let mut altered = lease.clone();
        altered.max_bytes += 1;
        assert_eq!(
            altered.verify_for_target(&target.public_key_bytes(), 1_800_000_100),
            Err(AnonymousMailboxProtocolError::ClaimsConflict)
        );
    }

    #[test]
    fn admission_lease_put_pull_and_ack_transcripts_are_golden() {
        let (lease, _, deposit, reader) = lease_fixture();
        let put = AnonymousMailboxPutV1::new(
            lease.mailbox_id,
            [0x61; 16],
            vec![0xa5; 128],
            1_800_000_010,
            1_800_000_110,
            &deposit,
        )
        .expect("put");
        let pull = AnonymousMailboxPullOneV1::new(
            lease.mailbox_id,
            [0x62; 16],
            vec![0xb5; 12],
            1_800_000_020,
            &reader,
        )
        .expect("pull");
        let ack = AnonymousMailboxAckV1::new(
            lease.mailbox_id,
            [0x63; 16],
            put.item_id,
            put.sealed_commitment(),
            1_800_000_030,
            &reader,
        )
        .expect("ack");
        let digest = |bytes: Vec<u8>| hex::encode(Sha256::digest(bytes));
        let actual = vec![
            digest(lease.admission.signing_bytes().expect("ticket transcript")),
            digest(lease.signing_bytes().expect("lease transcript")),
            digest(put.signing_bytes().expect("put transcript")),
            digest(pull.signing_bytes().expect("pull transcript")),
            digest(ack.signing_bytes().expect("ack transcript")),
        ];
        assert_eq!(
            actual,
            vec![
                "ac2dce2461c5a4c3a347f2eaafa1a73236bbe12dc10369ca2b32b7bde7eac151",
                "782262cb12aac411238356bcb7fcf5d4233a4c397c04249005b4f4235a13b37d",
                "4536811206416896a6874249fc0e81e7510da35573bb8fe8f512e78d33450267",
                "f9a06002d1b3dfe83d264a088152ab1a3c3158a81fa8ac2ed579e144e446f314",
                "e8ee682d44ebd34fcc3bb735cb8175a6c20e6ce7b73ad3c526c0f151e77e3ecb",
            ]
        );
    }

    #[test]
    fn all_eight_terminal_kinds_roundtrip_canonically() {
        let (lease, target, deposit, reader) = lease_fixture();
        let put = AnonymousMailboxPutV1::new(
            lease.mailbox_id,
            [0x61; 16],
            vec![0xa5; 128],
            1_800_000_010,
            1_800_000_110,
            &deposit,
        )
        .expect("put");
        let pull = AnonymousMailboxPullOneV1::new(
            lease.mailbox_id,
            [0x62; 16],
            vec![0xb5; 12],
            1_800_000_020,
            &reader,
        )
        .expect("pull");
        let ack = AnonymousMailboxAckV1::new(
            lease.mailbox_id,
            [0x63; 16],
            put.item_id,
            put.sealed_commitment(),
            1_800_000_030,
            &reader,
        )
        .expect("ack");
        let response = |operation, id, commitment| {
            AnonymousMailboxTerminalResponseV1::signed(
                operation,
                id,
                commitment,
                AnonymousMailboxOutcomeV1::Accepted,
                vec![0xc5; 32],
                1_800_000_040,
                &target,
            )
            .expect("response")
        };
        let frames = vec![
            AnonymousMailboxTerminalFrameV1::LeaseCreate(lease.clone()),
            AnonymousMailboxTerminalFrameV1::Put(put.clone()),
            AnonymousMailboxTerminalFrameV1::PullOne(pull.clone()),
            AnonymousMailboxTerminalFrameV1::Ack(ack.clone()),
            AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response(
                AnonymousMailboxOperationV1::LeaseCreate,
                lease.admission.ticket_id,
                lease.request_commitment().expect("lease"),
            )),
            AnonymousMailboxTerminalFrameV1::PutResponse(response(
                AnonymousMailboxOperationV1::Put,
                put.item_id,
                put.request_commitment().expect("put"),
            )),
            AnonymousMailboxTerminalFrameV1::PullOneResponse(response(
                AnonymousMailboxOperationV1::PullOne,
                pull.request_id,
                pull.request_commitment().expect("pull"),
            )),
            AnonymousMailboxTerminalFrameV1::AckResponse(response(
                AnonymousMailboxOperationV1::Ack,
                ack.request_id,
                ack.request_commitment().expect("ack"),
            )),
        ];
        let kinds = [1, 2, 3, 4, 129, 130, 131, 132];
        for (frame, kind) in frames.into_iter().zip(kinds) {
            let encoded = encode_anonymous_mailbox_terminal_frame(&frame).expect("encode");
            assert_eq!(&encoded[..4], &[0x41, 0x4d, 1, kind]);
            assert_eq!(
                decode_anonymous_mailbox_terminal_frame(&encoded).expect("decode"),
                frame
            );
        }
    }

    #[test]
    fn terminal_codec_rejects_version_kind_trailing_and_oversize() {
        let (_, _, _, reader) = lease_fixture();
        let pull = AnonymousMailboxPullOneV1::new(
            [0x41; 32],
            [0x71; 16],
            vec![0; 256],
            1_800_000_020,
            &reader,
        )
        .expect("pull");
        let encoded = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOne(pull),
        )
        .expect("encode");
        let mut bad = encoded.clone();
        bad[2] = 2;
        assert_eq!(
            decode_anonymous_mailbox_terminal_frame(&bad),
            Err(AnonymousMailboxProtocolError::UnsupportedVersion)
        );
        let mut bad = encoded.clone();
        bad[3] = 9;
        assert_eq!(
            decode_anonymous_mailbox_terminal_frame(&bad),
            Err(AnonymousMailboxProtocolError::UnsupportedOperation)
        );
        let mut bad = encoded;
        bad.push(0);
        assert_eq!(
            decode_anonymous_mailbox_terminal_frame(&bad),
            Err(AnonymousMailboxProtocolError::Malformed)
        );
        assert!(matches!(
            AnonymousMailboxPullOneV1::new([0; 32], [0; 16], vec![0; 257], 1, &reader),
            Err(AnonymousMailboxProtocolError::TooLarge)
        ));
        assert_eq!(
            decode_anonymous_mailbox_terminal_frame(&vec![
                0;
                MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES
                    + 1
            ]),
            Err(AnonymousMailboxProtocolError::TooLarge)
        );
    }

    #[test]
    fn route_transcript_binds_target_and_opaque_bytes() {
        let source = IdentityKeyPair::from_bytes(&[0x81; 32]).expect("source");
        let target = IdentityKeyPair::from_bytes(&[0x82; 32]).expect("target");
        let request = AnonymousMailboxRouteRequestV1::signed(
            [0x83; 16],
            target.public_key_bytes(),
            vec![0x84; 96],
            1_800_000_050,
            &source,
        )
        .expect("request");
        request
            .verify_from(
                &source.public_key_bytes(),
                &target.public_key_bytes(),
                1_800_000_051,
            )
            .expect("verify");
        assert_eq!(
            hex::encode(request.request_commitment().expect("commitment")),
            "492806ed99337973d022305f0855c91b291bbc0ecce12fd8cadb9da972745cde"
        );
        let response = AnonymousMailboxRouteResponseV1::signed(
            &request,
            AnonymousMailboxOutcomeV1::Accepted,
            vec![0x85; 64],
            1_800_000_052,
            &target,
        )
        .expect("response");
        response
            .verify_for_request(&request, &target.public_key_bytes())
            .expect("verify response");
        let mut redirected = request;
        redirected.target_node_id = source.public_key_bytes();
        assert_eq!(
            redirected.verify_from(
                &source.public_key_bytes(),
                &source.public_key_bytes(),
                1_800_000_051
            ),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
        );
        assert!(matches!(
            AnonymousMailboxRouteRequestV1::signed(
                [0x83; 16],
                target.public_key_bytes(),
                vec![0; MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES + 1],
                1_800_000_050,
                &source
            ),
            Err(AnonymousMailboxProtocolError::TooLarge)
        ));
    }
}
