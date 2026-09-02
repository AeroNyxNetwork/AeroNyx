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
//!
//! Last Modified: v1.3.0-AnonymousMailboxTicketIssuer — Added the bounded,
//! target-issued admission-ticket request/response building block.
//! v1.2.0-AnonymousMailboxSourceTerminalCarrier — Added the
//! canonical, non-circular request carrier that conveys the compact response
//! reply key and its request-frame binding to the terminal.
//! v1.1.0-AnonymousMailboxSourceSeal — Froze the padded pull result codec,
//! the compact source-sealed response carrier, and the reduced 159-KiB
//! admitted item ceiling under unchanged outer route limits.

use std::fmt;

use hkdf::Hkdf;
use rand::{rngs::OsRng, RngCore};
use serde::de::{DeserializeOwned, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::Zeroize;

use crate::crypto::{E2eSession, EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey};
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

const TICKET_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-AdmissionTicket-v1";
const TICKET_ISSUE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-TicketIssue-v1";
const TICKET_ISSUE_WORK_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-TicketIssueWork-v1";
const TICKET_ISSUE_RESPONSE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-TicketIssueResponse-v1";
const LEASE_CLAIMS_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-LeaseClaims-v1";
const LEASE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-LeaseCreate-v1";
const PUT_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-Put-v1";
const PULL_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-PullOne-v1";
const ACK_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-Ack-v1";
const RESPONSE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-Response-v1";
const ROUTE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-RouteRequest-v1";
const ROUTE_RESPONSE_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-RouteResponse-v1";
const EXACT_REQUEST_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-ExactRequest-v1";
const SOURCE_TERMINAL_CONTEXT_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-SourceTerminalContext-v1";
const SOURCE_TERMINAL_FRAME_DOMAIN: &[u8] = b"AeroNyx-AnonymousMailbox-SourceTerminalFrame-v1";

/// Initial anonymous mailbox protocol version.
pub const ANONYMOUS_MAILBOX_VERSION_V1: u8 = 1;
/// Maximum sealed ChatEnvelope bytes in one Put.
pub const MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES: usize = 159 * 1024;
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
/// Maximum bytes carried inside one signed terminal response payload.
pub const MAX_ANONYMOUS_MAILBOX_TERMINAL_RESPONSE_PAYLOAD_BYTES: usize = 160 * 1024;
/// Maximum encoded bytes of one compact source-sealed terminal response.
pub const MAX_ANONYMOUS_MAILBOX_SOURCE_SEALED_RESPONSE_BYTES: usize =
    MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES;
/// Maximum plaintext bytes admitted by the compact source-sealed carrier.
pub const MAX_ANONYMOUS_MAILBOX_SOURCE_SEALED_RESPONSE_PAYLOAD_BYTES: usize =
    MAX_ANONYMOUS_MAILBOX_SOURCE_SEALED_RESPONSE_BYTES - SOURCE_SEALED_RESPONSE_OVERHEAD_BYTES;
/// Maximum canonical request frame admitted inside one source-terminal carrier.
///
/// The carrier's fixed header still leaves more than 14 KiB below the existing
/// 191-KiB route field ceiling at this terminal-codec maximum.
pub const MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_REQUEST_FRAME_BYTES: usize =
    MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES;
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
/// Largest configurable proof-of-work difficulty for anonymous ticket issue.
///
/// The target supplies the actual non-zero difficulty; this hard ceiling keeps
/// an accidentally hostile configuration from making the protocol unusable.
pub const MAX_ANONYMOUS_MAILBOX_TICKET_ISSUE_WORK_BITS: u8 = 24;

const MAGIC: [u8; 2] = [0x41, 0x4d];
const HEADER_BYTES: usize = 8;
const BODY_BYTES: u64 = (MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES - HEADER_BYTES) as u64;
const PULL_RESULT_MAGIC: [u8; 4] = [0x41, 0x4d, 0x01, 0x01];
const SOURCE_SEALED_RESPONSE_MAGIC: [u8; 4] = *b"AMSR";
const SOURCE_SEALED_RESPONSE_VERSION_V1: u8 = 1;
const SOURCE_SEALED_RESPONSE_PREFIX_BYTES: usize = 4 + 1 + 32 + 24 + 4;
const SOURCE_SEALED_RESPONSE_AEAD_TAG_BYTES: usize = 16;
const SOURCE_SEALED_RESPONSE_OVERHEAD_BYTES: usize =
    SOURCE_SEALED_RESPONSE_PREFIX_BYTES + SOURCE_SEALED_RESPONSE_AEAD_TAG_BYTES;
const SOURCE_SEALED_RESPONSE_KEY_SALT: &[u8] = b"AeroNyx-AnonymousMailbox-SourceSeal-Key-v1";
const SOURCE_SEALED_RESPONSE_RESTART_MAGIC: [u8; 4] = *b"AMSS";
const SOURCE_SEALED_RESPONSE_RESTART_VERSION_V1: u16 = 1;
const SOURCE_SEALED_RESPONSE_RESTART_BYTES: usize = 4 + 2 + 16 + 32 + 32 + 32 + 32;
const SOURCE_TERMINAL_CARRIER_MAGIC: [u8; 4] = *b"AMST";
const SOURCE_TERMINAL_CARRIER_VERSION_V1: u8 = 1;
const SOURCE_TERMINAL_CARRIER_PREFIX_BYTES: usize = 4 + 1 + 32 + 32 + 4;

/// Exact fixed wire bytes for one canonical padded pull result.
pub const ANONYMOUS_MAILBOX_PULL_RESULT_BYTES: usize = 4
    + 16
    + 32
    + 4
    + 2
    + MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES
    + MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES;

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
    /// A target-bound anonymous admission proof did not satisfy policy.
    #[error("anonymous mailbox admission proof was rejected")]
    ProofRejected,
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
    /// Target-issued admission ticket.
    TicketIssue = 5,
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
fn response_payload_bytes<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    bounded_bytes::<D, MAX_ANONYMOUS_MAILBOX_TERMINAL_RESPONSE_PAYLOAD_BYTES>(deserializer)
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

fn fixed<const N: usize>(bytes: &[u8]) -> Result<[u8; N], AnonymousMailboxProtocolError> {
    bytes
        .try_into()
        .map_err(|_| AnonymousMailboxProtocolError::Malformed)
}

fn source_sealed_response_size(
    ciphertext_len: usize,
) -> Result<usize, AnonymousMailboxProtocolError> {
    SOURCE_SEALED_RESPONSE_PREFIX_BYTES
        .checked_add(ciphertext_len)
        .ok_or(AnonymousMailboxProtocolError::TooLarge)
}

fn derive_source_sealed_response_key(
    route_id: &[u8; 16],
    request_context_commitment: &[u8; 32],
    expected_terminal_node_id: &[u8; 32],
    reply_public_key: &[u8; 32],
    ephemeral_public_key: &[u8; 32],
    shared_secret: &[u8; 32],
) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
    let mut info = Vec::with_capacity(16 + 32 + 32 + 32);
    info.extend_from_slice(route_id);
    info.extend_from_slice(request_context_commitment);
    info.extend_from_slice(expected_terminal_node_id);
    info.extend_from_slice(reply_public_key);
    info.extend_from_slice(ephemeral_public_key);
    let hkdf = Hkdf::<Sha256>::new(Some(SOURCE_SEALED_RESPONSE_KEY_SALT), shared_secret);
    let mut key = [0u8; 32];
    hkdf.expand(&info, &mut key)
        .map_err(|_| AnonymousMailboxProtocolError::Malformed)?;
    Ok(key)
}

fn source_terminal_frame_commitment(
    terminal_frame: &[u8],
) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
    if terminal_frame.len() > MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_REQUEST_FRAME_BYTES {
        return Err(AnonymousMailboxProtocolError::TooLarge);
    }
    let frame_len =
        u32::try_from(terminal_frame.len()).map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
    let mut hasher = Sha256::new();
    hasher.update(SOURCE_TERMINAL_FRAME_DOMAIN);
    hasher.update(frame_len.to_le_bytes());
    hasher.update(terminal_frame);
    Ok(hasher.finalize().into())
}

fn source_terminal_context_commitment(
    route_id: &[u8; 16],
    target_node_id: &[u8; 32],
    terminal_frame: &[u8],
) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
    validate_canonical_terminal_request_frame(terminal_frame)?;
    IdentityPublicKey::from_bytes(target_node_id)
        .map_err(|_| AnonymousMailboxProtocolError::SignatureRejected)?;
    let frame_commitment = source_terminal_frame_commitment(terminal_frame)?;
    let mut hasher = Sha256::new();
    hasher.update(SOURCE_TERMINAL_CONTEXT_DOMAIN);
    hasher.update(SOURCE_TERMINAL_CARRIER_VERSION_V1.to_le_bytes());
    hasher.update(route_id);
    hasher.update(target_node_id);
    hasher.update(frame_commitment);
    Ok(hasher.finalize().into())
}

fn source_reply_public_key_is_valid(reply_public_key: &[u8; 32]) -> bool {
    if reply_public_key.iter().all(|byte| *byte == 0) {
        return false;
    }

    // [ANONYMOUS-MAILBOX-SOURCE-CARRIER 2026-09-02 by Codex] X25519 has no
    // fallible public-key parser. A fixed clamped scalar maps every low-order
    // point to the all-zero shared secret, letting terminal admission reject
    // it before it reaches the response-sealing path.
    let probe = StaticSecret::from([0x6d; 32]);
    let shared = probe.diffie_hellman(&X25519PublicKey::from(*reply_public_key));
    !shared.as_bytes().iter().all(|byte| *byte == 0)
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

/// Anonymous, target-bound request for one short-lived admission ticket.
///
/// The request intentionally contains no sender, receiver, wallet, route, IP,
/// or client signature. Its target-bound proof of work is a coarse resource
/// admission signal, not an identity. The target signs the returned
/// [`AnonymousMailboxAdmissionTicketV1`] only after durable exact-replay and
/// capacity checks in the local custody repository.
///
/// [ANONYMOUS-MAILBOX-TICKET-ISSUER 2026-09-03 by Codex] `request_id` and
/// `ticket_id` are distinct: the former binds terminal/route replay while the
/// latter is consumed once by LeaseCreate. Reusing either for a different
/// canonical request is a conflict, never a replacement.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxTicketIssueV1 {
    /// Schema version.
    pub version: u8,
    /// Exact terminal request retry identifier.
    pub request_id: [u8; 16],
    /// One-time ticket identifier later consumed by LeaseCreate.
    pub ticket_id: [u8; 16],
    /// Exact custody target expected to sign the ticket.
    pub target_node_id: [u8; 32],
    /// Commitment to every immutable lease claim.
    pub lease_claims_commitment: [u8; 32],
    /// Request issue time in Unix seconds.
    pub issued_at: u64,
    /// Requested ticket expiry in Unix seconds.
    pub expires_at: u64,
    /// Target-bound proof-of-work nonce.
    pub proof_nonce: u64,
}

impl AnonymousMailboxTicketIssueV1 {
    /// Creates one canonical ticket issuance request.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: [u8; 16],
        ticket_id: [u8; 16],
        target_node_id: [u8; 32],
        lease_claims_commitment: [u8; 32],
        issued_at: u64,
        expires_at: u64,
        proof_nonce: u64,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        let value = Self {
            version: ANONYMOUS_MAILBOX_VERSION_V1,
            request_id,
            ticket_id,
            target_node_id,
            lease_claims_commitment,
            issued_at,
            expires_at,
            proof_nonce,
        };
        value.shape()?;
        Ok(value)
    }

    /// Returns the domain-separated canonical request transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        self.shape()?;
        let mut data =
            Vec::with_capacity(TICKET_ISSUE_DOMAIN.len() + 1 + 16 + 16 + 32 + 32 + 8 + 8 + 8);
        data.extend_from_slice(TICKET_ISSUE_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.request_id);
        data.extend_from_slice(&self.ticket_id);
        data.extend_from_slice(&self.target_node_id);
        data.extend_from_slice(&self.lease_claims_commitment);
        data.extend_from_slice(&self.issued_at.to_le_bytes());
        data.extend_from_slice(&self.expires_at.to_le_bytes());
        data.extend_from_slice(&self.proof_nonce.to_le_bytes());
        Ok(data)
    }

    /// Returns the exact durable replay commitment for this unsigned request.
    pub fn request_commitment(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        let bytes = self.signing_bytes()?;
        let mut hash = Sha256::new();
        hash.update(EXACT_REQUEST_DOMAIN);
        hash.update(bytes);
        Ok(hash.finalize().into())
    }

    /// Returns the target-bound proof-of-work digest.
    pub fn proof_digest(&self) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        let mut hash = Sha256::new();
        hash.update(TICKET_ISSUE_WORK_DOMAIN);
        hash.update(self.signing_bytes()?);
        Ok(hash.finalize().into())
    }

    /// Verifies target, short lifetime, and configured proof-of-work.
    pub fn verify_for_target(
        &self,
        target_node_id: &[u8; 32],
        now: u64,
        work_bits: u8,
    ) -> Result<(), AnonymousMailboxProtocolError> {
        self.shape()?;
        if work_bits == 0 || work_bits > MAX_ANONYMOUS_MAILBOX_TICKET_ISSUE_WORK_BITS {
            return Err(AnonymousMailboxProtocolError::ProofRejected);
        }
        if &self.target_node_id != target_node_id {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        window(
            self.issued_at,
            self.expires_at,
            now,
            MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS,
        )?;
        if leading_zero_bits(&self.proof_digest()?) < u32::from(work_bits) {
            return Err(AnonymousMailboxProtocolError::ProofRejected);
        }
        Ok(())
    }

    fn shape(&self) -> Result<(), AnonymousMailboxProtocolError> {
        version(self.version)?;
        IdentityPublicKey::from_bytes(&self.target_node_id)
            .map_err(|_| AnonymousMailboxProtocolError::SignatureRejected)?;
        window(
            self.issued_at,
            self.expires_at,
            self.issued_at,
            MAX_ANONYMOUS_MAILBOX_ADMISSION_TTL_SECS,
        )
    }
}

fn leading_zero_bits(digest: &[u8; 32]) -> u32 {
    digest
        .iter()
        .map(|byte| byte.leading_zeros())
        .scan(true, |prefix, bits| {
            let current = *prefix;
            *prefix &= bits == 8;
            Some((current, bits))
        })
        .take_while(|(prefix, _)| *prefix)
        .map(|(_, bits)| bits)
        .sum()
}

impl fmt::Debug for AnonymousMailboxTicketIssueV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxTicketIssueV1")
            .field("version", &self.version)
            .field("capabilities", &"<redacted>")
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

/// Target-signed canonical result for one ticket issue request.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousMailboxTicketIssueResponseV1 {
    /// Schema version.
    pub version: u8,
    /// Exact request retry identifier.
    pub request_id: [u8; 16],
    /// Commitment to the complete canonical issue request.
    pub request_commitment: [u8; 32],
    /// Coarse issuance result.
    pub outcome: AnonymousMailboxOutcomeV1,
    /// Present only for a durably issued ticket.
    pub ticket: Option<AnonymousMailboxAdmissionTicketV1>,
    /// Target response time in Unix seconds.
    pub responded_at: u64,
    /// Target node that signed this response.
    pub responder_node_id: [u8; 32],
    /// Target-node signature.
    #[serde(with = "bytes64")]
    pub signature: [u8; 64],
}

impl AnonymousMailboxTicketIssueResponseV1 {
    /// Creates one target-signed, request-bound ticket result.
    pub fn signed(
        request: &AnonymousMailboxTicketIssueV1,
        outcome: AnonymousMailboxOutcomeV1,
        ticket: Option<AnonymousMailboxAdmissionTicketV1>,
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
            ticket,
            responded_at,
            responder_node_id: responder.public_key_bytes(),
            signature: [0; 64],
        };
        value.validate_binding(request, &value.responder_node_id)?;
        value.signature = responder.sign(&value.signing_bytes()?);
        Ok(value)
    }

    /// Returns the frozen response signing transcript.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        version(self.version)?;
        let ticket = bincode::serialize(&self.ticket)
            .map_err(|_| AnonymousMailboxProtocolError::Malformed)?;
        let ticket_len =
            u16::try_from(ticket.len()).map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
        let mut data = Vec::with_capacity(
            TICKET_ISSUE_RESPONSE_DOMAIN.len() + 1 + 16 + 32 + 1 + 2 + ticket.len() + 8 + 32,
        );
        data.extend_from_slice(TICKET_ISSUE_RESPONSE_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.request_id);
        data.extend_from_slice(&self.request_commitment);
        data.push(self.outcome.code());
        data.extend_from_slice(&ticket_len.to_le_bytes());
        data.extend_from_slice(&ticket);
        data.extend_from_slice(&self.responded_at.to_le_bytes());
        data.extend_from_slice(&self.responder_node_id);
        Ok(data)
    }

    /// Verifies response/target/request binding without exposing ticket data.
    pub fn verify_for_request(
        &self,
        request: &AnonymousMailboxTicketIssueV1,
        responder: &[u8; 32],
    ) -> Result<(), AnonymousMailboxProtocolError> {
        self.validate_binding(request, responder)?;
        verify(
            &self.responder_node_id,
            &self.signing_bytes()?,
            &self.signature,
        )
    }

    fn validate_binding(
        &self,
        request: &AnonymousMailboxTicketIssueV1,
        responder: &[u8; 32],
    ) -> Result<(), AnonymousMailboxProtocolError> {
        version(self.version)?;
        if self.request_id != request.request_id
            || self.request_commitment != request.request_commitment()?
            || &self.responder_node_id != responder
            || &request.target_node_id != responder
        {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        match (self.outcome, &self.ticket) {
            (AnonymousMailboxOutcomeV1::Accepted, Some(ticket)) => {
                if ticket.ticket_id != request.ticket_id
                    || ticket.target_node_id != request.target_node_id
                    || ticket.lease_claims_commitment != request.lease_claims_commitment
                    || ticket.issued_at != request.issued_at
                    || ticket.expires_at != request.expires_at
                {
                    return Err(AnonymousMailboxProtocolError::ClaimsConflict);
                }
                ticket.verify_at(
                    &request.target_node_id,
                    &request.lease_claims_commitment,
                    request.issued_at,
                )?;
            }
            (AnonymousMailboxOutcomeV1::Accepted, None) | (_, Some(_)) => {
                return Err(AnonymousMailboxProtocolError::Malformed);
            }
            (_, None) => {}
        }
        Ok(())
    }
}

impl fmt::Debug for AnonymousMailboxTicketIssueResponseV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnonymousMailboxTicketIssueResponseV1")
            .field("version", &self.version)
            .field("outcome", &self.outcome)
            .field("ticket", &self.ticket.as_ref().map(|_| "<redacted>"))
            .field("responded_at", &self.responded_at)
            .finish_non_exhaustive()
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
    #[serde(deserialize_with = "response_payload_bytes")]
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
            MAX_ANONYMOUS_MAILBOX_TERMINAL_RESPONSE_PAYLOAD_BYTES,
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

/// Canonical fixed-width pull result carried inside a terminal response.
///
/// [ANONYMOUS-MAILBOX-PULL-RESULT 2026-09-02 by Codex] This freezes the exact
/// 163130-byte wire image so future field reordering, omitted metadata, or
/// relaxed padding checks fail closed during decode and fixture replay.
#[derive(Clone, PartialEq, Eq)]
pub struct AnonymousMailboxPullResultV1 {
    /// Durable item identifier.
    pub item_id: [u8; 16],
    /// Commitment to the unpadded sealed item bytes.
    pub sealed_commitment: [u8; 32],
    /// Opaque cursor prefix; the remaining fixed field bytes are zero padding.
    pub cursor: Vec<u8>,
    /// Opaque sealed item prefix; the remaining fixed field bytes are zero padding.
    pub sealed_item: Vec<u8>,
}

impl AnonymousMailboxPullResultV1 {
    /// Builds one canonical pull result from opaque stored bytes.
    pub fn new(
        item_id: [u8; 16],
        cursor: Vec<u8>,
        sealed_item: Vec<u8>,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        if cursor.len() > MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES
            || sealed_item.len() > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
        {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        if sealed_item.is_empty() {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        Ok(Self {
            item_id,
            sealed_commitment: Sha256::digest(&sealed_item).into(),
            cursor,
            sealed_item,
        })
    }

    /// Returns the exact fixed-width wire bytes.
    pub fn encode(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        if self.cursor.len() > MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES
            || self.sealed_item.len() > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
        {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        if self.sealed_item.is_empty() {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let expected_commitment: [u8; 32] = Sha256::digest(&self.sealed_item).into();
        if self.sealed_commitment != expected_commitment {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        let cursor_len = u16::try_from(self.cursor.len())
            .map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
        let sealed_len = u32::try_from(self.sealed_item.len())
            .map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
        let mut encoded = Vec::with_capacity(ANONYMOUS_MAILBOX_PULL_RESULT_BYTES);
        encoded.extend_from_slice(&PULL_RESULT_MAGIC);
        encoded.extend_from_slice(&self.item_id);
        encoded.extend_from_slice(&self.sealed_commitment);
        encoded.extend_from_slice(&sealed_len.to_le_bytes());
        encoded.extend_from_slice(&cursor_len.to_le_bytes());
        encoded.extend_from_slice(&self.cursor);
        encoded.resize(58 + MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES, 0);
        encoded.extend_from_slice(&self.sealed_item);
        encoded.resize(ANONYMOUS_MAILBOX_PULL_RESULT_BYTES, 0);
        Ok(encoded)
    }

    /// Decodes one exact canonical padded pull result and rejects drift.
    pub fn decode(encoded: &[u8]) -> Result<Self, AnonymousMailboxProtocolError> {
        if encoded.len() != ANONYMOUS_MAILBOX_PULL_RESULT_BYTES || encoded[..4] != PULL_RESULT_MAGIC
        {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let item_id = fixed::<16>(&encoded[4..20])?;
        let sealed_commitment = fixed::<32>(&encoded[20..52])?;
        let sealed_len = u32::from_le_bytes(fixed::<4>(&encoded[52..56])?) as usize;
        let cursor_len = u16::from_le_bytes(fixed::<2>(&encoded[56..58])?) as usize;
        if sealed_len == 0
            || sealed_len > MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES
            || cursor_len > MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES
        {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let cursor_block = &encoded[58..58 + MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES];
        let item_block = &encoded[58 + MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES..];
        if cursor_block[cursor_len..].iter().any(|byte| *byte != 0)
            || item_block[sealed_len..].iter().any(|byte| *byte != 0)
        {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let cursor = cursor_block[..cursor_len].to_vec();
        let sealed_item = item_block[..sealed_len].to_vec();
        let expected_commitment: [u8; 32] = Sha256::digest(&sealed_item).into();
        if sealed_commitment != expected_commitment {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        Ok(Self {
            item_id,
            sealed_commitment,
            cursor,
            sealed_item,
        })
    }
}

impl fmt::Debug for AnonymousMailboxPullResultV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnonymousMailboxPullResultV1")
            .field("cursor_bytes", &self.cursor.len())
            .field("sealed_item_bytes", &self.sealed_item.len())
            .finish_non_exhaustive()
    }
}

/// Zeroizing private session bytes accepted only by an authenticated journal.
pub struct AnonymousMailboxSourceSealSessionRestartState {
    bytes: Vec<u8>,
}

impl AnonymousMailboxSourceSealSessionRestartState {
    /// Returns the opaque journal bytes for this single-use session.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl Drop for AnonymousMailboxSourceSealSessionRestartState {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

/// Coarse restart-state restore failures for the compact source-seal session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymousMailboxSourceSealSessionRestartError {
    /// Stored bytes are truncated or structurally invalid.
    Malformed,
    /// The opaque restart-state version is not supported.
    UnsupportedVersion,
    /// The retained reply key or bound context is unusable.
    InvalidSession,
}

struct RecoverableAnonymousMailboxReplyKey {
    secret: Option<StaticSecret>,
    public: X25519PublicKey,
}

impl RecoverableAnonymousMailboxReplyKey {
    fn generate() -> Self {
        let secret = StaticSecret::new(OsRng);
        let public = X25519PublicKey::from(&secret);
        Self {
            secret: Some(secret),
            public,
        }
    }

    fn public_key_bytes(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    fn persistence_secret(
        &self,
    ) -> Result<[u8; 32], AnonymousMailboxSourceSealSessionRestartError> {
        self.secret
            .as_ref()
            .map(StaticSecret::to_bytes)
            .ok_or(AnonymousMailboxSourceSealSessionRestartError::InvalidSession)
    }

    fn from_persistence_secret(
        mut secret_bytes: [u8; 32],
        public_bytes: [u8; 32],
    ) -> Result<Self, AnonymousMailboxSourceSealSessionRestartError> {
        if secret_bytes == [0; 32] {
            secret_bytes.zeroize();
            return Err(AnonymousMailboxSourceSealSessionRestartError::InvalidSession);
        }
        let secret = StaticSecret::from(secret_bytes);
        secret_bytes.zeroize();
        let public = X25519PublicKey::from(&secret);
        if public.to_bytes() != public_bytes {
            return Err(AnonymousMailboxSourceSealSessionRestartError::InvalidSession);
        }
        Ok(Self {
            secret: Some(secret),
            public,
        })
    }

    fn exchange(
        &mut self,
        peer_public: &[u8; 32],
    ) -> Result<[u8; 32], AnonymousMailboxProtocolError> {
        let secret = self
            .secret
            .take()
            .ok_or(AnonymousMailboxProtocolError::Malformed)?;
        let peer = X25519PublicKey::from(*peer_public);
        let shared = secret.diffie_hellman(&peer);
        Ok(*shared.as_bytes())
    }
}

/// Private single-use state for one mailbox-specific compact response.
///
/// [ANONYMOUS-MAILBOX-SOURCE-SEAL 2026-09-02 by Codex] The caller persists the
/// restart state only inside an authenticated encrypted journal and must delete
/// that journal entry after the first successful or terminally failed open.
pub struct AnonymousMailboxSourceSealSessionV1 {
    route_id: [u8; 16],
    expected_terminal_node_id: [u8; 32],
    request_context_commitment: [u8; 32],
    reply_key: RecoverableAnonymousMailboxReplyKey,
}

impl AnonymousMailboxSourceSealSessionV1 {
    /// Creates one single-use source session and returns its reply public key.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request_context_commitment: [u8; 32],
    ) -> Result<([u8; 32], Self), AnonymousMailboxProtocolError> {
        IdentityPublicKey::from_bytes(&expected_terminal_node_id)
            .map_err(|_| AnonymousMailboxProtocolError::SignatureRejected)?;
        let reply_key = RecoverableAnonymousMailboxReplyKey::generate();
        let reply_public_key = reply_key.public_key_bytes();
        Ok((
            reply_public_key,
            Self {
                route_id,
                expected_terminal_node_id,
                request_context_commitment,
                reply_key,
            },
        ))
    }

    /// Returns the single-use X25519 reply public key bound to this session.
    pub fn reply_public_key(&self) -> [u8; 32] {
        self.reply_key.public_key_bytes()
    }

    /// Encodes opaque restart state for immediate encrypted journaling only.
    pub fn encode_restart_state(
        &self,
    ) -> Result<
        AnonymousMailboxSourceSealSessionRestartState,
        AnonymousMailboxSourceSealSessionRestartError,
    > {
        let mut secret_bytes = self.reply_key.persistence_secret()?;
        let mut bytes = Vec::with_capacity(SOURCE_SEALED_RESPONSE_RESTART_BYTES);
        bytes.extend_from_slice(&SOURCE_SEALED_RESPONSE_RESTART_MAGIC);
        bytes.extend_from_slice(&SOURCE_SEALED_RESPONSE_RESTART_VERSION_V1.to_be_bytes());
        bytes.extend_from_slice(&self.route_id);
        bytes.extend_from_slice(&self.expected_terminal_node_id);
        bytes.extend_from_slice(&self.request_context_commitment);
        bytes.extend_from_slice(&self.reply_key.public_key_bytes());
        bytes.extend_from_slice(&secret_bytes);
        secret_bytes.zeroize();
        Ok(AnonymousMailboxSourceSealSessionRestartState { bytes })
    }

    /// Restores one unconsumed session from authenticated journal bytes.
    pub fn decode_restart_state(
        bytes: &[u8],
    ) -> Result<Self, AnonymousMailboxSourceSealSessionRestartError> {
        if bytes.len() != SOURCE_SEALED_RESPONSE_RESTART_BYTES
            || bytes[..4] != SOURCE_SEALED_RESPONSE_RESTART_MAGIC
        {
            return Err(AnonymousMailboxSourceSealSessionRestartError::Malformed);
        }
        if u16::from_be_bytes([bytes[4], bytes[5]]) != SOURCE_SEALED_RESPONSE_RESTART_VERSION_V1 {
            return Err(AnonymousMailboxSourceSealSessionRestartError::UnsupportedVersion);
        }
        let route_id = fixed::<16>(&bytes[6..22])
            .map_err(|_| AnonymousMailboxSourceSealSessionRestartError::Malformed)?;
        let expected_terminal_node_id = fixed::<32>(&bytes[22..54])
            .map_err(|_| AnonymousMailboxSourceSealSessionRestartError::Malformed)?;
        IdentityPublicKey::from_bytes(&expected_terminal_node_id)
            .map_err(|_| AnonymousMailboxSourceSealSessionRestartError::InvalidSession)?;
        let request_context_commitment = fixed::<32>(&bytes[54..86])
            .map_err(|_| AnonymousMailboxSourceSealSessionRestartError::Malformed)?;
        let public_bytes = fixed::<32>(&bytes[86..118])
            .map_err(|_| AnonymousMailboxSourceSealSessionRestartError::Malformed)?;
        let secret_bytes = fixed::<32>(&bytes[118..150])
            .map_err(|_| AnonymousMailboxSourceSealSessionRestartError::Malformed)?;
        let reply_key = RecoverableAnonymousMailboxReplyKey::from_persistence_secret(
            secret_bytes,
            public_bytes,
        )?;
        Ok(Self {
            route_id,
            expected_terminal_node_id,
            request_context_commitment,
            reply_key,
        })
    }

    /// Opens the one compact response bound to this exact route context.
    pub fn open(
        &mut self,
        encoded_response: &[u8],
    ) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        let response = AnonymousMailboxSourceSealedResponseV1::decode(encoded_response)?;
        let reply_public_key = self.reply_key.public_key_bytes();
        let mut shared_secret = self.reply_key.exchange(&response.ephemeral_public_key)?;
        if shared_secret.iter().all(|byte| *byte == 0) {
            shared_secret.zeroize();
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let mut key = derive_source_sealed_response_key(
            &self.route_id,
            &self.request_context_commitment,
            &self.expected_terminal_node_id,
            &reply_public_key,
            &response.ephemeral_public_key,
            &shared_secret,
        )?;
        shared_secret.zeroize();
        let session = E2eSession::new(key, response.ephemeral_public_key);
        key.zeroize();
        session
            .decrypt_raw(&response.ciphertext, &response.nonce)
            .map_err(|_| AnonymousMailboxProtocolError::SignatureRejected)
    }
}

/// Compact mailbox-specific source-sealed route response bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct AnonymousMailboxSourceSealedResponseV1 {
    /// Schema version.
    pub version: u8,
    /// Terminal-chosen ephemeral X25519 public key.
    pub ephemeral_public_key: [u8; 32],
    /// AEAD nonce.
    pub nonce: [u8; 24],
    /// Ciphertext including the XChaCha20-Poly1305 tag.
    pub ciphertext: Vec<u8>,
}

impl AnonymousMailboxSourceSealedResponseV1 {
    fn validate_shape(&self) -> Result<(), AnonymousMailboxProtocolError> {
        if self.version != SOURCE_SEALED_RESPONSE_VERSION_V1 {
            return Err(AnonymousMailboxProtocolError::UnsupportedVersion);
        }
        if self.ciphertext.len() < SOURCE_SEALED_RESPONSE_AEAD_TAG_BYTES {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        if source_sealed_response_size(self.ciphertext.len())?
            > MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES
            || self.ciphertext.len() > u32::MAX as usize
        {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        Ok(())
    }

    /// Encrypts already terminal-signed route-response bytes for the source.
    pub fn seal(
        route_id: [u8; 16],
        request_context_commitment: [u8; 32],
        expected_terminal_node_id: [u8; 32],
        reply_public_key: [u8; 32],
        payload: &[u8],
        terminal_identity: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        IdentityPublicKey::from_bytes(&expected_terminal_node_id)
            .map_err(|_| AnonymousMailboxProtocolError::SignatureRejected)?;
        if terminal_identity.public_key_bytes() != expected_terminal_node_id {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        if payload.is_empty() {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        if payload.len() > MAX_ANONYMOUS_MAILBOX_SOURCE_SEALED_RESPONSE_PAYLOAD_BYTES {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }

        let ephemeral = EphemeralKeyPair::generate();
        let ephemeral_public_key = ephemeral.public_key_bytes();
        let mut shared_secret = ephemeral.exchange(&reply_public_key);
        if shared_secret.iter().all(|byte| *byte == 0) {
            shared_secret.zeroize();
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let mut key = derive_source_sealed_response_key(
            &route_id,
            &request_context_commitment,
            &expected_terminal_node_id,
            &reply_public_key,
            &ephemeral_public_key,
            &shared_secret,
        )?;
        shared_secret.zeroize();
        let mut nonce = [0u8; 24];
        OsRng.fill_bytes(&mut nonce);
        let session = E2eSession::new(key, reply_public_key);
        key.zeroize();
        let ciphertext = session
            .encrypt_raw(payload, &nonce)
            .map_err(|_| AnonymousMailboxProtocolError::Malformed)?;
        let response = Self {
            version: SOURCE_SEALED_RESPONSE_VERSION_V1,
            ephemeral_public_key,
            nonce,
            ciphertext,
        };
        response.validate_shape()?;
        Ok(response)
    }

    /// Returns the exact compact transport bytes with no trailing slack.
    pub fn encode(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        self.validate_shape()?;
        let ciphertext_len = u32::try_from(self.ciphertext.len())
            .map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
        let mut encoded = Vec::with_capacity(source_sealed_response_size(self.ciphertext.len())?);
        encoded.extend_from_slice(&SOURCE_SEALED_RESPONSE_MAGIC);
        encoded.push(self.version);
        encoded.extend_from_slice(&self.ephemeral_public_key);
        encoded.extend_from_slice(&self.nonce);
        encoded.extend_from_slice(&ciphertext_len.to_be_bytes());
        encoded.extend_from_slice(&self.ciphertext);
        Ok(encoded)
    }

    /// Decodes one exact compact transport value and rejects trailing bytes.
    pub fn decode(encoded: &[u8]) -> Result<Self, AnonymousMailboxProtocolError> {
        if encoded.len() < SOURCE_SEALED_RESPONSE_PREFIX_BYTES
            || encoded[..4] != SOURCE_SEALED_RESPONSE_MAGIC
        {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let version = encoded[4];
        let ephemeral_public_key = fixed::<32>(&encoded[5..37])?;
        let nonce = fixed::<24>(&encoded[37..61])?;
        let ciphertext_len = u32::from_be_bytes(fixed::<4>(&encoded[61..65])?) as usize;
        if ciphertext_len < SOURCE_SEALED_RESPONSE_AEAD_TAG_BYTES
            || encoded.len() != SOURCE_SEALED_RESPONSE_PREFIX_BYTES + ciphertext_len
        {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let value = Self {
            version,
            ephemeral_public_key,
            nonce,
            ciphertext: encoded[65..].to_vec(),
        };
        value.validate_shape()?;
        Ok(value)
    }
}

impl fmt::Debug for AnonymousMailboxSourceSealedResponseV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnonymousMailboxSourceSealedResponseV1")
            .field("version", &self.version)
            .field("ciphertext_bytes", &self.ciphertext.len())
            .finish_non_exhaustive()
    }
}

/// Canonical terminal-bound request carrier for compact mailbox responses.
///
/// [`AnonymousMailboxRouteRequestV1::sealed_terminal_frame`] deliberately
/// remains an opaque outer field so forwarding relays cannot classify mailbox
/// operations. The terminal alone decodes this carrier after authenticating
/// the outer route request. It conveys the source reply key and a context that
/// was computed before that key existed, avoiding a route-commitment cycle.
///
/// [ANONYMOUS-MAILBOX-SOURCE-CARRIER 2026-09-02 by Codex] Do not add these
/// fields to `AnonymousMailboxRouteRequestV1`: its MemChain discriminant and
/// signed transcript are frozen. This explicit nested codec is the sole V1
/// source of terminal reply material.
pub struct AnonymousMailboxSourceTerminalCarrierV1 {
    context_commitment: [u8; 32],
    reply_public_key: [u8; 32],
    terminal_frame: Vec<u8>,
}

impl AnonymousMailboxSourceTerminalCarrierV1 {
    /// Prepares one canonical carrier and its matching one-shot source session.
    ///
    /// The returned carrier must be used as the exact opaque
    /// `sealed_terminal_frame` of a route request whose `request_id` equals
    /// `route_id`. M13C additionally binds that request id to the outer blind
    /// relay envelope route id before any terminal mutation.
    pub fn prepare(
        route_id: [u8; 16],
        target_node_id: [u8; 32],
        terminal_frame: Vec<u8>,
    ) -> Result<(Self, AnonymousMailboxSourceSealSessionV1), AnonymousMailboxProtocolError> {
        let context_commitment =
            source_terminal_context_commitment(&route_id, &target_node_id, &terminal_frame)?;
        let (reply_public_key, session) = AnonymousMailboxSourceSealSessionV1::prepare(
            route_id,
            target_node_id,
            context_commitment,
        )?;
        let carrier = Self {
            context_commitment,
            reply_public_key,
            terminal_frame,
        };
        carrier.validate_for_terminal(&route_id, &target_node_id)?;
        Ok((carrier, session))
    }

    /// Encodes one canonical carrier for the opaque outer route field.
    pub fn encode(&self) -> Result<Vec<u8>, AnonymousMailboxProtocolError> {
        if self.terminal_frame.len() > MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_REQUEST_FRAME_BYTES {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        validate_canonical_terminal_request_frame(&self.terminal_frame)?;
        if !source_reply_public_key_is_valid(&self.reply_public_key) {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let terminal_frame_len = u32::try_from(self.terminal_frame.len())
            .map_err(|_| AnonymousMailboxProtocolError::TooLarge)?;
        let encoded_len = SOURCE_TERMINAL_CARRIER_PREFIX_BYTES
            .checked_add(self.terminal_frame.len())
            .ok_or(AnonymousMailboxProtocolError::TooLarge)?;
        if encoded_len > MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        let mut encoded = Vec::with_capacity(encoded_len);
        encoded.extend_from_slice(&SOURCE_TERMINAL_CARRIER_MAGIC);
        encoded.push(SOURCE_TERMINAL_CARRIER_VERSION_V1);
        encoded.extend_from_slice(&self.context_commitment);
        encoded.extend_from_slice(&self.reply_public_key);
        encoded.extend_from_slice(&terminal_frame_len.to_le_bytes());
        encoded.extend_from_slice(&self.terminal_frame);
        Ok(encoded)
    }

    /// Decodes one carrier for the actual addressed terminal.
    ///
    /// The M13C terminal dispatcher must first authenticate the enclosing
    /// route request and require `request_id == BlindRelayEnvelope::route_id`.
    /// Its source signature then binds this carrier's reply key; the context
    /// intentionally excludes that key so source preparation stays
    /// non-circular while the source-seal KDF binds it separately.
    pub fn decode_for_terminal(
        encoded: &[u8],
        route_id: [u8; 16],
        local_target_node_id: [u8; 32],
    ) -> Result<Self, AnonymousMailboxProtocolError> {
        if encoded.len() > MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        if encoded.len() < SOURCE_TERMINAL_CARRIER_PREFIX_BYTES
            || encoded[..4] != SOURCE_TERMINAL_CARRIER_MAGIC
        {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        if encoded[4] != SOURCE_TERMINAL_CARRIER_VERSION_V1 {
            return Err(AnonymousMailboxProtocolError::UnsupportedVersion);
        }
        let context_commitment = fixed::<32>(&encoded[5..37])?;
        let reply_public_key = fixed::<32>(&encoded[37..69])?;
        let terminal_frame_len = u32::from_le_bytes(fixed::<4>(&encoded[69..73])?) as usize;
        if terminal_frame_len > MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_REQUEST_FRAME_BYTES
            || encoded.len() != SOURCE_TERMINAL_CARRIER_PREFIX_BYTES + terminal_frame_len
        {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let terminal_frame = encoded[SOURCE_TERMINAL_CARRIER_PREFIX_BYTES..].to_vec();
        let carrier = Self {
            context_commitment,
            reply_public_key,
            terminal_frame,
        };
        carrier.validate_for_terminal(&route_id, &local_target_node_id)?;
        Ok(carrier)
    }

    /// Returns the canonical request terminal frame for the terminal dispatcher.
    #[must_use]
    pub fn terminal_frame(&self) -> &[u8] {
        &self.terminal_frame
    }

    /// Returns the one-shot public key used to source-seal the terminal reply.
    #[must_use]
    pub const fn reply_public_key(&self) -> [u8; 32] {
        self.reply_public_key
    }

    /// Returns the non-circular context supplied to compact response sealing.
    #[must_use]
    pub const fn context_commitment(&self) -> [u8; 32] {
        self.context_commitment
    }

    fn validate_for_terminal(
        &self,
        route_id: &[u8; 16],
        target_node_id: &[u8; 32],
    ) -> Result<(), AnonymousMailboxProtocolError> {
        if self.terminal_frame.len() > MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_REQUEST_FRAME_BYTES {
            return Err(AnonymousMailboxProtocolError::TooLarge);
        }
        if !source_reply_public_key_is_valid(&self.reply_public_key) {
            return Err(AnonymousMailboxProtocolError::Malformed);
        }
        let expected_context =
            source_terminal_context_commitment(route_id, target_node_id, &self.terminal_frame)?;
        if self.context_commitment != expected_context {
            return Err(AnonymousMailboxProtocolError::ClaimsConflict);
        }
        Ok(())
    }
}

impl fmt::Debug for AnonymousMailboxSourceTerminalCarrierV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnonymousMailboxSourceTerminalCarrierV1")
            .field("terminal_frame_bytes", &self.terminal_frame.len())
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
    /// Kind 5.
    TicketIssue(AnonymousMailboxTicketIssueV1),
    /// Kind 129.
    LeaseCreateResponse(AnonymousMailboxTerminalResponseV1),
    /// Kind 130.
    PutResponse(AnonymousMailboxTerminalResponseV1),
    /// Kind 131.
    PullOneResponse(AnonymousMailboxTerminalResponseV1),
    /// Kind 132.
    AckResponse(AnonymousMailboxTerminalResponseV1),
    /// Kind 133.
    TicketIssueResponse(AnonymousMailboxTicketIssueResponseV1),
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
            Self::TicketIssue(_) => 5,
            Self::LeaseCreateResponse(_) => 129,
            Self::PutResponse(_) => 130,
            Self::PullOneResponse(_) => 131,
            Self::AckResponse(_) => 132,
            Self::TicketIssueResponse(_) => 133,
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
        AnonymousMailboxTerminalFrameV1::TicketIssueResponse(_) => {
            Some(AnonymousMailboxOperationV1::TicketIssue)
        }
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
        AnonymousMailboxTerminalFrameV1::TicketIssue(value) => {
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
        AnonymousMailboxTerminalFrameV1::TicketIssueResponse(value) => {
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
        5 => AnonymousMailboxTerminalFrameV1::TicketIssue(decode_body(body)?),
        129 => AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(decode_body(body)?),
        130 => AnonymousMailboxTerminalFrameV1::PutResponse(decode_body(body)?),
        131 => AnonymousMailboxTerminalFrameV1::PullOneResponse(decode_body(body)?),
        132 => AnonymousMailboxTerminalFrameV1::AckResponse(decode_body(body)?),
        133 => AnonymousMailboxTerminalFrameV1::TicketIssueResponse(decode_body(body)?),
        _ => return Err(AnonymousMailboxProtocolError::UnsupportedOperation),
    };
    if encode_anonymous_mailbox_terminal_frame(&frame)? != encoded {
        return Err(AnonymousMailboxProtocolError::Malformed);
    }
    Ok(frame)
}

fn validate_canonical_terminal_request_frame(
    encoded: &[u8],
) -> Result<(), AnonymousMailboxProtocolError> {
    let frame = decode_anonymous_mailbox_terminal_frame(encoded)?;
    match frame {
        AnonymousMailboxTerminalFrameV1::LeaseCreate(_)
        | AnonymousMailboxTerminalFrameV1::Put(_)
        | AnonymousMailboxTerminalFrameV1::PullOne(_)
        | AnonymousMailboxTerminalFrameV1::Ack(_)
        | AnonymousMailboxTerminalFrameV1::TicketIssue(_) => Ok(()),
        // [ANONYMOUS-MAILBOX-SOURCE-CARRIER 2026-09-02 by Codex] A source
        // carrier is terminal input only. Admitting a response kind here would
        // let an untrusted relay payload bypass M13C's request-only dispatcher.
        AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(_)
        | AnonymousMailboxTerminalFrameV1::PutResponse(_)
        | AnonymousMailboxTerminalFrameV1::PullOneResponse(_)
        | AnonymousMailboxTerminalFrameV1::AckResponse(_)
        | AnonymousMailboxTerminalFrameV1::TicketIssueResponse(_) => {
            Err(AnonymousMailboxProtocolError::UnsupportedOperation)
        }
    }
}

fn decode_body<T: DeserializeOwned>(body: &[u8]) -> Result<T, AnonymousMailboxProtocolError> {
    decode_bincode_bounded(body, BODY_BYTES, TrailingBytesPolicy::Reject)
        .map_err(|_| AnonymousMailboxProtocolError::Malformed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;

    use crate::protocol::memchain::{decode_memchain, encode_memchain, MemChainMessage};

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

    fn ticket_issue_fixture() -> (AnonymousMailboxTicketIssueV1, IdentityKeyPair) {
        let target = IdentityKeyPair::from_bytes(&[0x31; 32]).expect("target");
        for nonce in 0..u64::MAX {
            let request = AnonymousMailboxTicketIssueV1::new(
                [0x71; 16],
                [0x72; 16],
                target.public_key_bytes(),
                [0x73; 32],
                1_800_000_000,
                1_800_000_300,
                nonce,
            )
            .expect("ticket issue");
            if leading_zero_bits(&request.proof_digest().expect("work")) >= 4 {
                return (request, target);
            }
        }
        unreachable!("a four-bit proof must be reachable")
    }

    fn request_terminal_frames() -> (
        IdentityKeyPair,
        Vec<(
            AnonymousMailboxTerminalFrameV1,
            AnonymousMailboxOperationV1,
            [u8; 16],
            [u8; 32],
        )>,
    ) {
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
        let frames = vec![
            (
                AnonymousMailboxTerminalFrameV1::LeaseCreate(lease.clone()),
                AnonymousMailboxOperationV1::LeaseCreate,
                lease.admission.ticket_id,
                lease.request_commitment().expect("lease commitment"),
            ),
            (
                AnonymousMailboxTerminalFrameV1::Put(put.clone()),
                AnonymousMailboxOperationV1::Put,
                put.item_id,
                put.request_commitment().expect("put commitment"),
            ),
            (
                AnonymousMailboxTerminalFrameV1::PullOne(pull.clone()),
                AnonymousMailboxOperationV1::PullOne,
                pull.request_id,
                pull.request_commitment().expect("pull commitment"),
            ),
            (
                AnonymousMailboxTerminalFrameV1::Ack(ack.clone()),
                AnonymousMailboxOperationV1::Ack,
                ack.request_id,
                ack.request_commitment().expect("ack commitment"),
            ),
        ];
        (target, frames)
    }

    struct RoutedPullFixture {
        route_request: AnonymousMailboxRouteRequestV1,
        pull_request: AnonymousMailboxPullOneV1,
        terminal_frame: Vec<u8>,
        sealed_terminal_response: Vec<u8>,
        route_response: AnonymousMailboxRouteResponseV1,
        terminal: IdentityKeyPair,
        session: AnonymousMailboxSourceSealSessionV1,
    }

    fn pull_result_fixture() -> AnonymousMailboxPullResultV1 {
        AnonymousMailboxPullResultV1::new([0x11; 16], vec![0x33, 0x44], vec![0xaa, 0xbb, 0xcc])
            .expect("pull result")
    }

    fn routed_pull_fixture(pull_result: &AnonymousMailboxPullResultV1) -> RoutedPullFixture {
        let source = IdentityKeyPair::from_bytes(&[0x91; 32]).expect("source");
        let terminal = IdentityKeyPair::from_bytes(&[0x92; 32]).expect("terminal");
        let reader = IdentityKeyPair::from_bytes(&[0x93; 32]).expect("reader");
        let route_request = AnonymousMailboxRouteRequestV1::signed(
            [0x94; 16],
            terminal.public_key_bytes(),
            vec![0xa5; 96],
            1_800_000_000,
            &source,
        )
        .expect("route request");
        let route_commitment = route_request
            .request_commitment()
            .expect("route commitment");
        let (reply_public_key, session) = AnonymousMailboxSourceSealSessionV1::prepare(
            route_request.request_id,
            terminal.public_key_bytes(),
            route_commitment,
        )
        .expect("session");
        assert_eq!(session.reply_public_key(), reply_public_key);
        let pull_request = AnonymousMailboxPullOneV1::new(
            [0xb1; 32],
            [0xb2; 16],
            vec![0xc3; 12],
            1_800_000_001,
            &reader,
        )
        .expect("pull");
        let terminal_response = AnonymousMailboxTerminalResponseV1::signed(
            AnonymousMailboxOperationV1::PullOne,
            pull_request.request_id,
            pull_request.request_commitment().expect("pull commitment"),
            AnonymousMailboxOutcomeV1::Accepted,
            pull_result.encode().expect("pull result bytes"),
            1_800_000_002,
            &terminal,
        )
        .expect("terminal response");
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::PullOneResponse(terminal_response),
        )
        .expect("terminal frame");
        let sealed_terminal_response = AnonymousMailboxSourceSealedResponseV1::seal(
            route_request.request_id,
            route_commitment,
            terminal.public_key_bytes(),
            reply_public_key,
            &terminal_frame,
            &terminal,
        )
        .expect("seal")
        .encode()
        .expect("encode");
        let route_response = AnonymousMailboxRouteResponseV1::signed(
            &route_request,
            AnonymousMailboxOutcomeV1::Accepted,
            sealed_terminal_response.clone(),
            1_800_000_003,
            &terminal,
        )
        .expect("route response");
        RoutedPullFixture {
            route_request,
            pull_request,
            terminal_frame,
            sealed_terminal_response,
            route_response,
            terminal,
            session,
        }
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
    fn all_ten_terminal_kinds_roundtrip_canonically() {
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
        let (ticket_issue, ticket_target) = ticket_issue_fixture();
        let issued_ticket = AnonymousMailboxAdmissionTicketV1::issue(
            ticket_issue.ticket_id,
            ticket_issue.lease_claims_commitment,
            ticket_issue.issued_at,
            ticket_issue.expires_at,
            &ticket_target,
        )
        .expect("issued ticket");
        let ticket_response = AnonymousMailboxTicketIssueResponseV1::signed(
            &ticket_issue,
            AnonymousMailboxOutcomeV1::Accepted,
            Some(issued_ticket),
            1_800_000_010,
            &ticket_target,
        )
        .expect("ticket response");
        let frames = vec![
            AnonymousMailboxTerminalFrameV1::LeaseCreate(lease.clone()),
            AnonymousMailboxTerminalFrameV1::Put(put.clone()),
            AnonymousMailboxTerminalFrameV1::PullOne(pull.clone()),
            AnonymousMailboxTerminalFrameV1::Ack(ack.clone()),
            AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_issue),
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
            AnonymousMailboxTerminalFrameV1::TicketIssueResponse(ticket_response),
        ];
        let kinds = [1, 2, 3, 4, 5, 129, 130, 131, 132, 133];
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
    fn ticket_issue_transcript_is_target_bound_and_canonical() {
        let (request, target) = ticket_issue_fixture();
        request
            .verify_for_target(&target.public_key_bytes(), 1_800_000_100, 4)
            .expect("valid target-bound proof");
        assert_eq!(
            hex::encode(request.request_commitment().expect("commitment")),
            "ddcdf5fc2eb795deae806d3c14e27420d88a7496a64d84dbf948081c2a4bba89"
        );

        let mut changed_target = request.clone();
        changed_target.target_node_id = IdentityKeyPair::from_bytes(&[0x34; 32])
            .expect("other target")
            .public_key_bytes();
        assert!(changed_target
            .verify_for_target(&target.public_key_bytes(), 1_800_000_100, 4)
            .is_err());
        let mut changed_nonce = request.clone();
        changed_nonce.proof_nonce = changed_nonce.proof_nonce.wrapping_add(1);
        assert_ne!(
            changed_nonce.proof_digest().expect("changed work"),
            request.proof_digest().expect("original work")
        );

        let encoded = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::TicketIssue(request),
        )
        .expect("encode");
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(decode_anonymous_mailbox_terminal_frame(&trailing).is_err());
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
    fn source_terminal_carrier_roundtrips_all_request_kinds() {
        let (target, frames) = request_terminal_frames();
        for (index, (request_frame, operation, request_id, request_commitment)) in
            frames.into_iter().enumerate()
        {
            let route_id = [0x91 + index as u8; 16];
            let terminal_frame =
                encode_anonymous_mailbox_terminal_frame(&request_frame).expect("request frame");
            let (carrier, mut session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
                route_id,
                target.public_key_bytes(),
                terminal_frame.clone(),
            )
            .expect("prepare carrier");
            let encoded = carrier.encode().expect("encode carrier");
            let terminal_carrier = AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &encoded,
                route_id,
                target.public_key_bytes(),
            )
            .expect("terminal decode");
            assert_eq!(terminal_carrier.terminal_frame(), terminal_frame);
            assert_eq!(
                terminal_carrier.context_commitment(),
                carrier.context_commitment()
            );
            assert_eq!(
                terminal_carrier.reply_public_key(),
                carrier.reply_public_key()
            );

            let terminal_response = AnonymousMailboxTerminalResponseV1::signed(
                operation,
                request_id,
                request_commitment,
                AnonymousMailboxOutcomeV1::Accepted,
                vec![0xc1 + index as u8; 16],
                1_800_000_100 + index as u64,
                &target,
            )
            .expect("terminal response");
            let response_frame = match operation {
                AnonymousMailboxOperationV1::LeaseCreate => {
                    AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(terminal_response)
                }
                AnonymousMailboxOperationV1::Put => {
                    AnonymousMailboxTerminalFrameV1::PutResponse(terminal_response)
                }
                AnonymousMailboxOperationV1::PullOne => {
                    AnonymousMailboxTerminalFrameV1::PullOneResponse(terminal_response)
                }
                AnonymousMailboxOperationV1::Ack => {
                    AnonymousMailboxTerminalFrameV1::AckResponse(terminal_response)
                }
                // Ticket issuance has a dedicated response transcript that
                // carries the signed admission ticket, so it is covered by
                // the terminal-codec control rather than this generic
                // compact-response fixture.
                AnonymousMailboxOperationV1::TicketIssue => {
                    unreachable!("specialized ticket response")
                }
            };
            let response_frame =
                encode_anonymous_mailbox_terminal_frame(&response_frame).expect("response frame");
            let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
                route_id,
                terminal_carrier.context_commitment(),
                target.public_key_bytes(),
                terminal_carrier.reply_public_key(),
                &response_frame,
                &target,
            )
            .expect("seal response")
            .encode()
            .expect("encode response");
            assert_eq!(
                session.open(&sealed).expect("open response"),
                response_frame
            );
        }
    }

    #[test]
    fn source_terminal_carrier_rejects_mismatches_and_non_requests() {
        let (target, mut frames) = request_terminal_frames();
        let (request_frame, _, _, _) = frames.remove(0);
        let route_id = [0xa1; 16];
        let terminal_frame =
            encode_anonymous_mailbox_terminal_frame(&request_frame).expect("request frame");
        let (carrier, _) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            terminal_frame,
        )
        .expect("prepare");
        let encoded = carrier.encode().expect("encode");
        assert_eq!(carrier.encode().expect("exact retry"), encoded);
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &encoded,
                [0xa2; 16],
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxProtocolError::ClaimsConflict)
        ));
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &encoded, route_id, [0xa3; 32],
            ),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
                | Err(AnonymousMailboxProtocolError::ClaimsConflict)
        ));

        let mut context_tampered = encoded.clone();
        context_tampered[5] ^= 1;
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &context_tampered,
                route_id,
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxProtocolError::ClaimsConflict)
        ));
        let mut key_tampered = encoded.clone();
        key_tampered[37..69].fill(0);
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &key_tampered,
                route_id,
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxProtocolError::Malformed)
        ));
        let mut frame_tampered = encoded.clone();
        *frame_tampered.last_mut().expect("terminal frame") ^= 1;
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &frame_tampered,
                route_id,
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxProtocolError::Malformed)
                | Err(AnonymousMailboxProtocolError::ClaimsConflict)
        ));
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &trailing,
                route_id,
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxProtocolError::Malformed)
        ));
        let mut wrong_length = encoded.clone();
        wrong_length[69..73].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &wrong_length,
                route_id,
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxProtocolError::Malformed)
        ));
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::decode_for_terminal(
                &vec![0; MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES + 1],
                route_id,
                target.public_key_bytes(),
            ),
            Err(AnonymousMailboxProtocolError::TooLarge)
        ));

        let response = AnonymousMailboxTerminalResponseV1::signed(
            AnonymousMailboxOperationV1::LeaseCreate,
            [0xa4; 16],
            [0xa5; 32],
            AnonymousMailboxOutcomeV1::Accepted,
            vec![0xa6; 8],
            1_800_000_000,
            &target,
        )
        .expect("response");
        let response_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response),
        )
        .expect("response frame");
        assert!(matches!(
            AnonymousMailboxSourceTerminalCarrierV1::prepare(
                route_id,
                target.public_key_bytes(),
                response_frame,
            ),
            Err(AnonymousMailboxProtocolError::UnsupportedOperation)
        ));
    }

    #[test]
    fn source_terminal_carrier_outer_signature_binds_reply_key() {
        let source = IdentityKeyPair::from_bytes(&[0xb1; 32]).expect("source");
        let (target, mut frames) = request_terminal_frames();
        let (frame, _, _, _) = frames.remove(1);
        let route_id = [0xb2; 16];
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(&frame).expect("frame");
        let (carrier, _) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            terminal_frame,
        )
        .expect("prepare");
        let mut request = AnonymousMailboxRouteRequestV1::signed(
            route_id,
            target.public_key_bytes(),
            carrier.encode().expect("carrier"),
            1_800_000_000,
            &source,
        )
        .expect("route");
        request.sealed_terminal_frame[37] ^= 1;
        assert_eq!(
            request.verify_from(
                &source.public_key_bytes(),
                &target.public_key_bytes(),
                1_800_000_000,
            ),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
        );
    }

    #[test]
    fn source_terminal_carrier_source_session_is_one_shot_and_restartable() {
        let (target, mut frames) = request_terminal_frames();
        let (frame, _, _, _) = frames.remove(2);
        let route_id = [0xc1; 16];
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(&frame).expect("frame");
        let (carrier, mut session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            terminal_frame,
        )
        .expect("prepare");
        let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
            route_id,
            carrier.context_commitment(),
            target.public_key_bytes(),
            carrier.reply_public_key(),
            b"terminal response",
            &target,
        )
        .expect("seal")
        .encode()
        .expect("encode");
        assert_eq!(
            session.open(&sealed).expect("first open"),
            b"terminal response"
        );
        assert_eq!(
            session.open(&sealed),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let (carrier, session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            target.public_key_bytes(),
            encode_anonymous_mailbox_terminal_frame(&frames.remove(0).0).expect("frame"),
        )
        .expect("restart prepare");
        let restart = session.encode_restart_state().expect("restart state");
        let mut restored =
            AnonymousMailboxSourceSealSessionV1::decode_restart_state(restart.as_bytes())
                .expect("restore");
        let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
            route_id,
            carrier.context_commitment(),
            target.public_key_bytes(),
            carrier.reply_public_key(),
            b"restored response",
            &target,
        )
        .expect("seal restored")
        .encode()
        .expect("encode restored");
        assert_eq!(
            restored.open(&sealed).expect("restored open"),
            b"restored response"
        );
    }

    #[test]
    fn source_terminal_carrier_codec_golden_is_frozen() {
        let (target, mut frames) = request_terminal_frames();
        let terminal_frame =
            encode_anonymous_mailbox_terminal_frame(&frames.remove(3).0).expect("frame");
        let secret = StaticSecret::from([0xd1; 32]);
        let carrier = AnonymousMailboxSourceTerminalCarrierV1 {
            context_commitment: [0xd2; 32],
            reply_public_key: X25519PublicKey::from(&secret).to_bytes(),
            terminal_frame,
        };
        let encoded = carrier.encode().expect("encode");
        assert_eq!(&encoded[..5], b"AMST\x01");
        assert_eq!(
            hex::encode(Sha256::digest(&encoded)),
            "312061de86d71bcb2a24373cce438ba37e259cb378534789ab174cb342be1d70"
        );
        assert!(source_reply_public_key_is_valid(
            &carrier.reply_public_key()
        ));
        assert_ne!(target.public_key_bytes(), [0; 32]);
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

    #[test]
    fn pull_result_v1_wire_layout_golden_is_frozen() {
        let mut encoded = Vec::with_capacity(ANONYMOUS_MAILBOX_PULL_RESULT_BYTES);
        encoded.extend_from_slice(&PULL_RESULT_MAGIC);
        encoded.extend_from_slice(&[0x11; 16]);
        encoded.extend_from_slice(&[0x22; 32]);
        encoded.extend_from_slice(&3u32.to_le_bytes());
        encoded.extend_from_slice(&2u16.to_le_bytes());
        encoded.extend_from_slice(&[0x33, 0x44]);
        encoded.resize(58 + MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES, 0);
        encoded.extend_from_slice(&[0xaa, 0xbb, 0xcc]);
        encoded.resize(ANONYMOUS_MAILBOX_PULL_RESULT_BYTES, 0);
        assert_eq!(encoded.len(), ANONYMOUS_MAILBOX_PULL_RESULT_BYTES);
        assert_eq!(
            hex::encode(Sha256::digest(&encoded)),
            "6104e3a1e464cff1c757364fadf6c07852732958b5b18a6b405570427c546826"
        );
        assert_eq!(
            hex::encode(&encoded[..24]),
            "414d01011111111111111111111111111111111122222222"
        );
        assert_eq!(hex::encode(&encoded[52..64]), "030000000200334400000000");
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&encoded),
            Err(AnonymousMailboxProtocolError::ClaimsConflict)
        );
    }

    #[test]
    fn pull_result_v1_valid_roundtrip_is_canonical() {
        let fixture = pull_result_fixture();
        let encoded = fixture.encode().expect("encode");
        let decoded = AnonymousMailboxPullResultV1::decode(&encoded).expect("decode");
        assert_eq!(decoded, fixture);
        assert_eq!(decoded.encode().expect("re-encode"), encoded);
    }

    #[test]
    fn pull_result_v1_rejects_noncanonical_padding_lengths_and_commitment() {
        let encoded = pull_result_fixture().encode().expect("encode");

        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&encoded[..encoded.len() - 1]),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let mut bad = encoded.clone();
        bad.push(0);
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&bad),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let mut bad = encoded.clone();
        bad[52..56].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&bad),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let mut bad = encoded.clone();
        bad[52..56].copy_from_slice(
            &(u32::try_from(MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES + 1).expect("oversize"))
                .to_le_bytes(),
        );
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&bad),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let mut bad = encoded.clone();
        bad[56..58].copy_from_slice(&257u16.to_le_bytes());
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&bad),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let mut bad = encoded.clone();
        bad[60] = 1;
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&bad),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let mut bad = encoded.clone();
        *bad.last_mut().expect("last") = 1;
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&bad),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        let mut bad = encoded;
        bad[20] ^= 1;
        assert_eq!(
            AnonymousMailboxPullResultV1::decode(&bad),
            Err(AnonymousMailboxProtocolError::ClaimsConflict)
        );
    }

    #[test]
    fn terminal_response_payload_cap_stays_at_160_kib() {
        let responder = IdentityKeyPair::from_bytes(&[0xa1; 32]).expect("responder");
        AnonymousMailboxTerminalResponseV1::signed(
            AnonymousMailboxOperationV1::PullOne,
            [0xa2; 16],
            [0xa3; 32],
            AnonymousMailboxOutcomeV1::Accepted,
            vec![0xa4; MAX_ANONYMOUS_MAILBOX_TERMINAL_RESPONSE_PAYLOAD_BYTES],
            1_800_000_010,
            &responder,
        )
        .expect("max response payload");
        assert!(matches!(
            AnonymousMailboxTerminalResponseV1::signed(
                AnonymousMailboxOperationV1::PullOne,
                [0xa2; 16],
                [0xa3; 32],
                AnonymousMailboxOutcomeV1::Accepted,
                vec![0xa4; MAX_ANONYMOUS_MAILBOX_TERMINAL_RESPONSE_PAYLOAD_BYTES + 1],
                1_800_000_010,
                &responder,
            ),
            Err(AnonymousMailboxProtocolError::TooLarge)
        ));
    }

    #[test]
    fn compact_source_seal_roundtrip_is_single_use_and_restartable() {
        let pull_result = pull_result_fixture();
        let fixture = routed_pull_fixture(&pull_result);
        fixture
            .route_response
            .verify_for_request(&fixture.route_request, &fixture.terminal.public_key_bytes())
            .expect("outer response");

        let mut session = fixture.session;
        let opened = session
            .open(&fixture.sealed_terminal_response)
            .expect("first open");
        assert_eq!(opened, fixture.terminal_frame);
        match decode_anonymous_mailbox_terminal_frame(&opened).expect("terminal decode") {
            AnonymousMailboxTerminalFrameV1::PullOneResponse(response) => {
                response
                    .verify_for_request(
                        AnonymousMailboxOperationV1::PullOne,
                        &fixture.pull_request.request_id,
                        &fixture
                            .pull_request
                            .request_commitment()
                            .expect("pull commitment"),
                        &fixture.terminal.public_key_bytes(),
                    )
                    .expect("response verify");
                assert_eq!(
                    response.sealed_payload,
                    pull_result.encode().expect("payload")
                );
            }
            other => panic!("unexpected terminal frame kind: {}", other.kind()),
        }
        assert_eq!(
            session.open(&fixture.sealed_terminal_response),
            Err(AnonymousMailboxProtocolError::Malformed)
        );

        {
            let fixture = routed_pull_fixture(&pull_result);
            let restart = fixture.session.encode_restart_state().expect("restart");
            let mut restored =
                AnonymousMailboxSourceSealSessionV1::decode_restart_state(restart.as_bytes())
                    .expect("restore");
            let opened = restored
                .open(&fixture.sealed_terminal_response)
                .expect("restore open");
            assert_eq!(opened, fixture.terminal_frame);
        };
    }

    #[test]
    fn compact_source_seal_rejects_context_substitution_and_tamper() {
        let pull_result = pull_result_fixture();
        let fixture = routed_pull_fixture(&pull_result);
        let route_commitment = fixture
            .route_request
            .request_commitment()
            .expect("route commitment");

        let mut wrong_route = AnonymousMailboxSourceSealSessionV1::prepare(
            [0xd1; 16],
            fixture.terminal.public_key_bytes(),
            route_commitment,
        )
        .expect("wrong route")
        .1;
        assert_eq!(
            wrong_route.open(&fixture.sealed_terminal_response),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
        );

        let mut wrong_request = AnonymousMailboxSourceSealSessionV1::prepare(
            fixture.route_request.request_id,
            fixture.terminal.public_key_bytes(),
            [0xd2; 32],
        )
        .expect("wrong request")
        .1;
        assert_eq!(
            wrong_request.open(&fixture.sealed_terminal_response),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
        );

        let wrong_terminal = IdentityKeyPair::from_bytes(&[0xd3; 32]).expect("wrong terminal");
        let mut wrong_target = AnonymousMailboxSourceSealSessionV1::prepare(
            fixture.route_request.request_id,
            wrong_terminal.public_key_bytes(),
            route_commitment,
        )
        .expect("wrong target")
        .1;
        assert_eq!(
            wrong_target.open(&fixture.sealed_terminal_response),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
        );

        let mut wrong_key = AnonymousMailboxSourceSealSessionV1::prepare(
            fixture.route_request.request_id,
            fixture.terminal.public_key_bytes(),
            route_commitment,
        )
        .expect("wrong key")
        .1;
        assert_eq!(
            wrong_key.open(&fixture.sealed_terminal_response),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
        );

        let mut tampered = fixture.sealed_terminal_response.clone();
        *tampered.last_mut().expect("ciphertext") ^= 1;
        let mut tampered_session = routed_pull_fixture(&pull_result).session;
        assert_eq!(
            tampered_session.open(&tampered),
            Err(AnonymousMailboxProtocolError::SignatureRejected)
        );

        let mut bad_version = fixture.sealed_terminal_response.clone();
        bad_version[4] = 0;
        assert_eq!(
            AnonymousMailboxSourceSealedResponseV1::decode(&bad_version),
            Err(AnonymousMailboxProtocolError::UnsupportedVersion)
        );

        let route_commitment = fixture
            .route_request
            .request_commitment()
            .expect("route commitment");
        let (reply_public_key, _) = AnonymousMailboxSourceSealSessionV1::prepare(
            fixture.route_request.request_id,
            fixture.terminal.public_key_bytes(),
            route_commitment,
        )
        .expect("reply key");
        assert_eq!(
            AnonymousMailboxSourceSealedResponseV1::seal(
                fixture.route_request.request_id,
                route_commitment,
                fixture.terminal.public_key_bytes(),
                reply_public_key,
                &fixture.terminal_frame,
                &wrong_terminal,
            ),
            Err(AnonymousMailboxProtocolError::ClaimsConflict)
        );
    }

    #[test]
    fn compact_source_seal_and_outer_route_fit_frozen_sizes() {
        let item = vec![0xee; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES];
        let cursor = vec![0xdd; MAX_ANONYMOUS_MAILBOX_CURSOR_BYTES];
        let pull_result =
            AnonymousMailboxPullResultV1::new([0xcc; 16], cursor, item).expect("max pull result");
        let pull_result_bytes = pull_result.encode().expect("pull result bytes");
        assert_eq!(pull_result_bytes.len(), 163_130);

        let fixture = routed_pull_fixture(&pull_result);
        assert_eq!(fixture.terminal_frame.len(), 163_307);
        assert_eq!(fixture.sealed_terminal_response.len(), 163_388);

        let encoded_route_payload = bincode::serialize(
            &MemChainMessage::AnonymousMailboxRouteResponseV1(fixture.route_response.clone()),
        )
        .expect("memchain route response payload");
        match decode_memchain(&encoded_route_payload).expect("decode route response") {
            MemChainMessage::AnonymousMailboxRouteResponseV1(value) => {
                value
                    .verify_for_request(
                        &fixture.route_request,
                        &fixture.terminal.public_key_bytes(),
                    )
                    .expect("decoded route response verify");
            }
            _ => panic!("unexpected MemChain variant"),
        }
        assert_eq!(encoded_route_payload.len(), 163_557);
        assert_eq!(STANDARD.encode(&encoded_route_payload).len(), 218_076);

        let encoded_route = encode_memchain(&MemChainMessage::AnonymousMailboxRouteResponseV1(
            fixture.route_response.clone(),
        ))
        .expect("memchain route response");
        assert_eq!(&encoded_route[1..], encoded_route_payload.as_slice());
        assert_eq!(encoded_route.len(), 163_558);
    }

    #[test]
    fn compact_source_seal_rejects_max_plus_one_payload() {
        let terminal = IdentityKeyPair::from_bytes(&[0xe1; 32]).expect("terminal");
        let (reply_public_key, _) = AnonymousMailboxSourceSealSessionV1::prepare(
            [0xe2; 16],
            terminal.public_key_bytes(),
            [0xe3; 32],
        )
        .expect("reply");
        let max_payload = vec![
            0x44;
            MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES
                - SOURCE_SEALED_RESPONSE_OVERHEAD_BYTES
        ];
        AnonymousMailboxSourceSealedResponseV1::seal(
            [0xe2; 16],
            [0xe3; 32],
            terminal.public_key_bytes(),
            reply_public_key,
            &max_payload,
            &terminal,
        )
        .expect("exact max payload");
        let plus_one = vec![
            0x44;
            MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES
                - SOURCE_SEALED_RESPONSE_OVERHEAD_BYTES
                + 1
        ];
        assert_eq!(
            AnonymousMailboxSourceSealedResponseV1::seal(
                [0xe2; 16],
                [0xe3; 32],
                terminal.public_key_bytes(),
                reply_public_key,
                &plus_one,
                &terminal,
            ),
            Err(AnonymousMailboxProtocolError::TooLarge)
        );
    }
}
