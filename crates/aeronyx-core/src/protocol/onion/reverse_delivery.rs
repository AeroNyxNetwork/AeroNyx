// ============================================
// File: crates/aeronyx-core/src/protocol/onion/reverse_delivery.rs
// ============================================
//! Reverse onion adjacent-hop delivery V1, not a queue or transport runtime.
//!
//! [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] A private immediate recipient
//! polls a public relay. Source/previous-hop enqueue admission is separate.
//! Existing onion/envelope wire, source-sealed replies and SSRF remain unchanged.
//! Durable queue CAS, quotas, transport admission and source verification remain
//! mandatory integration boundaries; these pure types perform no I/O.
//! [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Recipient lease proof
//! authenticates relay execution authority, never the hidden source route.
//! [SOURCE-EVIDENCE-V1 2026-10-04 by Codex] Additive source-signed read-only
//! evidence queries and relay-signed bounded parts; no queue/network authority.

use sha2::{Digest, Sha256};
use thiserror::Error;

use super::is_onion_blob;
use crate::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use crate::protocol::chat::{
    decode_blind_relay_envelope, encode_blind_relay_envelope, BlindRelayEnvelope,
};
use crate::protocol::onion_reply::{decode_onion_sealed_response, MAX_ONION_SEALED_RESPONSE_BYTES};

// [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] Separate additive wire domain;
// no existing envelope, route discriminant, endpoint policy, or reply changes.
const REVERSE_ONION_MAGIC: [u8; 4] = *b"AXRD";
const REVERSE_ONION_SIGN_DOMAIN: &[u8] = b"AeroNyx-Reverse-Onion-Sign-v1\0";
const REVERSE_ONION_COMMIT_DOMAIN: &[u8] = b"AeroNyx-Reverse-Onion-Frame-v1\0";
const REVERSE_ONION_HEADER_BYTES: usize = 170;
const REVERSE_ONION_SIGNATURE_BYTES: usize = 64;
// [REVERSE-ONION-BOUND-EXPORT 2026-10-04 by Codex] Export the existing
// envelope bound as the single named source of truth for queue envelopes;
// this is an additive API name only and does not alter the V1 wire.
pub const MAX_REVERSE_ONION_ENVELOPE_BYTES: usize = 256 * 1024;

/// Strict claim lifetime; clocks are supplied by callers, never read here.
pub const REVERSE_ONION_CLAIM_LIFETIME_SECS: u64 = 30;
/// Matches the existing blind-relay envelope age ceiling, without extending it.
pub const REVERSE_ONION_ENVELOPE_LIFETIME_SECS: u64 = 600;
/// Recovery-only window after the immutable execution deadline. This never
/// authorizes a new delivery, lease, peel, execution, or source reply session.
pub const REVERSE_ONION_RESULT_RETENTION_SECS: u64 = 300;
/// Outer carrier bound, not an increase to any inner envelope/reply ceiling.
pub const MAX_REVERSE_ONION_FRAME_BYTES: usize = REVERSE_ONION_HEADER_BYTES
    + REVERSE_ONION_SIGNATURE_BYTES
    + MAX_ONION_SEALED_RESPONSE_BYTES;
/// Exact fixed Claim frame bound, derived from the existing canonical header
/// and signature sizes. Lease/Result use `MAX_REVERSE_ONION_FRAME_BYTES`.
pub const MAX_REVERSE_ONION_CLAIM_BYTES: usize =
    REVERSE_ONION_HEADER_BYTES + REVERSE_ONION_SIGNATURE_BYTES;

/// Frozen additive frame codes. A result is NOT an execution acknowledgement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReverseOnionKindV1 {
    /// Recipient authenticates one bounded poll at one adjacent relay.
    Claim = 1,
    /// Relay binds one unchanged signed envelope to that claim.
    Lease = 2,
    /// Recipient returns only an opaque fixed-class source reply.
    Result = 3,
}

/// Coarse errors deliberately carry no identity, route, payload, or path.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ReverseOnionError {
    #[error("reverse delivery rejected")]
    Rejected,
    #[error("reverse delivery expired")]
    Expired,
    #[error("reverse delivery conflict")]
    Conflict,
    #[error("reverse delivery transition rejected")]
    InvalidTransition,
}

/// Private-field canonical adjacent-hop carrier. Intentionally has no Debug.
///
/// [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] Wire: AXRD[4], version=1,
/// kind[1], relay[32], immediate_recipient[32], claim_id[16], route_id[16],
/// lease_id[16], parent_commitment[32], issued_at[8], expires_at[8],
/// payload_length[4], payload[length], signature[64]. Integers are big-endian.
/// Signature covers SIGN_DOMAIN || SHA256(all preceding bytes). Commitment
/// covers COMMIT_DOMAIN || the entire signed canonical frame.
///
/// Only adjacent identities are visible; the deeper terminal remains inside
/// the onion. A Result's outer signature authenticates its immediate sender,
/// NOT execution or the hidden terminal. Sources must use
/// `OnionReplySession::prepare_source_sealed` and consuming `open` verification;
/// no relay can infer the encrypted proof mode from a sealed response.
pub struct ReverseOnionFrameV1 {
    kind: ReverseOnionKindV1,
    relay: [u8; 32],
    recipient: [u8; 32],
    claim_id: [u8; 16],
    route_id: [u8; 16],
    lease_id: [u8; 16],
    parent_commitment: [u8; 32],
    issued_at: u64,
    expires_at: u64,
    payload: Vec<u8>,
    signature: [u8; 64],
}

/// [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Borrowed, private-field
/// proof of a pinned relay's execution lease. No scalar deadline constructor,
/// serialization, or Debug. It is not a source-route proof or durable arm token.
/// A retained proof must still pass the journal's fresh execution check.
pub struct VerifiedRecipientLease<'a> {
    lease: &'a ReverseOnionFrameV1,
}

impl VerifiedRecipientLease<'_> {
    pub fn lease(&self) -> &ReverseOnionFrameV1 { self.lease }
    pub fn relay_execution_expiry(&self) -> u64 { self.lease.expires_at }
}

impl ReverseOnionFrameV1 {
    /// [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Fresh acquisition
    /// only. Claim freshness is checked at lease issuance, not response arrival.
    /// R admission and S verification must separately enforce source deadlines.
    /// This API deliberately accepts no caller-provided execution deadline.
    pub fn verify_recipient_lease(
        &self,
        claim: &Self,
        pinned_relay: [u8; 32],
        local_recipient: [u8; 32],
        now: u64,
    ) -> Result<VerifiedRecipientLease<'_>, ReverseOnionError> {
        if self.relay != pinned_relay || self.recipient != local_recipient
            || self.kind != ReverseOnionKindV1::Lease || self.claim_id == [0; 16]
            || self.route_id == [0; 16] || self.lease_id == [0; 16]
            || self.issued_at == 0
        { return Err(ReverseOnionError::Rejected); }
        claim.verify_claim(pinned_relay, local_recipient, self.issued_at)?;
        self.verify_at(now)?;
        // The equal bound here represents R authority only, NOT independent
        // source-route evidence. Legacy R/S verifier semantics stay unchanged.
        self.verify_lease_binding(claim, self.expires_at)?;
        Ok(VerifiedRecipientLease { lease: self })
    }

    /// Sign a one-item poll. Does not allocate queue capacity or grant custody.
    pub fn claim(
        relay: [u8; 32],
        claim_id: [u8; 16],
        issued_at: u64,
        expires_at: u64,
        recipient: &IdentityKeyPair,
    ) -> Result<Self, ReverseOnionError> {
        Self {
            kind: ReverseOnionKindV1::Claim,
            relay,
            recipient: recipient.public_key_bytes(),
            claim_id,
            route_id: [0; 16],
            lease_id: [0; 16],
            parent_commitment: [0; 32],
            issued_at,
            expires_at,
            payload: Vec::new(),
            signature: [0; 64],
        }
        .signed(recipient)
    }

    /// Sign a lease for an exact envelope previously signed by this relay.
    /// `route_deadline` must come from already authenticated route admission;
    /// it must not be supplied or extended by an unauthenticated poller.
    /// Persist this exact frame under the route's immutable envelope binding
    /// before returning any bytes; retries must not call this constructor again.
    pub fn lease(
        claim: &Self,
        envelope: &BlindRelayEnvelope,
        lease_id: [u8; 16],
        route_deadline: u64,
        now: u64,
        relay: &IdentityKeyPair,
    ) -> Result<Self, ReverseOnionError> {
        claim.verify_claim(relay.public_key_bytes(), claim.recipient, now)?;
        let envelope_deadline = envelope
            .timestamp
            .checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS)
            .ok_or(ReverseOnionError::Rejected)?;
        let frame = Self {
            kind: ReverseOnionKindV1::Lease,
            relay: claim.relay,
            recipient: claim.recipient,
            claim_id: claim.claim_id,
            route_id: envelope.route_id,
            lease_id,
            parent_commitment: claim.commitment(),
            issued_at: now,
            expires_at: route_deadline.min(envelope_deadline),
            payload: encode_blind_relay_envelope(envelope)
                .map_err(|_| ReverseOnionError::Rejected)?,
            signature: [0; 64],
        }
        .signed(relay)?;
        frame.verify_lease(claim, route_deadline, now)?;
        Ok(frame)
    }

    /// Sign only a bounded opaque reply; never attach a clear terminal receipt.
    /// Verification here is structural. Only the original source can validate
    /// the terminal proof and operation through its single-use reply session.
    /// Persist the first exact signed Result before sending it; retries must
    /// replay that byte string, not regenerate a signature/timestamp/carrier.
    pub fn result(
        claim: &Self,
        lease: &Self,
        sealed_reply: &[u8],
        route_deadline: u64,
        now: u64,
        recipient: &IdentityKeyPair,
    ) -> Result<Self, ReverseOnionError> {
        lease.verify_lease_binding(claim, route_deadline)?;
        let retain_until = lease.result_retention_deadline()?;
        if now < lease.issued_at || now >= retain_until {
            return Err(ReverseOnionError::Expired);
        }
        if recipient.public_key_bytes() != lease.recipient {
            return Err(ReverseOnionError::Rejected);
        }
        decode_onion_sealed_response(sealed_reply).map_err(|_| ReverseOnionError::Rejected)?;
        Self {
            kind: ReverseOnionKindV1::Result,
            relay: lease.relay,
            recipient: lease.recipient,
            claim_id: lease.claim_id,
            route_id: lease.route_id,
            lease_id: lease.lease_id,
            parent_commitment: lease.commitment(),
            issued_at: now,
            expires_at: retain_until,
            payload: sealed_reply.to_vec(),
            signature: [0; 64],
        }
        .signed(recipient)
    }

    /// Check the poll against independently authenticated adjacent identities.
    pub fn verify_claim(
        &self,
        expected_relay: [u8; 32],
        authenticated_recipient: [u8; 32],
        now: u64,
    ) -> Result<(), ReverseOnionError> {
        self.verify_at(now)?;
        if self.kind != ReverseOnionKindV1::Claim
            || self.relay != expected_relay
            || self.recipient != authenticated_recipient
        {
            return Err(ReverseOnionError::Rejected);
        }
        Ok(())
    }

    /// Verify the lease's exact parent and signed inner envelope before peel.
    /// The caller must first admit `claim` with `verify_claim`, not trust a key
    /// merely because it is carried inside an otherwise valid frame.
    pub fn verify_lease(
        &self,
        claim: &Self,
        route_deadline: u64,
        now: u64,
    ) -> Result<BlindRelayEnvelope, ReverseOnionError> {
        self.verify_at(now)?;
        self.verify_lease_binding(claim, route_deadline)
    }

    // Claim freshness is evaluated at ORIGINAL issuance, not at restart or
    // response arrival. The short poll credential cannot truncate recovery.
    fn verify_lease_binding(
        &self,
        claim: &Self,
        route_deadline: u64,
    ) -> Result<BlindRelayEnvelope, ReverseOnionError> {
        self.verify_signature()?;
        claim.verify_claim(self.relay, self.recipient, self.issued_at)?;
        if self.kind != ReverseOnionKindV1::Lease
            || self.claim_id != claim.claim_id
            || self.parent_commitment != claim.commitment()
            || self.issued_at < claim.issued_at
            || self.expires_at > route_deadline
        {
            return Err(ReverseOnionError::Rejected);
        }
        self.validated_envelope(self.issued_at)
    }

    /// Verify exact leased-item binding before persisting an opaque result.
    /// After execution expiry, this permits recovery only from an already
    /// persisted exact Armed lease. It does not permit execution or revive an
    /// Ambiguous/Expired phase. Durable phase CAS remains mandatory.
    pub fn verify_result(
        &self,
        claim: &Self,
        lease: &Self,
        route_deadline: u64,
        now: u64,
    ) -> Result<&[u8], ReverseOnionError> {
        lease.verify_lease_binding(claim, route_deadline)?;
        self.verify_at(now)?;
        if self.kind != ReverseOnionKindV1::Result
            || self.relay != lease.relay
            || self.recipient != lease.recipient
            || self.claim_id != lease.claim_id
            || self.route_id != lease.route_id
            || self.lease_id != lease.lease_id
            || self.parent_commitment != lease.commitment()
            || self.issued_at < lease.issued_at
            || self.expires_at != lease.result_retention_deadline()?
        {
            return Err(ReverseOnionError::Rejected);
        }
        Ok(&self.payload)
    }

    /// Canonical exact bytes, suitable for immutable persistence and replay.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = self.unsigned_bytes();
        bytes.extend_from_slice(&self.signature);
        bytes
    }

    /// Decode a currently fresh frame. Expected identity/parent verification
    /// remains mandatory; decoding alone never authorizes delivery/execution.
    pub fn decode(bytes: &[u8], now: u64) -> Result<Self, ReverseOnionError> {
        let frame = Self::decode_for_recovery(bytes)?;
        frame.verify_at(now)?;
        Ok(frame)
    }

    /// Authenticate historical bytes without treating them as fresh authority.
    /// Recovery must decode the original expired claim too; operation-specific
    /// verify methods enforce the distinct execution/result windows. Unknown
    /// versions/kinds, noncanonical nested data and trailing bytes fail closed.
    pub fn decode_for_recovery(bytes: &[u8]) -> Result<Self, ReverseOnionError> {
        if bytes.len() > MAX_REVERSE_ONION_FRAME_BYTES {
            return Err(ReverseOnionError::Rejected);
        }
        let mut cursor = ReverseOnionCursor(bytes);
        if cursor.array::<4>()? != REVERSE_ONION_MAGIC || cursor.array::<1>()? != [1] {
            return Err(ReverseOnionError::Rejected);
        }
        let kind = match cursor.array::<1>()?[0] {
            1 => ReverseOnionKindV1::Claim,
            2 => ReverseOnionKindV1::Lease,
            3 => ReverseOnionKindV1::Result,
            _ => return Err(ReverseOnionError::Rejected),
        };
        let relay = cursor.array()?;
        let recipient = cursor.array()?;
        let claim_id = cursor.array()?;
        let route_id = cursor.array()?;
        let lease_id = cursor.array()?;
        let parent_commitment = cursor.array()?;
        let issued_at = u64::from_be_bytes(cursor.array()?);
        let expires_at = u64::from_be_bytes(cursor.array()?);
        let length = u32::from_be_bytes(cursor.array()?) as usize;
        if length > Self::payload_limit(kind)
            || cursor.0.len() != length + REVERSE_ONION_SIGNATURE_BYTES
        {
            return Err(ReverseOnionError::Rejected);
        }
        let payload = cursor.take(length)?.to_vec();
        let signature = cursor.array()?;
        let frame = Self {
            kind,
            relay,
            recipient,
            claim_id,
            route_id,
            lease_id,
            parent_commitment,
            issued_at,
            expires_at,
            payload,
            signature,
        };
        frame.verify_signature()?;
        Ok(frame)
    }

    /// Digest of the entire canonical signed frame, including opaque payload.
    pub fn commitment(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(REVERSE_ONION_COMMIT_DOMAIN);
        hash.update(self.encode());
        hash.finalize().into()
    }

    /// Exact retry only: same identity but different signed bytes is conflict.
    /// This comparison does not extend expiry or authorize retransmission.
    pub fn require_exact_retry(&self, candidate: &Self) -> Result<(), ReverseOnionError> {
        if self.encode() != candidate.encode() {
            return Err(ReverseOnionError::Conflict);
        }
        Ok(())
    }

    pub fn kind(&self) -> ReverseOnionKindV1 {
        self.kind
    }

    pub fn relay(&self) -> [u8; 32] {
        self.relay
    }

    pub fn immediate_recipient(&self) -> [u8; 32] {
        self.recipient
    }

    pub fn claim_id(&self) -> [u8; 16] {
        self.claim_id
    }

    pub fn route_id(&self) -> [u8; 16] {
        self.route_id
    }

    pub fn lease_id(&self) -> [u8; 16] {
        self.lease_id
    }

    pub fn issued_at(&self) -> u64 {
        self.issued_at
    }

    /// Kind-specific bound: claim freshness, lease execution, result retention.
    pub fn expires_at(&self) -> u64 {
        self.expires_at
    }

    /// Recovery bound derived only from an immutable lease execution deadline.
    pub fn result_retention_deadline(&self) -> Result<u64, ReverseOnionError> {
        if self.kind != ReverseOnionKindV1::Lease {
            return Err(ReverseOnionError::Rejected);
        }
        self.expires_at
            .checked_add(REVERSE_ONION_RESULT_RETENTION_SECS)
            .ok_or(ReverseOnionError::Rejected)
    }

    /// Minimum retention for the immutable route binding/tombstone. Shortening
    /// a route deadline must not allow the still-fresh envelope to be re-enqueued
    /// after cleanup. This is NOT permission to serve an expired opaque result.
    pub fn replay_evidence_deadline(&self) -> Result<u64, ReverseOnionError> {
        let result_deadline = self.result_retention_deadline()?;
        let envelope = self.validated_envelope(self.issued_at)?;
        let envelope_deadline = envelope
            .timestamp
            .checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS)
            .ok_or(ReverseOnionError::Rejected)?;
        Ok(result_deadline.max(envelope_deadline))
    }

    fn payload_limit(kind: ReverseOnionKindV1) -> usize {
        match kind {
            ReverseOnionKindV1::Claim => 0,
            ReverseOnionKindV1::Lease => MAX_REVERSE_ONION_ENVELOPE_BYTES,
            ReverseOnionKindV1::Result => MAX_ONION_SEALED_RESPONSE_BYTES,
        }
    }

    fn signed(mut self, signer: &IdentityKeyPair) -> Result<Self, ReverseOnionError> {
        self.validate_shape()?;
        if signer.public_key_bytes() != self.signer() {
            return Err(ReverseOnionError::Rejected);
        }
        self.signature = signer.sign(&self.signing_data());
        Ok(self)
    }

    fn signer(&self) -> [u8; 32] {
        if self.kind == ReverseOnionKindV1::Lease {
            self.relay
        } else {
            self.recipient
        }
    }

    fn verify_at(&self, now: u64) -> Result<(), ReverseOnionError> {
        self.verify_signature()?;
        if now < self.issued_at || now >= self.expires_at {
            return Err(ReverseOnionError::Expired);
        }
        Ok(())
    }

    fn verify_signature(&self) -> Result<(), ReverseOnionError> {
        self.validate_shape()?;
        IdentityPublicKey::from_bytes(&self.signer())
            .and_then(|key| key.verify(&self.signing_data(), &self.signature))
            .map_err(|_| ReverseOnionError::Rejected)
    }

    fn validate_shape(&self) -> Result<(), ReverseOnionError> {
        let maximum_lifetime = match self.kind {
            ReverseOnionKindV1::Claim => REVERSE_ONION_CLAIM_LIFETIME_SECS,
            ReverseOnionKindV1::Lease => REVERSE_ONION_ENVELOPE_LIFETIME_SECS,
            ReverseOnionKindV1::Result => {
                REVERSE_ONION_ENVELOPE_LIFETIME_SECS + REVERSE_ONION_RESULT_RETENTION_SECS
            }
        };
        if self.payload.len() > Self::payload_limit(self.kind)
            || self.issued_at >= self.expires_at
            || self.expires_at - self.issued_at > maximum_lifetime
            || self.relay == self.recipient
        {
            return Err(ReverseOnionError::Rejected);
        }
        IdentityPublicKey::from_bytes(&self.relay).map_err(|_| ReverseOnionError::Rejected)?;
        IdentityPublicKey::from_bytes(&self.recipient).map_err(|_| ReverseOnionError::Rejected)?;
        match self.kind {
            ReverseOnionKindV1::Claim => {
                if self.route_id != [0; 16]
                    || self.lease_id != [0; 16]
                    || self.parent_commitment != [0; 32]
                {
                    return Err(ReverseOnionError::Rejected);
                }
            }
            ReverseOnionKindV1::Lease => {
                self.validated_envelope(self.issued_at)?;
                self.result_retention_deadline()?;
            }
            ReverseOnionKindV1::Result => {
                decode_onion_sealed_response(&self.payload)
                    .map_err(|_| ReverseOnionError::Rejected)?;
            }
        }
        Ok(())
    }

    fn validated_envelope(&self, now: u64) -> Result<BlindRelayEnvelope, ReverseOnionError> {
        let envelope = decode_blind_relay_envelope(&self.payload)
            .map_err(|_| ReverseOnionError::Rejected)?;
        let deadline = envelope
            .timestamp
            .checked_add(REVERSE_ONION_ENVELOPE_LIFETIME_SECS)
            .ok_or(ReverseOnionError::Rejected)?;
        if envelope.next_hop != self.recipient
            || envelope.route_id != self.route_id
            || envelope.ttl == 0
            || !is_onion_blob(&envelope.encrypted_blob)
            || envelope.timestamp > self.issued_at
            || now >= deadline
            || self.expires_at > deadline
            || encode_blind_relay_envelope(&envelope).map_err(|_| ReverseOnionError::Rejected)?
                != self.payload
        {
            return Err(ReverseOnionError::Rejected);
        }
        let relay = IdentityPublicKey::from_bytes(&self.relay)
            .map_err(|_| ReverseOnionError::Rejected)?;
        envelope
            .verify_signature_from(&relay)
            .map_err(|_| ReverseOnionError::Rejected)?;
        Ok(envelope)
    }

    fn unsigned_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(REVERSE_ONION_HEADER_BYTES + self.payload.len());
        bytes.extend_from_slice(&REVERSE_ONION_MAGIC);
        bytes.extend_from_slice(&[1, self.kind as u8]);
        bytes.extend_from_slice(&self.relay);
        bytes.extend_from_slice(&self.recipient);
        bytes.extend_from_slice(&self.claim_id);
        bytes.extend_from_slice(&self.route_id);
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.parent_commitment);
        bytes.extend_from_slice(&self.issued_at.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at.to_be_bytes());
        bytes.extend_from_slice(&(self.payload.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    fn signing_data(&self) -> Vec<u8> {
        let mut transcript = Vec::with_capacity(REVERSE_ONION_SIGN_DOMAIN.len() + 32);
        transcript.extend_from_slice(REVERSE_ONION_SIGN_DOMAIN);
        transcript.extend_from_slice(&Sha256::digest(self.unsigned_bytes()));
        transcript
    }
}

// [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] Bounds are checked before any
// attacker-selected allocation; exact remaining length excludes trailing data.
// [SOURCE-EVIDENCE-V1 2026-10-04 by Codex] Separate wire and domains: no
// changes to AXRD, onion caps, request commitments, or execution authority.
const SOURCE_QUERY_MAGIC: &[u8; 4] = b"AXRQ";
const SOURCE_EVIDENCE_MAGIC: &[u8; 4] = b"AXRE";
const SOURCE_QUERY_SIGN: &[u8] = b"AeroNyx-Reverse-Source-Query-Sign-v1\0";
const SOURCE_QUERY_COMMIT: &[u8] = b"AeroNyx-Reverse-Source-Query-Commit-v1\0";
const SOURCE_EVIDENCE_SIGN: &[u8] = b"AeroNyx-Reverse-Source-Evidence-Sign-v1\0";
pub const REVERSE_ONION_SOURCE_QUERY_BYTES: usize = 4 + 1 + 32 + 32 + 16 + 32 + 1 + 32 + 8 + 8 + 64;
pub const REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD: usize = 4 + 1 + 32 + 32 + 32 + 1 + 1 + 8 + 8 + 32 + 32 + 32 + 4 + 64;
/// New endpoint-only cap. Never apply it to existing AXRD/onion/HTTP carriers.
pub const MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES: usize =
    REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD + MAX_REVERSE_ONION_FRAME_BYTES;
pub const REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS: u64 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SourceEvidencePartV1 { Claim = 1, Lease = 2, Result = 3 }

impl SourceEvidencePartV1 {
    fn decode(value: u8) -> Result<Self, ReverseOnionError> {
        match value { 1 => Ok(Self::Claim), 2 => Ok(Self::Lease), 3 => Ok(Self::Result), _ => Err(ReverseOnionError::Rejected) }
    }
    fn limit(self) -> usize {
        match self {
            Self::Claim => MAX_REVERSE_ONION_CLAIM_BYTES,
            Self::Lease => REVERSE_ONION_HEADER_BYTES + REVERSE_ONION_SIGNATURE_BYTES + MAX_REVERSE_ONION_ENVELOPE_BYTES,
            Self::Result => MAX_REVERSE_ONION_FRAME_BYTES,
        }
    }
    fn kind(self) -> ReverseOnionKindV1 {
        match self { Self::Claim => ReverseOnionKindV1::Claim, Self::Lease => ReverseOnionKindV1::Lease, Self::Result => ReverseOnionKindV1::Result }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SourceEvidenceStateV1 { Pending = 1, Available = 2, Unavailable = 3 }

impl SourceEvidenceStateV1 {
    fn decode(value: u8) -> Result<Self, ReverseOnionError> {
        match value { 1 => Ok(Self::Pending), 2 => Ok(Self::Available), 3 => Ok(Self::Unavailable), _ => Err(ReverseOnionError::Rejected) }
    }
}

/// No Debug/serde. Original commitment is the EXISTING authenticated full
/// PeerBlindRelayRequest bincode commitment, not JSON or envelope-only hash.
pub struct ReverseOnionSourceQueryV1 {
    source: [u8; 32], relay: [u8; 32], route: [u8; 16], original_request: [u8; 32],
    part: SourceEvidencePartV1, nonce: [u8; 32], issued_at: u64, expires_at: u64,
    signature: [u8; 64],
}

/// Crypto binding only. The caller MUST obtain expected fields from the
/// original durable authenticated S admission, never from the query itself.
/// This token cannot prove database authorization or grant queue mutation.
pub struct VerifiedSourceQuery<'a> { query: &'a ReverseOnionSourceQueryV1 }

impl ReverseOnionSourceQueryV1 {
    pub fn sign(source: &IdentityKeyPair, relay: [u8; 32], route: [u8; 16],
        original_request: [u8; 32], part: SourceEvidencePartV1, nonce: [u8; 32],
        issued_at: u64, expires_at: u64) -> Result<Self, ReverseOnionError> {
        let mut query = Self { source: source.public_key_bytes(), relay, route, original_request,
            part, nonce, issued_at, expires_at, signature: [0; 64] };
        query.shape()?;
        query.signature = source.sign(&source_evidence_transcript(SOURCE_QUERY_SIGN, &query.unsigned()));
        Ok(query)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ReverseOnionError> {
        if bytes.len() != REVERSE_ONION_SOURCE_QUERY_BYTES { return Err(ReverseOnionError::Rejected); }
        let mut c = ReverseOnionCursor(bytes);
        if c.take(4)? != SOURCE_QUERY_MAGIC || c.array::<1>()? != [1] { return Err(ReverseOnionError::Rejected); }
        let query = Self { source: c.array()?, relay: c.array()?, route: c.array()?, original_request: c.array()?,
            part: SourceEvidencePartV1::decode(c.array::<1>()?[0])?, nonce: c.array()?,
            issued_at: u64::from_be_bytes(c.array()?), expires_at: u64::from_be_bytes(c.array()?), signature: c.array()? };
        query.verify_signature()?;
        Ok(query)
    }

    pub fn verify_binding(&self, source: [u8; 32], relay: [u8; 32], route: [u8; 16],
        original_request: [u8; 32], now: u64) -> Result<VerifiedSourceQuery<'_>, ReverseOnionError> {
        self.verify_at(now)?;
        if self.source != source || self.relay != relay || self.route != route || self.original_request != original_request {
            return Err(ReverseOnionError::Rejected);
        }
        Ok(VerifiedSourceQuery { query: self })
    }

    pub fn source(&self) -> [u8; 32] { self.source }
    pub fn relay(&self) -> [u8; 32] { self.relay }
    pub fn route_id(&self) -> [u8; 16] { self.route }
    pub fn original_request_commitment(&self) -> [u8; 32] { self.original_request }
    pub fn part(&self) -> SourceEvidencePartV1 { self.part }
    pub fn nonce(&self) -> [u8; 32] { self.nonce }
    pub fn issued_at(&self) -> u64 { self.issued_at }
    pub fn expires_at(&self) -> u64 { self.expires_at }
    pub fn encode(&self) -> Vec<u8> { let mut out = self.unsigned(); out.extend_from_slice(&self.signature); out }
    pub fn commitment(&self) -> [u8; 32] {
        let mut h = Sha256::new(); h.update(SOURCE_QUERY_COMMIT); h.update(self.encode()); h.finalize().into()
    }
    fn unsigned(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(REVERSE_ONION_SOURCE_QUERY_BYTES - 64);
        out.extend_from_slice(SOURCE_QUERY_MAGIC); out.push(1);
        out.extend_from_slice(&self.source); out.extend_from_slice(&self.relay); out.extend_from_slice(&self.route);
        out.extend_from_slice(&self.original_request); out.push(self.part as u8); out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&self.issued_at.to_be_bytes()); out.extend_from_slice(&self.expires_at.to_be_bytes()); out
    }
    fn shape(&self) -> Result<(), ReverseOnionError> {
        source_evidence_window(self.issued_at, self.expires_at)?;
        if self.source == [0; 32] || self.relay == [0; 32] || self.source == self.relay
            || self.route == [0; 16] || self.original_request == [0; 32] || self.nonce == [0; 32] {
            return Err(ReverseOnionError::Rejected);
        }
        IdentityPublicKey::from_bytes(&self.source).map_err(|_| ReverseOnionError::Rejected)?;
        IdentityPublicKey::from_bytes(&self.relay).map_err(|_| ReverseOnionError::Rejected)?;
        Ok(())
    }
    fn verify_signature(&self) -> Result<(), ReverseOnionError> {
        self.shape()?;
        source_evidence_verify(self.source, SOURCE_QUERY_SIGN, &self.unsigned(), &self.signature)
    }
    fn verify_at(&self, now: u64) -> Result<(), ReverseOnionError> {
        self.verify_signature()?;
        if now < self.issued_at || now >= self.expires_at { return Err(ReverseOnionError::Expired); }
        Ok(())
    }
}

/// No endpoint/terminal/source metadata beyond adjacent R and the query hash.
/// Available commits to the SAME complete immutable durable snapshot on all
/// three parts. Core validates signatures/linkage, but cannot attest fsync.
pub struct ReverseOnionSourceEvidenceV1 {
    relay: [u8; 32], query_commitment: [u8; 32], nonce: [u8; 32], part: SourceEvidencePartV1,
    state: SourceEvidenceStateV1, issued_at: u64, expires_at: u64,
    commitments: [[u8; 32]; 3], payload: Vec<u8>, signature: [u8; 64],
}

/// Only produced after fresh query/response verification. Retaining a proof
/// after query expiry does not extend execution or Result retention windows.
pub struct VerifiedSourceEvidencePart<'a> {
    query: &'a ReverseOnionSourceQueryV1,
    response: &'a ReverseOnionSourceEvidenceV1,
}

impl VerifiedSourceEvidencePart<'_> {
    pub fn state(&self) -> SourceEvidenceStateV1 { self.response.state }
}

impl ReverseOnionSourceEvidenceV1 {
    pub fn pending(query: &VerifiedSourceQuery<'_>, now: u64, expires_at: u64, relay: &IdentityKeyPair)
        -> Result<Self, ReverseOnionError> {
        Self::signed(query, SourceEvidenceStateV1::Pending, [[0; 32]; 3], Vec::new(), now, expires_at, relay)
    }
    /// A signed unavailability statement is NOT authenticated absence of an
    /// earlier effect, and MUST NOT authorize another route/lease/execution.
    pub fn unavailable(query: &VerifiedSourceQuery<'_>, now: u64, expires_at: u64, relay: &IdentityKeyPair)
        -> Result<Self, ReverseOnionError> {
        Self::signed(query, SourceEvidenceStateV1::Unavailable, [[0; 32]; 3], Vec::new(), now, expires_at, relay)
    }
    /// R must load the complete original-S-authorized durable chain. Arbitrary
    /// caller hashes/frames do not bypass full signature/linkage verification.
    /// `admitted_deadline` is R's immutable authenticated admission bound.
    pub fn available(query: &VerifiedSourceQuery<'_>, claim: &ReverseOnionFrameV1,
        lease: &ReverseOnionFrameV1, result: &ReverseOnionFrameV1, admitted_deadline: u64,
        now: u64, expires_at: u64, relay: &IdentityKeyPair) -> Result<Self, ReverseOnionError> {
        verify_source_chain(query.query.relay, query.query.route, claim, lease, result, admitted_deadline, now)?;
        let frames = [claim, lease, result];
        let commitments = [claim.commitment(), lease.commitment(), result.commitment()];
        let payload = frames[query.query.part as usize - 1].encode();
        Self::signed(query, SourceEvidenceStateV1::Available, commitments, payload, now, expires_at, relay)
    }
    fn signed(query: &VerifiedSourceQuery<'_>, state: SourceEvidenceStateV1,
        commitments: [[u8; 32]; 3], payload: Vec<u8>, now: u64, expires_at: u64,
        relay: &IdentityKeyPair) -> Result<Self, ReverseOnionError> {
        query.query.verify_at(now)?;
        if relay.public_key_bytes() != query.query.relay || expires_at > query.query.expires_at {
            return Err(ReverseOnionError::Rejected);
        }
        let mut response = Self { relay: relay.public_key_bytes(), query_commitment: query.query.commitment(),
            nonce: query.query.nonce, part: query.query.part, state, issued_at: now, expires_at,
            commitments, payload, signature: [0; 64] };
        response.shape()?;
        response.signature = relay.sign(&source_evidence_transcript(SOURCE_EVIDENCE_SIGN, &response.unsigned()));
        Ok(response)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ReverseOnionError> {
        if bytes.len() < REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD || bytes.len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
            return Err(ReverseOnionError::Rejected);
        }
        let mut c = ReverseOnionCursor(bytes);
        if c.take(4)? != SOURCE_EVIDENCE_MAGIC || c.array::<1>()? != [1] { return Err(ReverseOnionError::Rejected); }
        let relay = c.array()?; let query_commitment = c.array()?; let nonce = c.array()?;
        let part = SourceEvidencePartV1::decode(c.array::<1>()?[0])?;
        let state = SourceEvidenceStateV1::decode(c.array::<1>()?[0])?;
        let issued_at = u64::from_be_bytes(c.array()?); let expires_at = u64::from_be_bytes(c.array()?);
        let commitments = [c.array()?, c.array()?, c.array()?];
        let length = u32::from_be_bytes(c.array()?) as usize;
        // All size/state admission precedes allocation AND signature work.
        source_evidence_payload_admission(part, state, length, bytes.len())?;
        source_evidence_window(issued_at, expires_at)?;
        let payload = c.take(length)?; let signature = c.array()?;
        source_evidence_verify(relay, SOURCE_EVIDENCE_SIGN, &bytes[..bytes.len() - 64], &signature)?;
        let response = Self { relay, query_commitment, nonce, part, state, issued_at, expires_at,
            commitments, payload: payload.to_vec(), signature };
        response.shape()?;
        Ok(response)
    }

    pub fn verify_for_query<'a>(&'a self, query: &'a ReverseOnionSourceQueryV1, now: u64)
        -> Result<VerifiedSourceEvidencePart<'a>, ReverseOnionError> {
        query.verify_at(now)?; self.shape()?;
        source_evidence_verify(self.relay, SOURCE_EVIDENCE_SIGN, &self.unsigned(), &self.signature)?;
        if self.relay != query.relay || self.query_commitment != query.commitment() || self.nonce != query.nonce
            || self.part != query.part || self.issued_at < query.issued_at || self.expires_at > query.expires_at
        { return Err(ReverseOnionError::Rejected); }
        if now < self.issued_at || now >= self.expires_at { return Err(ReverseOnionError::Expired); }
        Ok(VerifiedSourceEvidencePart { query, response: self })
    }
    pub fn state(&self) -> SourceEvidenceStateV1 { self.state }
    pub fn encode(&self) -> Vec<u8> { let mut out = self.unsigned(); out.extend_from_slice(&self.signature); out }
    fn unsigned(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD - 64 + self.payload.len());
        out.extend_from_slice(SOURCE_EVIDENCE_MAGIC); out.push(1); out.extend_from_slice(&self.relay);
        out.extend_from_slice(&self.query_commitment); out.extend_from_slice(&self.nonce);
        out.extend_from_slice(&[self.part as u8, self.state as u8]);
        out.extend_from_slice(&self.issued_at.to_be_bytes()); out.extend_from_slice(&self.expires_at.to_be_bytes());
        for commitment in &self.commitments { out.extend_from_slice(commitment); }
        out.extend_from_slice(&(self.payload.len() as u32).to_be_bytes()); out.extend_from_slice(&self.payload); out
    }
    fn shape(&self) -> Result<(), ReverseOnionError> {
        source_evidence_window(self.issued_at, self.expires_at)?;
        source_evidence_payload_admission(self.part, self.state, self.payload.len(),
            REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD.checked_add(self.payload.len()).ok_or(ReverseOnionError::Rejected)?)?;
        if self.query_commitment == [0; 32] || self.nonce == [0; 32] { return Err(ReverseOnionError::Rejected); }
        IdentityPublicKey::from_bytes(&self.relay).map_err(|_| ReverseOnionError::Rejected)?;
        if self.state == SourceEvidenceStateV1::Available {
            if self.commitments.iter().any(|c| *c == [0; 32]) { return Err(ReverseOnionError::Rejected); }
            let frame = ReverseOnionFrameV1::decode_for_recovery(&self.payload)?;
            if frame.kind() != self.part.kind() || frame.relay() != self.relay || frame.encode() != self.payload
                || frame.commitment() != self.commitments[self.part as usize - 1]
            { return Err(ReverseOnionError::Rejected); }
        } else if self.commitments != [[0; 32]; 3] { return Err(ReverseOnionError::Rejected); }
        Ok(())
    }
}

/// Full adjacent proof only, never a source-sealed terminal result. The source
/// must still compare its pre-send retained-envelope expectation, then consume
/// its own reply session. Query freshness is checked when collecting parts;
/// execution deadline and Result retention are independently checked here.
pub struct VerifiedSourceEvidenceChain {
    claim: ReverseOnionFrameV1, lease: ReverseOnionFrameV1, result: ReverseOnionFrameV1,
}
impl VerifiedSourceEvidenceChain {
    pub fn claim(&self) -> &ReverseOnionFrameV1 { &self.claim }
    pub fn lease(&self) -> &ReverseOnionFrameV1 { &self.lease }
    pub fn result(&self) -> &ReverseOnionFrameV1 { &self.result }
    pub fn verify(parts: [&VerifiedSourceEvidencePart<'_>; 3], source_admitted_deadline: u64, now: u64)
        -> Result<Self, ReverseOnionError> {
        let first = parts[0];
        for (index, part) in parts.iter().enumerate() {
            if part.response.state != SourceEvidenceStateV1::Available || part.response.part as usize != index + 1
                || part.query.source != first.query.source || part.query.relay != first.query.relay
                || part.query.route != first.query.route || part.query.original_request != first.query.original_request
                || part.response.commitments != first.response.commitments
            { return Err(ReverseOnionError::Conflict); }
        }
        let chain = Self { claim: ReverseOnionFrameV1::decode_for_recovery(&parts[0].response.payload)?,
            lease: ReverseOnionFrameV1::decode_for_recovery(&parts[1].response.payload)?,
            result: ReverseOnionFrameV1::decode_for_recovery(&parts[2].response.payload)? };
        verify_source_chain(first.query.relay, first.query.route, &chain.claim, &chain.lease, &chain.result, source_admitted_deadline, now)?;
        Ok(chain)
    }
}

fn verify_source_chain(relay: [u8; 32], route: [u8; 16], claim: &ReverseOnionFrameV1,
    lease: &ReverseOnionFrameV1, result: &ReverseOnionFrameV1, deadline: u64, now: u64) -> Result<(), ReverseOnionError> {
    if lease.relay() != relay || lease.route_id() != route || route == [0; 16]
        || lease.claim_id() == [0; 16] || lease.lease_id() == [0; 16]
    { return Err(ReverseOnionError::Rejected); }
    claim.verify_claim(relay, lease.immediate_recipient(), lease.issued_at())?;
    lease.verify_lease(claim, deadline, lease.issued_at())?;
    result.verify_result(claim, lease, deadline, now)?;
    Ok(())
}

fn source_evidence_payload_admission(part: SourceEvidencePartV1, state: SourceEvidenceStateV1,
    length: usize, total: usize) -> Result<(), ReverseOnionError> {
    if length > part.limit() || REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD.checked_add(length) != Some(total)
        || total > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES
        || (state != SourceEvidenceStateV1::Available && length != 0)
        || (state == SourceEvidenceStateV1::Available && length < MAX_REVERSE_ONION_CLAIM_BYTES)
    { return Err(ReverseOnionError::Rejected); }
    Ok(())
}

fn source_evidence_window(issued: u64, expires: u64) -> Result<(), ReverseOnionError> {
    if issued == 0 || expires <= issued || issued.checked_add(REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS).is_none()
        || expires - issued > REVERSE_ONION_SOURCE_QUERY_LIFETIME_SECS
    { return Err(ReverseOnionError::Rejected); }
    Ok(())
}

fn source_evidence_transcript(domain: &[u8], unsigned: &[u8]) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(domain.len() + 32);
    transcript.extend_from_slice(domain); transcript.extend_from_slice(&Sha256::digest(unsigned)); transcript
}
fn source_evidence_verify(key: [u8; 32], domain: &[u8], unsigned: &[u8], signature: &[u8; 64])
    -> Result<(), ReverseOnionError> {
    IdentityPublicKey::from_bytes(&key).and_then(|key| key.verify(&source_evidence_transcript(domain, unsigned), signature))
        .map_err(|_| ReverseOnionError::Rejected)
}

struct ReverseOnionCursor<'a>(&'a [u8]);

impl<'a> ReverseOnionCursor<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], ReverseOnionError> {
        if length > self.0.len() {
            return Err(ReverseOnionError::Rejected);
        }
        let (head, tail) = self.0.split_at(length);
        self.0 = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], ReverseOnionError> {
        self.take(N)?
            .try_into()
            .map_err(|_| ReverseOnionError::Rejected)
    }
}

/// Repository transition vocabulary; not a durable queue implementation.
///
/// [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] Queue uniqueness must bind
/// (relay, route_id) to exact canonical signed envelope bytes, recipient, TTL
/// and admitted deadline. Same route/different bytes is Conflict. The claim
/// and lease are immutable once selected; CAS binds their commitments and the
/// authenticated claimant. Persist Armed BEFORE the first possible write.
/// On restart Armed can only replay the SAME lease bytes before execution expiry;
/// never mint a new lease, change a recipient, rewrap, or reroute. Ambiguous
/// is terminal, not evidence of failure or permission for another effect.
///
/// Consumers must atomically compare expected phase AND immutable binding,
/// then persist the accepted next phase. No enum value proves a persisted CAS.
/// Expiry never deletes evidence; retention/cleanup require separate policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReverseOnionPhaseV1 {
    Queued,
    Armed,
    ResultAvailable,
    Ambiguous,
    Expired,
}

/// Bound transition proposal, not a durable authorization token. No Debug.
///
/// [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] Every returned next state must
/// be committed using the caller's previous state and immutable lease digest
/// as CAS expectations BEFORE acting. Reconstructing `queued` on restart is
/// forbidden: retain the durable phase and use `restore` after row integrity
/// validation. There is deliberately no arbitrary `set_phase` operation.
#[derive(Clone)]
pub struct ReverseOnionDeliveryStateV1 {
    phase: ReverseOnionPhaseV1,
    lease_commitment: [u8; 32],
    execution_not_before: u64,
    execution_deadline: u64,
    retention_deadline: u64,
}

impl ReverseOnionDeliveryStateV1 {
    /// For the initial immutable lease insertion only, not restart recovery.
    pub fn queued(
        claim: &ReverseOnionFrameV1,
        lease: &ReverseOnionFrameV1,
        route_deadline: u64,
        now: u64,
    ) -> Result<Self, ReverseOnionError> {
        lease.verify_lease(claim, route_deadline, now)?;
        Self::restore(claim, lease, route_deadline, ReverseOnionPhaseV1::Queued)
    }

    /// Restore ONLY a phase read from an integrity-checked durable row, never
    /// a peer-supplied phase. Historical verification grants no fresh execution.
    pub fn restore(
        claim: &ReverseOnionFrameV1,
        lease: &ReverseOnionFrameV1,
        route_deadline: u64,
        persisted_phase: ReverseOnionPhaseV1,
    ) -> Result<Self, ReverseOnionError> {
        lease.verify_lease_binding(claim, route_deadline)?;
        Ok(Self {
            phase: persisted_phase,
            lease_commitment: lease.commitment(),
            execution_not_before: lease.issued_at,
            execution_deadline: lease.expires_at,
            retention_deadline: lease.result_retention_deadline()?,
        })
    }

    pub fn phase(&self) -> ReverseOnionPhaseV1 {
        self.phase
    }

    /// CAS binding for the repository. Never log it or expose it as a locator.
    pub fn lease_commitment(&self) -> [u8; 32] {
        self.lease_commitment
    }

    /// Propose the before-send durable barrier. This method performs no write.
    pub fn arm(&self, now: u64) -> Result<Self, ReverseOnionError> {
        if self.phase != ReverseOnionPhaseV1::Queued {
            return Err(ReverseOnionError::InvalidTransition);
        }
        if now < self.execution_not_before || now >= self.execution_deadline {
            return Err(ReverseOnionError::Expired);
        }
        Ok(self.with_phase(ReverseOnionPhaseV1::Armed))
    }

    /// Exact bytes only, after the caller has confirmed durable Armed. After
    /// execution expiry this rejects even though result recovery remains open.
    pub fn exact_replay_bytes(
        &self,
        lease: &ReverseOnionFrameV1,
        now: u64,
    ) -> Result<Vec<u8>, ReverseOnionError> {
        if self.phase != ReverseOnionPhaseV1::Armed {
            return Err(ReverseOnionError::InvalidTransition);
        }
        if lease.commitment() != self.lease_commitment {
            return Err(ReverseOnionError::Conflict);
        }
        lease.verify_at(now)?;
        Ok(lease.encode())
    }

    /// Accept only an exact-bound opaque response from an existing Armed row.
    /// May run during recovery grace; never permits another network execution.
    pub fn accept_result(
        &self,
        claim: &ReverseOnionFrameV1,
        lease: &ReverseOnionFrameV1,
        result: &ReverseOnionFrameV1,
        route_deadline: u64,
        now: u64,
    ) -> Result<Self, ReverseOnionError> {
        if self.phase != ReverseOnionPhaseV1::Armed {
            return Err(ReverseOnionError::InvalidTransition);
        }
        if lease.commitment() != self.lease_commitment {
            return Err(ReverseOnionError::Conflict);
        }
        result.verify_result(claim, lease, route_deadline, now)?;
        Ok(self.with_phase(ReverseOnionPhaseV1::ResultAvailable))
    }

    /// Terminal uncertainty, never permission for a new lease or route. A
    /// transport awaiting exact recovery may instead retain its Armed record.
    pub fn mark_ambiguous(&self) -> Result<Self, ReverseOnionError> {
        if self.phase != ReverseOnionPhaseV1::Armed {
            return Err(ReverseOnionError::InvalidTransition);
        }
        Ok(self.with_phase(ReverseOnionPhaseV1::Ambiguous))
    }

    /// Expire Queued at execution expiry; Armed only after result grace ends.
    /// This is a state transition, NOT permission to delete replay evidence.
    pub fn expire(&self, now: u64) -> Result<Self, ReverseOnionError> {
        let deadline = match self.phase {
            ReverseOnionPhaseV1::Queued => self.execution_deadline,
            ReverseOnionPhaseV1::Armed => self.retention_deadline,
            _ => return Err(ReverseOnionError::InvalidTransition),
        };
        if now < deadline {
            return Err(ReverseOnionError::InvalidTransition);
        }
        Ok(self.with_phase(ReverseOnionPhaseV1::Expired))
    }

    fn with_phase(&self, phase: ReverseOnionPhaseV1) -> Self {
        Self {
            phase,
            ..self.clone()
        }
    }
}

// [REVERSE-ONION-RECOVERY-TESTS 2026-10-04 by Codex] Authored only; execution
// is deferred. Fixed clocks/identity seeds, no network or filesystem fixtures.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::onion::{build_onion_envelope, OnionHop};
    use crate::protocol::onion_reply::{
        encode_onion_sealed_response, seal_onion_reply, OnionReplySession,
        OnionSealedResponse, ONION_REPLY_RESPONSE_SIZE_CLASSES,
    };

    const NOW: u64 = 1_800_000_000;

    struct Fixture {
        relay: IdentityKeyPair,
        recipient: IdentityKeyPair,
        claim: ReverseOnionFrameV1,
        lease: ReverseOnionFrameV1,
    }

    impl Fixture {
        fn new() -> Self {
            let relay = IdentityKeyPair::from_bytes(&[11; 32]).unwrap();
            let recipient = IdentityKeyPair::from_bytes(&[22; 32]).unwrap();
            let claim = ReverseOnionFrameV1::claim(
                relay.public_key_bytes(), [1; 16], NOW, NOW + 30, &recipient,
            ).unwrap();
            let (_, kem) = recipient.to_x25519();
            let envelope = build_onion_envelope(
                &[OnionHop { node_id: recipient.public_key_bytes(), kem_pub: kem.to_bytes() }],
                b"opaque request", [2; 16], 1, NOW, &relay,
            ).unwrap();
            let lease = ReverseOnionFrameV1::lease(
                &claim, &envelope, [3; 16], NOW + 600, NOW, &relay,
            ).unwrap();
            Self { relay, recipient, claim, lease }
        }

        fn result(&self, now: u64) -> (ReverseOnionFrameV1, OnionReplySession) {
            let (request, session) = OnionReplySession::prepare_source_sealed(
                self.lease.route_id(), self.recipient.public_key_bytes(),
                ONION_REPLY_RESPONSE_SIZE_CLASSES[0], b"operation".to_vec(),
            ).unwrap();
            let sealed = seal_onion_reply(self.lease.route_id(), &request, b"result", &self.recipient).unwrap();
            let result = ReverseOnionFrameV1::result(
                &self.claim, &self.lease, &encode_onion_sealed_response(&sealed).unwrap(),
                NOW + 600, now, &self.recipient,
            ).unwrap();
            (result, session)
        }
    }

    #[test]
    fn claim_layout_codes_and_exact_roundtrip_are_frozen() {
        let f = Fixture::new();
        let bytes = f.claim.encode();
        assert_eq!(bytes.len(), 234);
        assert_eq!(&bytes[..6], b"AXRD\x01\x01");
        assert_eq!(&bytes[6..38], &f.relay.public_key_bytes());
        assert_eq!(&bytes[38..70], &f.recipient.public_key_bytes());
        assert_eq!(&bytes[70..86], &[1; 16]);
        assert_eq!(&bytes[86..150], &[0; 64]);
        assert_eq!(&bytes[150..158], &NOW.to_be_bytes());
        assert_eq!(&bytes[158..166], &(NOW + 30).to_be_bytes());
        assert_eq!(&bytes[166..170], &[0; 4]);
        let restored = ReverseOnionFrameV1::decode(&bytes, NOW).unwrap();
        assert!(restored.require_exact_retry(&f.claim).is_ok());
        assert_eq!(restored.commitment(), f.claim.commitment());
        assert_eq!(f.lease.encode()[5], 2);
        assert_eq!(f.result(NOW + 1).0.encode()[5], 3);
    }

    // [RECIPIENT-LEASE-AUTHORITY 2026-10-04 by Codex] Authored, unexecuted.
    #[test]
    fn recipient_authority_is_pinned_fresh_and_not_source_deadline_evidence() {
        let f = Fixture::new();
        let proof = f.lease.verify_recipient_lease(&f.claim,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW + 31).unwrap();
        assert_eq!(proof.relay_execution_expiry(), f.lease.expires_at());
        assert_eq!(proof.lease().encode(), f.lease.encode());
        assert!(f.lease.verify_recipient_lease(&f.claim,
            f.recipient.public_key_bytes(), f.relay.public_key_bytes(), NOW).is_err());
        assert!(f.lease.verify_recipient_lease(&f.claim,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW + 600).is_err());
        // Relay authority must not weaken independently enforced R/S bounds.
        assert!(f.lease.verify_lease(&f.claim, NOW + 599, NOW).is_err());
        let (result, _) = f.result(NOW + 601);
        assert!(result.verify_result(&f.claim, &f.lease, NOW + 600, NOW + 601).is_ok());
    }

    #[test]
    fn recipient_proof_rejects_wrong_parent_signature_and_zero_identifiers() {
        let f = Fixture::new();
        let other = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(),
            [9; 16], NOW, NOW + 30, &f.recipient).unwrap();
        assert!(f.lease.verify_recipient_lease(&other,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW).is_err());
        let mut bad = ReverseOnionFrameV1::decode_for_recovery(&f.lease.encode()).unwrap();
        bad.signature[0] ^= 1;
        assert!(bad.verify_recipient_lease(&f.claim,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW).is_err());
        let mut zero = ReverseOnionFrameV1::decode_for_recovery(&f.lease.encode()).unwrap();
        zero.lease_id = [0; 16];
        let zero = zero.signed(&f.relay).unwrap();
        assert!(zero.verify_recipient_lease(&f.claim,
            f.relay.public_key_bytes(), f.recipient.public_key_bytes(), NOW).is_err());
    }

    #[test]
    fn malformed_version_kind_length_trailing_and_signature_fail_closed() {
        let f = Fixture::new();
        let original = f.claim.encode();
        for end in 0..original.len() {
            assert!(ReverseOnionFrameV1::decode_for_recovery(&original[..end]).is_err());
        }
        for offset in [0, 4, 5, 6, 38, 70, 166, 233] {
            let mut changed = original.clone();
            changed[offset] ^= 0xff;
            assert!(ReverseOnionFrameV1::decode_for_recovery(&changed).is_err());
        }
        let mut trailing = original.clone();
        trailing.push(0);
        assert!(ReverseOnionFrameV1::decode_for_recovery(&trailing).is_err());
        assert!(ReverseOnionFrameV1::decode_for_recovery(&vec![0; MAX_REVERSE_ONION_FRAME_BYTES + 1]).is_err());
    }

    #[test]
    fn authenticated_substitution_and_self_relay_are_rejected() {
        let f = Fixture::new();
        let other = ReverseOnionFrameV1::claim(
            f.relay.public_key_bytes(), [4; 16], NOW, NOW + 30, &f.recipient,
        ).unwrap();
        assert!(f.lease.verify_lease(&other, NOW + 600, NOW).is_err());
        assert_eq!(f.claim.require_exact_retry(&other).err(), Some(ReverseOnionError::Conflict));
        assert!(f.claim.verify_claim(f.recipient.public_key_bytes(), f.relay.public_key_bytes(), NOW).is_err());
        assert!(ReverseOnionFrameV1::claim(
            f.recipient.public_key_bytes(), [1; 16], NOW, NOW + 30, &f.recipient,
        ).is_err());
        assert!(ReverseOnionFrameV1::claim(
            f.relay.public_key_bytes(), [1; 16], NOW, NOW + 31, &f.recipient,
        ).is_err());
        let mut envelope = f.lease.verify_lease(&f.claim, NOW + 600, NOW).unwrap();
        envelope.next_hop = f.relay.public_key_bytes();
        let envelope = envelope.sign_with(&f.relay);
        assert!(ReverseOnionFrameV1::lease(&f.claim, &envelope, [3; 16], NOW + 600, NOW, &f.relay).is_err());
    }

    #[test]
    fn historical_claim_does_not_truncate_execution_or_result_grace() {
        let f = Fixture::new();
        assert!(ReverseOnionFrameV1::decode(&f.claim.encode(), NOW + 30).is_err());
        let historical = ReverseOnionFrameV1::decode_for_recovery(&f.claim.encode()).unwrap();
        assert!(f.lease.verify_lease(&historical, NOW + 600, NOW + 599).is_ok());
        assert!(f.lease.verify_lease(&historical, NOW + 600, NOW + 600).is_err());
        let (result, source) = f.result(NOW + 610);
        let bytes = result.verify_result(&historical, &f.lease, NOW + 600, NOW + 899).unwrap();
        assert_eq!(source.open(bytes).unwrap().payload.as_slice(), b"result");
        assert!(result.verify_result(&historical, &f.lease, NOW + 600, NOW + 900).is_err());
        assert!(ReverseOnionFrameV1::decode_for_recovery(&result.encode()).is_ok());
    }

    #[test]
    fn result_parent_substitution_requires_more_than_a_valid_sender_signature() {
        let f = Fixture::new();
        let (result, _) = f.result(NOW + 1);
        let mut altered = ReverseOnionFrameV1::decode_for_recovery(&result.encode()).unwrap();
        altered.parent_commitment[0] ^= 1;
        let altered = altered.signed(&f.recipient).unwrap();
        assert!(ReverseOnionFrameV1::decode(&altered.encode(), NOW + 1).is_ok());
        assert!(altered.verify_result(&f.claim, &f.lease, NOW + 600, NOW + 1).is_err());
        assert!(f.lease.verify_lease(&f.claim, NOW + 599, NOW).is_err());
    }

    #[test]
    fn maximum_reply_carrier_fits_without_expanding_inner_caps() {
        let f = Fixture::new();
        // Structural opaque carrier only, not a terminal-execution proof.
        let opaque = OnionSealedResponse {
            version: 1, ephemeral_public_key: [9; 32], nonce: [0; 24],
            ciphertext: vec![0; ONION_REPLY_RESPONSE_SIZE_CLASSES[3] + 16],
        };
        let encoded = encode_onion_sealed_response(&opaque).unwrap();
        assert_eq!(encoded.len(), MAX_ONION_SEALED_RESPONSE_BYTES);
        let result = ReverseOnionFrameV1::result(&f.claim, &f.lease, &encoded, NOW + 600, NOW, &f.recipient).unwrap();
        assert_eq!(result.encode().len(), MAX_REVERSE_ONION_FRAME_BYTES);
        assert!(ReverseOnionFrameV1::decode(&result.encode(), NOW).is_ok());
        let mut oversized = encoded;
        oversized.push(0);
        assert!(ReverseOnionFrameV1::result(&f.claim, &f.lease, &oversized, NOW + 600, NOW, &f.recipient).is_err());
    }

    // [SOURCE-EVIDENCE-V1 2026-10-04 by Codex] Authored / unexecuted.
    fn evidence_query(f: &Fixture, part: SourceEvidencePartV1, now: u64) -> ReverseOnionSourceQueryV1 {
        let source = IdentityKeyPair::from_bytes(&[33; 32]).unwrap();
        ReverseOnionSourceQueryV1::sign(&source, f.relay.public_key_bytes(), f.lease.route_id(),
            [9; 32], part, [part as u8; 32], now, now + 20).unwrap()
    }

    fn query_authority<'a>(f: &Fixture, query: &'a ReverseOnionSourceQueryV1, now: u64) -> VerifiedSourceQuery<'a> {
        let source = IdentityKeyPair::from_bytes(&[33; 32]).unwrap();
        query.verify_binding(source.public_key_bytes(), f.relay.public_key_bytes(),
            f.lease.route_id(), [9; 32], now).unwrap()
    }

    #[test]
    fn source_evidence_frozen_layout_domains_and_query_golden() {
        assert_eq!(REVERSE_ONION_SOURCE_QUERY_BYTES, 230);
        assert_eq!(REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD, 283);
        assert_eq!(MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES, 279126);
        assert_eq!([SourceEvidencePartV1::Claim as u8, SourceEvidencePartV1::Lease as u8, SourceEvidencePartV1::Result as u8], [1, 2, 3]);
        assert_eq!([SourceEvidenceStateV1::Pending as u8, SourceEvidenceStateV1::Available as u8, SourceEvidenceStateV1::Unavailable as u8], [1, 2, 3]);
        // RFC8032 public test seed, not a production credential.
        let seed: [u8; 32] = hex::decode("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60").unwrap().try_into().unwrap();
        let source = IdentityKeyPair::from_bytes(&seed).unwrap();
        let relay: [u8; 32] = hex::decode("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c").unwrap().try_into().unwrap();
        let query = ReverseOnionSourceQueryV1::sign(&source, relay, [2; 16], [3; 32],
            SourceEvidencePartV1::Lease, [4; 32], 1, 2).unwrap();
        let golden = hex::decode(concat!(
            "4158525101",
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            "0202020202020202", "0202020202020202",
            "0303030303030303", "0303030303030303", "0303030303030303", "0303030303030303",
            "02",
            "0404040404040404", "0404040404040404", "0404040404040404", "0404040404040404",
            "00000000000000010000000000000002"
        )).unwrap();
        assert_eq!(golden.len(), REVERSE_ONION_SOURCE_QUERY_BYTES - 64);
        assert_eq!(query.unsigned(), golden);
        let mut sign_transcript = b"AeroNyx-Reverse-Source-Query-Sign-v1\0".to_vec();
        sign_transcript.extend_from_slice(&Sha256::digest(&golden));
        assert_eq!(query.signature, source.sign(&sign_transcript));
        let mut commitment = Sha256::new();
        commitment.update(b"AeroNyx-Reverse-Source-Query-Commit-v1\0"); commitment.update(query.encode());
        assert_eq!(query.commitment(), <[u8; 32]>::from(commitment.finalize()));
        let abc = hex::decode("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad").unwrap();
        let mut frozen = b"AeroNyx-Reverse-Source-Evidence-Sign-v1\0".to_vec(); frozen.extend_from_slice(&abc);
        assert_eq!(source_evidence_transcript(SOURCE_EVIDENCE_SIGN, b"abc"), frozen);
        assert_eq!(ReverseOnionSourceQueryV1::decode(&query.encode()).unwrap().encode(), query.encode());
        assert_ne!(source_evidence_transcript(SOURCE_QUERY_SIGN, &golden), source_evidence_transcript(REVERSE_ONION_SIGN_DOMAIN, &golden));
    }

    #[test]
    fn source_query_bounds_signature_binding_and_freshness_fail_closed() {
        let f = Fixture::new(); let query = evidence_query(&f, SourceEvidencePartV1::Claim, NOW);
        let bytes = query.encode();
        for end in 0..bytes.len() { assert!(ReverseOnionSourceQueryV1::decode(&bytes[..end]).is_err()); }
        for offset in [0, 4, 5, 37, 69, 85, 117, 118, 150, 166, 229] {
            let mut changed = bytes.clone(); changed[offset] ^= 0xff;
            assert!(ReverseOnionSourceQueryV1::decode(&changed).is_err());
        }
        let mut long = bytes.clone(); long.push(0);
        assert!(ReverseOnionSourceQueryV1::decode(&long).is_err());
        assert!(query.verify_binding(query.source(), query.relay(), [8; 16], [9; 32], NOW).is_err());
        assert!(query.verify_binding(query.source(), query.relay(), query.route_id(), [8; 32], NOW).is_err());
        assert!(query.verify_binding(query.relay(), query.source(), query.route_id(), [9; 32], NOW).is_err());
        assert!(query.verify_binding(f.recipient.public_key_bytes(), query.relay(), query.route_id(), [9; 32], NOW).is_err());
        assert!(query.verify_binding(query.source(), f.recipient.public_key_bytes(), query.route_id(), [9; 32], NOW).is_err());
        assert!(query.verify_binding(query.source(), query.relay(), query.route_id(), [9; 32], NOW - 1).is_err());
        assert!(query.verify_binding(query.source(), query.relay(), query.route_id(), [9; 32], NOW + 20).is_err());
        assert!(source_evidence_window(0, 1).is_err());
        assert!(source_evidence_window(u64::MAX - 1, u64::MAX).is_err());
        assert!(source_evidence_window(NOW, NOW + 31).is_err());
    }

    #[test]
    fn available_requires_complete_chain_before_signing_and_snapshot_agreement() {
        let f = Fixture::new(); let (result, _) = f.result(NOW + 601);
        let now = NOW + 610;
        let queries = [evidence_query(&f, SourceEvidencePartV1::Claim, now),
            evidence_query(&f, SourceEvidencePartV1::Lease, now), evidence_query(&f, SourceEvidencePartV1::Result, now)];
        let responses: Vec<_> = queries.iter().map(|query| ReverseOnionSourceEvidenceV1::available(
            &query_authority(&f, query, now), &f.claim, &f.lease, &result, NOW + 600, now, now + 20, &f.relay).unwrap()).collect();
        let parts: Vec<_> = responses.iter().zip(&queries).map(|(response, query)| response.verify_for_query(query, now).unwrap()).collect();
        let chain = VerifiedSourceEvidenceChain::verify([&parts[0], &parts[1], &parts[2]], NOW + 600, NOW + 899).unwrap();
        assert_eq!(chain.claim().encode(), f.claim.encode());
        assert_eq!(chain.lease().encode(), f.lease.encode());
        assert_eq!(chain.result().encode(), result.encode());
        // Query freshness is acquisition-only, never a replacement for Result
        // grace or the independent source admission deadline.
        assert!(VerifiedSourceEvidenceChain::verify([&parts[0], &parts[1], &parts[2]], NOW + 599, now).is_err());
        assert!(VerifiedSourceEvidenceChain::verify([&parts[0], &parts[1], &parts[2]], NOW + 600, NOW + 900).is_err());
        let authority = query_authority(&f, &queries[0], now);
        let other_claim = ReverseOnionFrameV1::claim(f.relay.public_key_bytes(), [8; 16], NOW, NOW + 30, &f.recipient).unwrap();
        assert!(ReverseOnionSourceEvidenceV1::available(&authority, &other_claim, &f.lease, &result, NOW + 600, now, now + 20, &f.relay).is_err());
        let mut bad = ReverseOnionFrameV1::decode_for_recovery(&f.lease.encode()).unwrap(); bad.signature[0] ^= 1;
        assert!(ReverseOnionSourceEvidenceV1::available(&authority, &f.claim, &bad, &result, NOW + 600, now, now + 20, &f.relay).is_err());
        assert!(ReverseOnionSourceEvidenceV1::available(&authority, &f.claim, &f.lease, &result, NOW + 600, now, now + 20, &f.recipient).is_err());
        let (different, _) = f.result(NOW + 602);
        let swapped = ReverseOnionSourceEvidenceV1::available(&query_authority(&f, &queries[2], now),
            &f.claim, &f.lease, &different, NOW + 600, now, now + 20, &f.relay).unwrap();
        let swapped_part = swapped.verify_for_query(&queries[2], now).unwrap();
        assert!(VerifiedSourceEvidenceChain::verify([&parts[0], &parts[1], &swapped_part], NOW + 600, now).is_err());
        assert!(VerifiedSourceEvidenceChain::verify([&parts[1], &parts[0], &parts[2]], NOW + 600, now).is_err());
    }

    #[test]
    fn pending_unavailable_and_query_swap_never_supply_effect_authority() {
        let f = Fixture::new(); let query = evidence_query(&f, SourceEvidencePartV1::Claim, NOW);
        let authority = query_authority(&f, &query, NOW);
        let pending = ReverseOnionSourceEvidenceV1::pending(&authority, NOW, NOW + 20, &f.relay).unwrap();
        let unavailable = ReverseOnionSourceEvidenceV1::unavailable(&authority, NOW, NOW + 20, &f.relay).unwrap();
        for response in [&pending, &unavailable] {
            let encoded = response.encode();
            assert_eq!(encoded.len(), 283); assert_eq!(&encoded[..5], b"AXRE\x01");
            assert_eq!(encoded[101], 1); assert_eq!(&encoded[119..215], &[0; 96]);
            assert_eq!(&encoded[215..219], &[0; 4]);
            assert_eq!(ReverseOnionSourceEvidenceV1::decode(&encoded).unwrap().encode(), encoded);
            let proof = response.verify_for_query(&query, NOW).unwrap();
            assert!(VerifiedSourceEvidenceChain::verify([&proof, &proof, &proof], NOW + 600, NOW).is_err());
            assert!(response.verify_for_query(&query, NOW + 20).is_err());
        }
        let source = IdentityKeyPair::from_bytes(&[33; 32]).unwrap();
        let different_nonce = ReverseOnionSourceQueryV1::sign(&source, f.relay.public_key_bytes(), f.lease.route_id(),
            [9; 32], SourceEvidencePartV1::Claim, [99; 32], NOW, NOW + 20).unwrap();
        assert!(pending.verify_for_query(&different_nonce, NOW).is_err());
        let other_part = evidence_query(&f, SourceEvidencePartV1::Lease, NOW);
        assert!(pending.verify_for_query(&other_part, NOW).is_err());
        let (result, _) = f.result(NOW + 1);
        let available = ReverseOnionSourceEvidenceV1::available(&query_authority(&f, &query, NOW + 1),
            &f.claim, &f.lease, &result, NOW + 600, NOW + 1, NOW + 20, &f.relay).unwrap();
        assert_eq!(available.verify_for_query(&query, NOW + 1).unwrap().response.state, SourceEvidenceStateV1::Available);
    }

    #[test]
    fn source_evidence_actual_maximum_and_preallocation_admission_are_bounded() {
        let f = Fixture::new(); let query = evidence_query(&f, SourceEvidencePartV1::Result, NOW);
        let opaque = OnionSealedResponse { version: 1, ephemeral_public_key: [9; 32], nonce: [0; 24],
            ciphertext: vec![0; ONION_REPLY_RESPONSE_SIZE_CLASSES[3] + 16] };
        let result = ReverseOnionFrameV1::result(&f.claim, &f.lease, &encode_onion_sealed_response(&opaque).unwrap(),
            NOW + 600, NOW, &f.recipient).unwrap();
        let response = ReverseOnionSourceEvidenceV1::available(&query_authority(&f, &query, NOW),
            &f.claim, &f.lease, &result, NOW + 600, NOW, NOW + 20, &f.relay).unwrap();
        let bytes = response.encode();
        assert_eq!(bytes.len(), MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES);
        assert_eq!(ReverseOnionSourceEvidenceV1::decode(&bytes).unwrap().encode(), bytes);
        let mut long = bytes.clone(); long.push(0);
        assert!(ReverseOnionSourceEvidenceV1::decode(&long).is_err());
        for offset in [0, 4, 5, 37, 69, 101, 102, 215, bytes.len() - 1] {
            let mut changed = bytes.clone(); changed[offset] ^= 0xff;
            assert!(ReverseOnionSourceEvidenceV1::decode(&changed).is_err());
        }
        assert!(source_evidence_payload_admission(SourceEvidencePartV1::Result, SourceEvidenceStateV1::Available, usize::MAX, usize::MAX).is_err());
        assert!(source_evidence_payload_admission(SourceEvidencePartV1::Claim, SourceEvidenceStateV1::Available, 235, 283 + 235).is_err());
        assert!(source_evidence_payload_admission(SourceEvidencePartV1::Result, SourceEvidenceStateV1::Pending, 1, 284).is_err());
        for part in [SourceEvidencePartV1::Claim, SourceEvidencePartV1::Lease, SourceEvidencePartV1::Result] {
            let maximum = part.limit();
            assert!(source_evidence_payload_admission(part, SourceEvidenceStateV1::Available,
                maximum, REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD + maximum).is_ok());
            assert!(source_evidence_payload_admission(part, SourceEvidenceStateV1::Available,
                maximum + 1, REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD + maximum + 1).is_err());
            // Invalid length must fail even with no payload and invalid outer
            // signature: decoder admission precedes copy and signature checks.
            let mut malformed = bytes[..REVERSE_ONION_SOURCE_EVIDENCE_OVERHEAD].to_vec();
            malformed[101] = part as u8;
            malformed[215..219].copy_from_slice(&u32::MAX.to_be_bytes());
            assert!(ReverseOnionSourceEvidenceV1::decode(&malformed).is_err());
        }
        let mut truncated = bytes; truncated.truncate(282);
        assert!(ReverseOnionSourceEvidenceV1::decode(&truncated).is_err());
    }

    #[test]
    fn typed_transitions_preserve_armed_barrier_and_never_revive_terminal_state() {
        let f = Fixture::new();
        let queued = ReverseOnionDeliveryStateV1::queued(&f.claim, &f.lease, NOW + 600, NOW).unwrap();
        assert!(queued.arm(NOW - 1).is_err());
        let armed = queued.arm(NOW + 31).unwrap();
        assert!(armed.arm(NOW + 32).is_err());
        assert_eq!(armed.exact_replay_bytes(&f.lease, NOW + 599).unwrap(), f.lease.encode());
        assert!(armed.exact_replay_bytes(&f.lease, NOW + 600).is_err());
        let (result, _) = f.result(NOW + 610);
        let completed = armed.accept_result(&f.claim, &f.lease, &result, NOW + 600, NOW + 610).unwrap();
        assert_eq!(completed.phase(), ReverseOnionPhaseV1::ResultAvailable);
        assert!(completed.arm(NOW + 611).is_err());
        assert!(armed.mark_ambiguous().unwrap().accept_result(&f.claim, &f.lease, &result, NOW + 600, NOW + 610).is_err());
        assert!(armed.expire(NOW + 899).is_err());
        assert_eq!(armed.expire(NOW + 900).unwrap().phase(), ReverseOnionPhaseV1::Expired);
    }

    #[test]
    fn shortened_route_preserves_envelope_replay_evidence() {
        let f = Fixture::new();
        let envelope = f.lease.verify_lease(&f.claim, NOW + 600, NOW).unwrap();
        let short = ReverseOnionFrameV1::lease(&f.claim, &envelope, [8; 16], NOW + 10, NOW, &f.relay).unwrap();
        assert_eq!(short.expires_at(), NOW + 10);
        assert_eq!(short.result_retention_deadline().unwrap(), NOW + 310);
        assert_eq!(short.replay_evidence_deadline().unwrap(), NOW + 600);
    }
}
