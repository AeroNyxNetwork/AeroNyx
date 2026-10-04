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
const REVERSE_ONION_ENVELOPE_BYTES: usize = 256 * 1024;

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

impl ReverseOnionFrameV1 {
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
            ReverseOnionKindV1::Lease => REVERSE_ONION_ENVELOPE_BYTES,
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
