// ============================================================================
// File: crates/aeronyx-core/src/protocol/discovery_endpoint_proof.rs
// ============================================================================
//! Canonical Stage A endpoint-possession evidence for permissionless discovery.
//!
//! This module defines cryptographic evidence only. A valid proof says that the
//! node identity signed one fresh, challenger-bound endpoint challenge. It does
//! not promote a node, establish reachability from multiple networks, select a
//! canonical directory, or grant consensus, routing, reputation, or economic
//! authority.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};

// [PERMISSIONLESS-ENDPOINT-PROOF 2026-09-23 by Codex] Freeze a small,
// allocation-bounded wire independently from the existing discovery codec.
const FRAME_MAGIC: [u8; 4] = *b"ADEP";
const FRAME_HEADER_BYTES: usize = 8;
const FRAME_KIND_CHALLENGE: u8 = 1;
const FRAME_KIND_PROOF: u8 = 2;
const CHALLENGE_BODY_BYTES_U16: u16 = 272;
const PROOF_BODY_BYTES_U16: u16 = 312;
const CHALLENGE_BODY_BYTES: usize = CHALLENGE_BODY_BYTES_U16 as usize;
const PROOF_BODY_BYTES: usize = PROOF_BODY_BYTES_U16 as usize;
const CHALLENGE_FRAME_BYTES: usize = FRAME_HEADER_BYTES + CHALLENGE_BODY_BYTES;
const PROOF_FRAME_BYTES: usize = FRAME_HEADER_BYTES + PROOF_BODY_BYTES;

const CHALLENGE_SIGNATURE_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointChallengeV1\0";
const CHALLENGE_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointChallengeCommitmentV1\0";
const ENDPOINT_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx/DiscoveryPublicEndpointV1\0";
const PROOF_SIGNATURE_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointProofV1\0";

// [PERMISSIONLESS-ENDPOINT-TRANSPORT 2026-09-24 by Codex] Keep transport
// authentication separate from both endpoint evidence and the server adapter.
const TRANSPORT_MAGIC: [u8; 4] = *b"ADET";
const TRANSPORT_HEADER_BYTES: usize = 8;
const TRANSPORT_FIXED_BODY_BYTES: usize = 32 * 6 + 8 * 2 + 2 + 64;
const TRANSPORT_SIGNATURE_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointAuthenticatedTransportV1\0";
const TRANSPORT_INNER_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx/DiscoveryEndpointAuthenticatedTransportInnerV1\0";
const TRANSPORT_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx/DiscoveryEndpointAuthenticatedTransportCommitmentV1\0";
const STAGE_C_INNER_MAGIC: [u8; 4] = *b"ADEA";
const STAGE_C_INNER_VERSION_V1: u8 = 1;
const STAGE_C_INNER_HEADER_BYTES: usize = 8;

/// Frozen version of the permissionless endpoint-proof protocol.
pub const DISCOVERY_ENDPOINT_PROOF_VERSION_V1: u8 = 1;
/// Maximum lifetime of one challenge, in seconds.
pub const DISCOVERY_ENDPOINT_CHALLENGE_MAX_TTL_SECS: u64 = 300;
/// Maximum accepted positive clock skew for a challenge issue time.
pub const DISCOVERY_ENDPOINT_CHALLENGE_FUTURE_SKEW_SECS: u64 = 30;
/// Exact encoded length of a V1 challenge frame.
pub const DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1: usize = CHALLENGE_FRAME_BYTES;
/// Exact encoded length of a V1 proof frame.
pub const DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1: usize = PROOF_FRAME_BYTES;
/// Frozen version of the authenticated Stage C transport envelope.
pub const DISCOVERY_ENDPOINT_TRANSPORT_VERSION_V1: u8 = 1;
/// Maximum lifetime of one authenticated transport request, in seconds.
pub const DISCOVERY_ENDPOINT_TRANSPORT_MAX_TTL_SECS: u64 = 120;
/// Maximum accepted positive clock skew for an authenticated transport request.
pub const DISCOVERY_ENDPOINT_TRANSPORT_FUTURE_SKEW_SECS: u64 = 30;
/// Maximum exact Stage C ADEA frame carried by the authenticated transport.
pub const DISCOVERY_ENDPOINT_TRANSPORT_MAX_INNER_BYTES_V1: usize = 644;
/// Maximum canonical encoded transport frame length.
pub const DISCOVERY_ENDPOINT_TRANSPORT_MAX_FRAME_BYTES_V1: usize = TRANSPORT_HEADER_BYTES
    + TRANSPORT_FIXED_BODY_BYTES
    + DISCOVERY_ENDPOINT_TRANSPORT_MAX_INNER_BYTES_V1;

/// Coarse, privacy-safe endpoint-proof validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryEndpointProofError {
    /// A field, endpoint, length, or canonical representation is malformed.
    Malformed,
    /// The frame version or kind is not supported.
    Unsupported,
    /// A challenge is not currently valid.
    NotCurrentlyValid,
    /// A caller-supplied target, challenge, or context does not match.
    ContextMismatch,
    /// An Ed25519 signature did not verify.
    SignatureRejected,
}

impl fmt::Display for DiscoveryEndpointProofError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Malformed => "discovery endpoint proof is malformed",
            Self::Unsupported => "discovery endpoint proof version or kind is unsupported",
            Self::NotCurrentlyValid => "discovery endpoint proof is not currently valid",
            Self::ContextMismatch => "discovery endpoint proof context does not match",
            Self::SignatureRejected => "discovery endpoint proof signature was rejected",
        })
    }
}

impl std::error::Error for DiscoveryEndpointProofError {}

/// Frozen Stage C operation and path identity carried by authenticated transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DiscoveryEndpointTransportOperationV1 {
    /// Issues or exactly replays one endpoint challenge.
    Issue = 1,
    /// Verifies and atomically consumes one endpoint proof.
    Verify = 2,
}

impl DiscoveryEndpointTransportOperationV1 {
    /// Returns the exact HTTP path bound by this operation's signature.
    #[must_use]
    pub const fn path(self) -> &'static str {
        match self {
            Self::Issue => "/api/discovery/endpoint-proof/challenge",
            Self::Verify => "/api/discovery/endpoint-proof/verify",
        }
    }

    const fn from_u8(value: u8) -> Result<Self, DiscoveryEndpointProofError> {
        match value {
            1 => Ok(Self::Issue),
            2 => Ok(Self::Verify),
            _ => Err(DiscoveryEndpointProofError::Unsupported),
        }
    }
}

/// Challenger-signed request proving the exact endpoint and discovery context.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointChallengeV1 {
    target_node_id: [u8; 32],
    descriptor_commitment: [u8; 32],
    endpoint_commitment: [u8; 32],
    nonce: [u8; 32],
    challenger_node_id: [u8; 32],
    challenger_context: [u8; 32],
    issued_at: u64,
    expires_at: u64,
    signature: [u8; 64],
}

impl fmt::Debug for DiscoveryEndpointChallengeV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointChallengeV1")
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

impl DiscoveryEndpointChallengeV1 {
    /// Issues one exact challenge. The caller must generate `nonce` with a CSPRNG.
    ///
    /// # Errors
    /// Returns [`DiscoveryEndpointProofError::Malformed`] for reserved values or
    /// an invalid lifetime.
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        target_node_id: [u8; 32],
        descriptor_commitment: [u8; 32],
        endpoint_commitment: [u8; 32],
        nonce: [u8; 32],
        challenger_context: [u8; 32],
        issued_at: u64,
        expires_at: u64,
        challenger: &IdentityKeyPair,
    ) -> Result<Self, DiscoveryEndpointProofError> {
        let challenger_node_id = challenger.public_key_bytes();
        validate_claims(
            &target_node_id,
            &descriptor_commitment,
            &endpoint_commitment,
            &nonce,
            &challenger_node_id,
            &challenger_context,
            issued_at,
            expires_at,
        )?;
        let mut challenge = Self {
            target_node_id,
            descriptor_commitment,
            endpoint_commitment,
            nonce,
            challenger_node_id,
            challenger_context,
            issued_at,
            expires_at,
            signature: [0; 64],
        };
        challenge.signature = challenger.sign(&challenge.signing_bytes());
        Ok(challenge)
    }

    /// Verifies the signature, lifetime, and exact caller-selected context.
    ///
    /// # Errors
    /// Returns a coarse error when claims, time, context, or signature fail.
    pub fn verify_at(
        &self,
        now: u64,
        expected_context: &[u8; 32],
    ) -> Result<(), DiscoveryEndpointProofError> {
        validate_claims(
            &self.target_node_id,
            &self.descriptor_commitment,
            &self.endpoint_commitment,
            &self.nonce,
            &self.challenger_node_id,
            &self.challenger_context,
            self.issued_at,
            self.expires_at,
        )?;
        if &self.challenger_context != expected_context {
            return Err(DiscoveryEndpointProofError::ContextMismatch);
        }
        validate_time(self.issued_at, self.expires_at, now)?;
        verify_signature(
            &self.challenger_node_id,
            &self.signing_bytes(),
            &self.signature,
        )
    }

    /// Returns the exact target node identity.
    #[must_use]
    pub const fn target_node_id(&self) -> [u8; 32] {
        self.target_node_id
    }

    /// Returns the signed descriptor commitment.
    #[must_use]
    pub const fn descriptor_commitment(&self) -> [u8; 32] {
        self.descriptor_commitment
    }

    /// Returns the canonical public endpoint commitment.
    #[must_use]
    pub const fn endpoint_commitment(&self) -> [u8; 32] {
        self.endpoint_commitment
    }

    /// Returns the cryptographic challenge nonce.
    #[must_use]
    pub const fn nonce(&self) -> [u8; 32] {
        self.nonce
    }

    /// Returns the challenger identity.
    #[must_use]
    pub const fn challenger_node_id(&self) -> [u8; 32] {
        self.challenger_node_id
    }

    /// Returns the opaque application-selected challenger context.
    #[must_use]
    pub const fn challenger_context(&self) -> [u8; 32] {
        self.challenger_context
    }

    /// Returns the issue time in Unix epoch seconds.
    #[must_use]
    pub const fn issued_at(&self) -> u64 {
        self.issued_at
    }

    /// Returns the expiry time in Unix epoch seconds.
    #[must_use]
    pub const fn expires_at(&self) -> u64 {
        self.expires_at
    }

    /// Returns a commitment to the exact canonical signed challenge frame.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        domain_hash(CHALLENGE_COMMITMENT_DOMAIN, &self.encode())
    }

    /// Encodes one canonical fixed-width challenge frame.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut body = self.unsigned_body();
        body.extend_from_slice(&self.signature);
        encode_frame(FRAME_KIND_CHALLENGE, &body)
    }

    /// Decodes one canonical fixed-width challenge frame.
    ///
    /// # Errors
    /// Returns a coarse error for an unsupported or non-canonical frame.
    pub fn decode(bytes: &[u8]) -> Result<Self, DiscoveryEndpointProofError> {
        let body = decode_frame(bytes, FRAME_KIND_CHALLENGE, CHALLENGE_BODY_BYTES)?;
        let mut offset = 0;
        let challenge = Self {
            target_node_id: take_array(body, &mut offset)?,
            descriptor_commitment: take_array(body, &mut offset)?,
            endpoint_commitment: take_array(body, &mut offset)?,
            nonce: take_array(body, &mut offset)?,
            challenger_node_id: take_array(body, &mut offset)?,
            challenger_context: take_array(body, &mut offset)?,
            issued_at: take_u64(body, &mut offset)?,
            expires_at: take_u64(body, &mut offset)?,
            signature: take_array(body, &mut offset)?,
        };
        if offset != body.len() || is_reserved(&challenge.signature) {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        validate_claims(
            &challenge.target_node_id,
            &challenge.descriptor_commitment,
            &challenge.endpoint_commitment,
            &challenge.nonce,
            &challenge.challenger_node_id,
            &challenge.challenger_context,
            challenge.issued_at,
            challenge.expires_at,
        )?;
        if challenge.encode() != bytes {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        Ok(challenge)
    }

    fn unsigned_body(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(CHALLENGE_BODY_BYTES - 64);
        bytes.extend_from_slice(&self.target_node_id);
        bytes.extend_from_slice(&self.descriptor_commitment);
        bytes.extend_from_slice(&self.endpoint_commitment);
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&self.challenger_node_id);
        bytes.extend_from_slice(&self.challenger_context);
        bytes.extend_from_slice(&self.issued_at.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at.to_be_bytes());
        bytes
    }

    fn signing_bytes(&self) -> Vec<u8> {
        domain_bytes(CHALLENGE_SIGNATURE_DOMAIN, &self.unsigned_body())
    }
}

/// Target-signed response to one exact verified endpoint challenge.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointProofV1 {
    challenge_commitment: [u8; 32],
    target_node_id: [u8; 32],
    descriptor_commitment: [u8; 32],
    endpoint_commitment: [u8; 32],
    nonce: [u8; 32],
    challenger_node_id: [u8; 32],
    challenger_context: [u8; 32],
    issued_at: u64,
    expires_at: u64,
    responded_at: u64,
    signature: [u8; 64],
}

impl fmt::Debug for DiscoveryEndpointProofV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointProofV1")
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .field("responded_at", &self.responded_at)
            .finish_non_exhaustive()
    }
}

impl DiscoveryEndpointProofV1 {
    /// Signs a proof after validating the challenge and exact target identity.
    ///
    /// # Errors
    /// Returns a coarse error when the challenge or target binding is invalid.
    pub fn respond(
        challenge: &DiscoveryEndpointChallengeV1,
        expected_context: &[u8; 32],
        responded_at: u64,
        target: &IdentityKeyPair,
    ) -> Result<Self, DiscoveryEndpointProofError> {
        challenge.verify_at(responded_at, expected_context)?;
        let target_node_id = target.public_key_bytes();
        if target_node_id != challenge.target_node_id {
            return Err(DiscoveryEndpointProofError::ContextMismatch);
        }
        let mut proof = Self {
            challenge_commitment: challenge.commitment(),
            target_node_id,
            descriptor_commitment: challenge.descriptor_commitment,
            endpoint_commitment: challenge.endpoint_commitment,
            nonce: challenge.nonce,
            challenger_node_id: challenge.challenger_node_id,
            challenger_context: challenge.challenger_context,
            issued_at: challenge.issued_at,
            expires_at: challenge.expires_at,
            responded_at,
            signature: [0; 64],
        };
        proof.signature = target.sign(&proof.signing_bytes());
        Ok(proof)
    }

    /// Verifies the proof against the exact canonical challenge and context.
    ///
    /// # Errors
    /// Returns a coarse error when any exact binding, time, or signature fails.
    pub fn verify_for_challenge(
        &self,
        challenge: &DiscoveryEndpointChallengeV1,
        now: u64,
        expected_context: &[u8; 32],
    ) -> Result<(), DiscoveryEndpointProofError> {
        challenge.verify_at(now, expected_context)?;
        validate_claims(
            &self.target_node_id,
            &self.descriptor_commitment,
            &self.endpoint_commitment,
            &self.nonce,
            &self.challenger_node_id,
            &self.challenger_context,
            self.issued_at,
            self.expires_at,
        )?;
        if self.challenge_commitment != challenge.commitment()
            || self.target_node_id != challenge.target_node_id
            || self.descriptor_commitment != challenge.descriptor_commitment
            || self.endpoint_commitment != challenge.endpoint_commitment
            || self.nonce != challenge.nonce
            || self.challenger_node_id != challenge.challenger_node_id
            || self.challenger_context != *expected_context
            || self.issued_at != challenge.issued_at
            || self.expires_at != challenge.expires_at
            || self.responded_at < self.issued_at
            || self.responded_at > self.expires_at
            || self.responded_at > now.saturating_add(DISCOVERY_ENDPOINT_CHALLENGE_FUTURE_SKEW_SECS)
        {
            return Err(DiscoveryEndpointProofError::ContextMismatch);
        }
        verify_signature(&self.target_node_id, &self.signing_bytes(), &self.signature)
    }

    /// Returns the exact challenge commitment.
    #[must_use]
    pub const fn challenge_commitment(&self) -> [u8; 32] {
        self.challenge_commitment
    }

    /// Returns the proof response time in Unix epoch seconds.
    #[must_use]
    pub const fn responded_at(&self) -> u64 {
        self.responded_at
    }

    /// Encodes one canonical fixed-width proof frame.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut body = self.unsigned_body();
        body.extend_from_slice(&self.signature);
        encode_frame(FRAME_KIND_PROOF, &body)
    }

    /// Decodes one canonical fixed-width proof frame.
    ///
    /// # Errors
    /// Returns a coarse error for an unsupported or non-canonical frame.
    pub fn decode(bytes: &[u8]) -> Result<Self, DiscoveryEndpointProofError> {
        let body = decode_frame(bytes, FRAME_KIND_PROOF, PROOF_BODY_BYTES)?;
        let mut offset = 0;
        let proof = Self {
            challenge_commitment: take_array(body, &mut offset)?,
            target_node_id: take_array(body, &mut offset)?,
            descriptor_commitment: take_array(body, &mut offset)?,
            endpoint_commitment: take_array(body, &mut offset)?,
            nonce: take_array(body, &mut offset)?,
            challenger_node_id: take_array(body, &mut offset)?,
            challenger_context: take_array(body, &mut offset)?,
            issued_at: take_u64(body, &mut offset)?,
            expires_at: take_u64(body, &mut offset)?,
            responded_at: take_u64(body, &mut offset)?,
            signature: take_array(body, &mut offset)?,
        };
        if offset != body.len()
            || is_reserved(&proof.challenge_commitment)
            || is_reserved(&proof.signature)
            || proof.responded_at < proof.issued_at
            || proof.responded_at > proof.expires_at
        {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        validate_claims(
            &proof.target_node_id,
            &proof.descriptor_commitment,
            &proof.endpoint_commitment,
            &proof.nonce,
            &proof.challenger_node_id,
            &proof.challenger_context,
            proof.issued_at,
            proof.expires_at,
        )?;
        if proof.encode() != bytes {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        Ok(proof)
    }

    fn unsigned_body(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PROOF_BODY_BYTES - 64);
        bytes.extend_from_slice(&self.challenge_commitment);
        bytes.extend_from_slice(&self.target_node_id);
        bytes.extend_from_slice(&self.descriptor_commitment);
        bytes.extend_from_slice(&self.endpoint_commitment);
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&self.challenger_node_id);
        bytes.extend_from_slice(&self.challenger_context);
        bytes.extend_from_slice(&self.issued_at.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at.to_be_bytes());
        bytes.extend_from_slice(&self.responded_at.to_be_bytes());
        bytes
    }

    fn signing_bytes(&self) -> Vec<u8> {
        domain_bytes(PROOF_SIGNATURE_DOMAIN, &self.unsigned_body())
    }
}

/// Target-authenticated transport for one exact Stage C ADEA request frame.
///
/// This envelope proves only that the target node key signed one bounded,
/// fresh request for the exact operation and path. It does not establish
/// endpoint reachability, promote a descriptor, or grant routing authority.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointAuthenticatedTransportV1 {
    operation: DiscoveryEndpointTransportOperationV1,
    request_id: [u8; 32],
    target_node_id: [u8; 32],
    descriptor_commitment: [u8; 32],
    endpoint_commitment: [u8; 32],
    flow_context: [u8; 32],
    issued_at: u64,
    expires_at: u64,
    inner_commitment: [u8; 32],
    inner_frame: Vec<u8>,
    signature: [u8; 64],
}

impl fmt::Debug for DiscoveryEndpointAuthenticatedTransportV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointAuthenticatedTransportV1")
            .field("operation", &self.operation)
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

impl DiscoveryEndpointAuthenticatedTransportV1 {
    /// Builds and signs one canonical transport envelope.
    ///
    /// `inner_frame` must be a canonical Stage C ADEA V1 request whose kind
    /// matches `operation` and whose first body field is the same request id.
    ///
    /// # Errors
    /// Returns a coarse error for reserved fields, invalid time, malformed or
    /// oversized inner bytes, or a signer that does not match the target id.
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        operation: DiscoveryEndpointTransportOperationV1,
        request_id: [u8; 32],
        descriptor_commitment: [u8; 32],
        endpoint_commitment: [u8; 32],
        flow_context: [u8; 32],
        issued_at: u64,
        expires_at: u64,
        inner_frame: &[u8],
        target: &IdentityKeyPair,
    ) -> Result<Self, DiscoveryEndpointProofError> {
        let target_node_id = target.public_key_bytes();
        validate_transport_claims(
            operation,
            &request_id,
            &target_node_id,
            &descriptor_commitment,
            &endpoint_commitment,
            &flow_context,
            issued_at,
            expires_at,
            inner_frame,
        )?;
        let mut transport = Self {
            operation,
            request_id,
            target_node_id,
            descriptor_commitment,
            endpoint_commitment,
            flow_context,
            issued_at,
            expires_at,
            inner_commitment: domain_hash(TRANSPORT_INNER_COMMITMENT_DOMAIN, inner_frame),
            inner_frame: inner_frame.to_vec(),
            signature: [0; 64],
        };
        transport.signature = target.sign(&transport.signing_bytes());
        Ok(transport)
    }

    /// Decodes one canonical bounded transport frame without trusting it.
    ///
    /// Call [`Self::verify_at`] before using any identity or context field.
    ///
    /// # Errors
    /// Returns a coarse error for unsupported, malformed, non-canonical,
    /// oversized, trailing, or reserved data.
    pub fn decode(bytes: &[u8]) -> Result<Self, DiscoveryEndpointProofError> {
        if bytes.len() < TRANSPORT_HEADER_BYTES
            || bytes.len() > DISCOVERY_ENDPOINT_TRANSPORT_MAX_FRAME_BYTES_V1
        {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        if bytes[..4] != TRANSPORT_MAGIC {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        if bytes[4] != DISCOVERY_ENDPOINT_TRANSPORT_VERSION_V1 {
            return Err(DiscoveryEndpointProofError::Unsupported);
        }
        let operation = DiscoveryEndpointTransportOperationV1::from_u8(bytes[5])?;
        let body_len = usize::from(u16::from_be_bytes([bytes[6], bytes[7]]));
        if body_len < TRANSPORT_FIXED_BODY_BYTES || bytes.len() != TRANSPORT_HEADER_BYTES + body_len
        {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        let body = &bytes[TRANSPORT_HEADER_BYTES..];
        let mut offset = 0;
        let request_id = take_array(body, &mut offset)?;
        let target_node_id = take_array(body, &mut offset)?;
        let descriptor_commitment = take_array(body, &mut offset)?;
        let endpoint_commitment = take_array(body, &mut offset)?;
        let flow_context = take_array(body, &mut offset)?;
        let issued_at = take_u64(body, &mut offset)?;
        let expires_at = take_u64(body, &mut offset)?;
        let inner_commitment = take_array(body, &mut offset)?;
        let inner_len = usize::from(u16::from_be_bytes(take_array(body, &mut offset)?));
        if inner_len == 0 || inner_len > DISCOVERY_ENDPOINT_TRANSPORT_MAX_INNER_BYTES_V1 {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        let inner_end = offset
            .checked_add(inner_len)
            .ok_or(DiscoveryEndpointProofError::Malformed)?;
        let inner_frame = body
            .get(offset..inner_end)
            .ok_or(DiscoveryEndpointProofError::Malformed)?
            .to_vec();
        offset = inner_end;
        let signature = take_array(body, &mut offset)?;
        if offset != body.len()
            || is_reserved(&inner_commitment)
            || is_reserved(&signature)
            || inner_commitment != domain_hash(TRANSPORT_INNER_COMMITMENT_DOMAIN, &inner_frame)
        {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        validate_transport_claims(
            operation,
            &request_id,
            &target_node_id,
            &descriptor_commitment,
            &endpoint_commitment,
            &flow_context,
            issued_at,
            expires_at,
            &inner_frame,
        )?;
        let transport = Self {
            operation,
            request_id,
            target_node_id,
            descriptor_commitment,
            endpoint_commitment,
            flow_context,
            issued_at,
            expires_at,
            inner_commitment,
            inner_frame,
            signature,
        };
        if transport.encode() != bytes {
            return Err(DiscoveryEndpointProofError::Malformed);
        }
        Ok(transport)
    }

    /// Verifies freshness, exact expected operation, target, flow context, and signature.
    ///
    /// # Errors
    /// Returns a coarse error for a stale request, context mismatch, or invalid
    /// target signature.
    pub fn verify_at(
        &self,
        now: u64,
        expected_operation: DiscoveryEndpointTransportOperationV1,
        expected_target_node_id: &[u8; 32],
        expected_flow_context: &[u8; 32],
    ) -> Result<(), DiscoveryEndpointProofError> {
        validate_transport_claims(
            self.operation,
            &self.request_id,
            &self.target_node_id,
            &self.descriptor_commitment,
            &self.endpoint_commitment,
            &self.flow_context,
            self.issued_at,
            self.expires_at,
            &self.inner_frame,
        )?;
        if self.operation != expected_operation
            || &self.target_node_id != expected_target_node_id
            || &self.flow_context != expected_flow_context
            || self.inner_commitment
                != domain_hash(TRANSPORT_INNER_COMMITMENT_DOMAIN, &self.inner_frame)
        {
            return Err(DiscoveryEndpointProofError::ContextMismatch);
        }
        validate_transport_time(self.issued_at, self.expires_at, now)?;
        verify_signature(&self.target_node_id, &self.signing_bytes(), &self.signature)
    }

    /// Decodes and verifies one transport frame before exposing its authority.
    ///
    /// # Errors
    /// Returns a coarse decode, context, time, or signature error.
    pub fn decode_verified_at(
        bytes: &[u8],
        now: u64,
        expected_operation: DiscoveryEndpointTransportOperationV1,
        expected_target_node_id: &[u8; 32],
        expected_flow_context: &[u8; 32],
    ) -> Result<Self, DiscoveryEndpointProofError> {
        let transport = Self::decode(bytes)?;
        transport.verify_at(
            now,
            expected_operation,
            expected_target_node_id,
            expected_flow_context,
        )?;
        Ok(transport)
    }

    /// Returns the exact operation and bound path identity.
    #[must_use]
    pub const fn operation(&self) -> DiscoveryEndpointTransportOperationV1 {
        self.operation
    }

    /// Returns the exact replay request id.
    #[must_use]
    pub const fn request_id(&self) -> [u8; 32] {
        self.request_id
    }

    /// Returns the authenticated target node identity.
    #[must_use]
    pub const fn target_node_id(&self) -> [u8; 32] {
        self.target_node_id
    }

    /// Returns the exact signed descriptor commitment.
    #[must_use]
    pub const fn descriptor_commitment(&self) -> [u8; 32] {
        self.descriptor_commitment
    }

    /// Returns the canonical public endpoint commitment.
    #[must_use]
    pub const fn endpoint_commitment(&self) -> [u8; 32] {
        self.endpoint_commitment
    }

    /// Returns the stable authenticated flow context.
    #[must_use]
    pub const fn flow_context(&self) -> [u8; 32] {
        self.flow_context
    }

    /// Returns the exact inner ADEA frame commitment.
    #[must_use]
    pub const fn inner_commitment(&self) -> [u8; 32] {
        self.inner_commitment
    }

    /// Returns the exact canonical Stage C ADEA request bytes.
    #[must_use]
    pub fn inner_frame(&self) -> &[u8] {
        &self.inner_frame
    }

    /// Returns a commitment to this exact canonical signed transport frame.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        domain_hash(TRANSPORT_COMMITMENT_DOMAIN, &self.encode())
    }

    /// Encodes one canonical bounded transport frame.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let unsigned = self.unsigned_body();
        let body_len = unsigned.len() + self.signature.len();
        let body_len = u16::try_from(body_len).unwrap_or(u16::MAX);
        let mut frame = Vec::with_capacity(TRANSPORT_HEADER_BYTES + usize::from(body_len));
        frame.extend_from_slice(&TRANSPORT_MAGIC);
        frame.push(DISCOVERY_ENDPOINT_TRANSPORT_VERSION_V1);
        frame.push(self.operation as u8);
        frame.extend_from_slice(&body_len.to_be_bytes());
        frame.extend_from_slice(&unsigned);
        frame.extend_from_slice(&self.signature);
        frame
    }

    fn unsigned_body(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(TRANSPORT_FIXED_BODY_BYTES - 64 + self.inner_frame.len());
        body.extend_from_slice(&self.request_id);
        body.extend_from_slice(&self.target_node_id);
        body.extend_from_slice(&self.descriptor_commitment);
        body.extend_from_slice(&self.endpoint_commitment);
        body.extend_from_slice(&self.flow_context);
        body.extend_from_slice(&self.issued_at.to_be_bytes());
        body.extend_from_slice(&self.expires_at.to_be_bytes());
        body.extend_from_slice(&self.inner_commitment);
        body.extend_from_slice(
            &u16::try_from(self.inner_frame.len())
                .unwrap_or(u16::MAX)
                .to_be_bytes(),
        );
        body.extend_from_slice(&self.inner_frame);
        body
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let unsigned = self.unsigned_body();
        let mut binding = Vec::with_capacity(1 + 2 + self.operation.path().len() + unsigned.len());
        binding.push(self.operation as u8);
        binding.extend_from_slice(
            &u16::try_from(self.operation.path().len())
                .unwrap_or(u16::MAX)
                .to_be_bytes(),
        );
        binding.extend_from_slice(self.operation.path().as_bytes());
        binding.extend_from_slice(&unsigned);
        domain_bytes(TRANSPORT_SIGNATURE_DOMAIN, &binding)
    }
}

/// Commits to one canonical public IP socket endpoint.
///
/// DNS names are intentionally excluded from V1 so resolution changes cannot
/// alter what a signed proof means. Later networking code must still establish
/// that it reached this endpoint; this helper performs no network I/O.
///
/// # Errors
/// Returns [`DiscoveryEndpointProofError::Malformed`] unless `endpoint` is the
/// canonical string form of one nonzero-port public IP socket address.
pub fn canonical_public_endpoint_commitment(
    endpoint: &str,
) -> Result<[u8; 32], DiscoveryEndpointProofError> {
    if endpoint.is_empty() || endpoint.len() > 64 || endpoint.trim() != endpoint {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    let parsed: SocketAddr = endpoint
        .parse()
        .map_err(|_| DiscoveryEndpointProofError::Malformed)?;
    if parsed.port() == 0 || parsed.to_string() != endpoint || !is_public_ip(parsed.ip()) {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    Ok(domain_hash(ENDPOINT_COMMITMENT_DOMAIN, endpoint.as_bytes()))
}

#[allow(clippy::too_many_arguments)]
fn validate_claims(
    target_node_id: &[u8; 32],
    descriptor_commitment: &[u8; 32],
    endpoint_commitment: &[u8; 32],
    nonce: &[u8; 32],
    challenger_node_id: &[u8; 32],
    challenger_context: &[u8; 32],
    issued_at: u64,
    expires_at: u64,
) -> Result<(), DiscoveryEndpointProofError> {
    if [
        target_node_id,
        descriptor_commitment,
        endpoint_commitment,
        nonce,
        challenger_node_id,
        challenger_context,
    ]
    .into_iter()
    .any(is_reserved)
        || issued_at == 0
        || expires_at <= issued_at
        || expires_at - issued_at > DISCOVERY_ENDPOINT_CHALLENGE_MAX_TTL_SECS
    {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_transport_claims(
    operation: DiscoveryEndpointTransportOperationV1,
    request_id: &[u8; 32],
    target_node_id: &[u8; 32],
    descriptor_commitment: &[u8; 32],
    endpoint_commitment: &[u8; 32],
    flow_context: &[u8; 32],
    issued_at: u64,
    expires_at: u64,
    inner_frame: &[u8],
) -> Result<(), DiscoveryEndpointProofError> {
    if [
        request_id,
        target_node_id,
        descriptor_commitment,
        endpoint_commitment,
        flow_context,
    ]
    .into_iter()
    .any(is_reserved)
        || issued_at == 0
        || expires_at <= issued_at
        || expires_at - issued_at > DISCOVERY_ENDPOINT_TRANSPORT_MAX_TTL_SECS
    {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    validate_stage_c_inner_frame(operation, request_id, inner_frame)
}

fn validate_stage_c_inner_frame(
    operation: DiscoveryEndpointTransportOperationV1,
    request_id: &[u8; 32],
    inner_frame: &[u8],
) -> Result<(), DiscoveryEndpointProofError> {
    if inner_frame.len() < STAGE_C_INNER_HEADER_BYTES + 32
        || inner_frame.len() > DISCOVERY_ENDPOINT_TRANSPORT_MAX_INNER_BYTES_V1
        || inner_frame[..4] != STAGE_C_INNER_MAGIC
    {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    if inner_frame[4] != STAGE_C_INNER_VERSION_V1 || inner_frame[5] != operation as u8 {
        return Err(DiscoveryEndpointProofError::Unsupported);
    }
    let body_len = usize::from(u16::from_be_bytes([inner_frame[6], inner_frame[7]]));
    if inner_frame.len() != STAGE_C_INNER_HEADER_BYTES + body_len
        || inner_frame[STAGE_C_INNER_HEADER_BYTES..STAGE_C_INNER_HEADER_BYTES + 32]
            != request_id[..]
    {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    Ok(())
}

const fn validate_time(
    issued_at: u64,
    expires_at: u64,
    now: u64,
) -> Result<(), DiscoveryEndpointProofError> {
    let latest_issue = now.saturating_add(DISCOVERY_ENDPOINT_CHALLENGE_FUTURE_SKEW_SECS);
    if issued_at > latest_issue || now > expires_at {
        return Err(DiscoveryEndpointProofError::NotCurrentlyValid);
    }
    Ok(())
}

const fn validate_transport_time(
    issued_at: u64,
    expires_at: u64,
    now: u64,
) -> Result<(), DiscoveryEndpointProofError> {
    let latest_issue = now.saturating_add(DISCOVERY_ENDPOINT_TRANSPORT_FUTURE_SKEW_SECS);
    if issued_at > latest_issue || now > expires_at {
        return Err(DiscoveryEndpointProofError::NotCurrentlyValid);
    }
    Ok(())
}

fn verify_signature(
    public_key: &[u8; 32],
    message: &[u8],
    signature: &[u8; 64],
) -> Result<(), DiscoveryEndpointProofError> {
    IdentityPublicKey::from_bytes(public_key)
        .map_err(|_| DiscoveryEndpointProofError::SignatureRejected)?
        .verify(message, signature)
        .map_err(|_| DiscoveryEndpointProofError::SignatureRejected)
}

fn encode_frame(kind: u8, body: &[u8]) -> Vec<u8> {
    let body_len = match kind {
        FRAME_KIND_CHALLENGE => CHALLENGE_BODY_BYTES_U16,
        FRAME_KIND_PROOF => PROOF_BODY_BYTES_U16,
        _ => 0,
    };
    debug_assert_eq!(body.len(), usize::from(body_len));
    let mut frame = Vec::with_capacity(FRAME_HEADER_BYTES + body.len());
    frame.extend_from_slice(&FRAME_MAGIC);
    frame.push(DISCOVERY_ENDPOINT_PROOF_VERSION_V1);
    frame.push(kind);
    frame.extend_from_slice(&body_len.to_be_bytes());
    frame.extend_from_slice(body);
    frame
}

fn decode_frame(
    bytes: &[u8],
    expected_kind: u8,
    expected_body_len: usize,
) -> Result<&[u8], DiscoveryEndpointProofError> {
    if bytes.len() < FRAME_HEADER_BYTES || bytes[..4] != FRAME_MAGIC {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    if bytes[4] != DISCOVERY_ENDPOINT_PROOF_VERSION_V1 || bytes[5] != expected_kind {
        return Err(DiscoveryEndpointProofError::Unsupported);
    }
    let body_len = usize::from(u16::from_be_bytes([bytes[6], bytes[7]]));
    if body_len != expected_body_len || bytes.len() != FRAME_HEADER_BYTES + body_len {
        return Err(DiscoveryEndpointProofError::Malformed);
    }
    Ok(&bytes[FRAME_HEADER_BYTES..])
}

fn take_array<const N: usize>(
    bytes: &[u8],
    offset: &mut usize,
) -> Result<[u8; N], DiscoveryEndpointProofError> {
    let end = offset
        .checked_add(N)
        .ok_or(DiscoveryEndpointProofError::Malformed)?;
    let value = bytes
        .get(*offset..end)
        .ok_or(DiscoveryEndpointProofError::Malformed)?;
    let mut array = [0; N];
    array.copy_from_slice(value);
    *offset = end;
    Ok(array)
}

fn take_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, DiscoveryEndpointProofError> {
    Ok(u64::from_be_bytes(take_array(bytes, offset)?))
}

fn domain_bytes(domain: &[u8], body: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(domain.len() + 4 + body.len());
    bytes.extend_from_slice(domain);
    let body_len = u32::try_from(body.len()).unwrap_or(u32::MAX);
    bytes.extend_from_slice(&body_len.to_be_bytes());
    bytes.extend_from_slice(body);
    bytes
}

fn domain_hash(domain: &[u8], body: &[u8]) -> [u8; 32] {
    Sha256::digest(domain_bytes(domain, body)).into()
}

#[allow(clippy::missing_const_for_fn)]
fn is_reserved<const N: usize>(value: &[u8; N]) -> bool {
    value.iter().all(|byte| *byte == 0) || value.iter().all(|byte| *byte == u8::MAX)
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_public_ipv4(ip),
        IpAddr::V6(ip) => is_public_ipv6(ip),
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    !ip.is_private()
        && !ip.is_loopback()
        && !ip.is_link_local()
        && !ip.is_multicast()
        && !ip.is_broadcast()
        && !ip.is_unspecified()
        && octets[0] != 0
        && !(octets[0] == 100 && (64..=127).contains(&octets[1]))
        && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        && !(octets[0] == 198 && (18..=19).contains(&octets[1]))
        && !(octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        && !(octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
        && octets[0] < 240
}

#[allow(clippy::missing_const_for_fn)]
fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    (segments[0] & 0xe000) == 0x2000 && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000;

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed test key")
    }

    fn fixture() -> (
        DiscoveryEndpointChallengeV1,
        DiscoveryEndpointProofV1,
        [u8; 32],
    ) {
        let challenger = key(7);
        let target = key(9);
        let context = [0x44; 32];
        let challenge = DiscoveryEndpointChallengeV1::issue(
            target.public_key_bytes(),
            [0x22; 32],
            canonical_public_endpoint_commitment("8.8.8.8:51820").expect("public endpoint"),
            [0x33; 32],
            context,
            NOW,
            NOW + 120,
            &challenger,
        )
        .expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &context, NOW + 1, &target)
            .expect("proof");
        (challenge, proof, context)
    }

    fn stage_c_inner(
        operation: DiscoveryEndpointTransportOperationV1,
        request_id: [u8; 32],
        body_tail: &[u8],
    ) -> Vec<u8> {
        let body_len = 32 + body_tail.len();
        let mut frame = Vec::with_capacity(STAGE_C_INNER_HEADER_BYTES + body_len);
        frame.extend_from_slice(&STAGE_C_INNER_MAGIC);
        frame.push(STAGE_C_INNER_VERSION_V1);
        frame.push(operation as u8);
        frame.extend_from_slice(
            &u16::try_from(body_len)
                .expect("bounded Stage C body")
                .to_be_bytes(),
        );
        frame.extend_from_slice(&request_id);
        frame.extend_from_slice(body_tail);
        frame
    }

    fn transport_fixture() -> DiscoveryEndpointAuthenticatedTransportV1 {
        let target = key(9);
        let request_id = [0x11; 32];
        DiscoveryEndpointAuthenticatedTransportV1::sign(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            [0x22; 32],
            canonical_public_endpoint_commitment("8.8.8.8:51820").expect("endpoint commitment"),
            [0x44; 32],
            NOW,
            NOW + 60,
            &stage_c_inner(
                DiscoveryEndpointTransportOperationV1::Issue,
                request_id,
                &[0x55; 65],
            ),
            &target,
        )
        .expect("authenticated transport")
    }

    #[test]
    fn canonical_roundtrip_and_golden_digests() {
        let (challenge, proof, context) = fixture();
        let challenge_frame = challenge.encode();
        let proof_frame = proof.encode();
        assert_eq!(challenge_frame.len(), CHALLENGE_FRAME_BYTES);
        assert_eq!(proof_frame.len(), PROOF_FRAME_BYTES);
        assert_eq!(
            DiscoveryEndpointChallengeV1::decode(&challenge_frame),
            Ok(challenge.clone())
        );
        assert_eq!(
            DiscoveryEndpointProofV1::decode(&proof_frame),
            Ok(proof.clone())
        );
        assert_eq!(challenge.verify_at(NOW + 1, &context), Ok(()));
        assert_eq!(
            proof.verify_for_challenge(&challenge, NOW + 1, &context),
            Ok(())
        );

        assert_eq!(
            hex::encode(Sha256::digest(&challenge_frame)),
            "9df5242b1d632e564388315c5d4f1b9ffbca6f4de051ea2a44c0588e7d1a22b9"
        );
        assert_eq!(
            hex::encode(Sha256::digest(&proof_frame)),
            "fe2be71650d32d9cfc3f6138aff52be5169e6108422674bfd1cab4a4f06b90e3"
        );
    }

    #[test]
    fn authenticated_transport_roundtrip_and_golden_digest() {
        let transport = transport_fixture();
        let frame = transport.encode();
        assert_eq!(frame.len(), 387);
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::decode(&frame),
            Ok(transport.clone())
        );
        assert_eq!(
            transport.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &key(9).public_key_bytes(),
                &[0x44; 32],
            ),
            Ok(())
        );
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::decode_verified_at(
                &frame,
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &key(9).public_key_bytes(),
                &[0x44; 32],
            ),
            Ok(transport.clone())
        );
        assert_eq!(
            transport.operation().path(),
            "/api/discovery/endpoint-proof/challenge"
        );
        assert_ne!(
            DiscoveryEndpointTransportOperationV1::Issue.path(),
            DiscoveryEndpointTransportOperationV1::Verify.path()
        );
        assert_eq!(
            hex::encode(Sha256::digest(&frame)),
            "abfeaa957026694b4efe072ff5b8f5eb1f3d4dd6ebdbbc8de71e84047318832e"
        );
        assert_eq!(
            hex::encode(transport.commitment()),
            "4da5cebe7e690e982f44c06de996577ddcc70fe6be392d703a3d01406b8ba917"
        );
    }

    #[test]
    fn authenticated_transport_field_and_inner_tamper_fail_closed() {
        let target_id = key(9).public_key_bytes();
        let mut request_changed = transport_fixture();
        request_changed.request_id[0] ^= 1;
        request_changed.inner_frame[STAGE_C_INNER_HEADER_BYTES] ^= 1;
        request_changed.inner_commitment = domain_hash(
            TRANSPORT_INNER_COMMITMENT_DOMAIN,
            &request_changed.inner_frame,
        );
        assert_eq!(
            request_changed.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &target_id,
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::SignatureRejected)
        );

        let mut descriptor_changed = transport_fixture();
        descriptor_changed.descriptor_commitment[0] ^= 1;
        assert_eq!(
            descriptor_changed.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &target_id,
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::SignatureRejected)
        );

        let mut endpoint_changed = transport_fixture();
        endpoint_changed.endpoint_commitment[0] ^= 1;
        assert_eq!(
            endpoint_changed.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &target_id,
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::SignatureRejected)
        );

        let mut operation_changed = transport_fixture();
        operation_changed.operation = DiscoveryEndpointTransportOperationV1::Verify;
        operation_changed.inner_frame[5] = DiscoveryEndpointTransportOperationV1::Verify as u8;
        operation_changed.inner_commitment = domain_hash(
            TRANSPORT_INNER_COMMITMENT_DOMAIN,
            &operation_changed.inner_frame,
        );
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::decode(&operation_changed.encode())
                .expect("structural operation substitution")
                .verify_at(
                    NOW + 1,
                    DiscoveryEndpointTransportOperationV1::Verify,
                    &target_id,
                    &[0x44; 32],
                ),
            Err(DiscoveryEndpointProofError::SignatureRejected)
        );

        let mut inner_changed = transport_fixture();
        *inner_changed.inner_frame.last_mut().expect("inner byte") ^= 1;
        inner_changed.inner_commitment = domain_hash(
            TRANSPORT_INNER_COMMITMENT_DOMAIN,
            &inner_changed.inner_frame,
        );
        assert_eq!(
            inner_changed.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &target_id,
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::SignatureRejected)
        );
    }

    #[test]
    fn authenticated_transport_context_target_time_and_signature_fail_closed() {
        let transport = transport_fixture();
        assert_eq!(
            transport.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Verify,
                &key(9).public_key_bytes(),
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::ContextMismatch)
        );
        assert_eq!(
            transport.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &key(10).public_key_bytes(),
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::ContextMismatch)
        );
        assert_eq!(
            transport.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &key(9).public_key_bytes(),
                &[0x45; 32],
            ),
            Err(DiscoveryEndpointProofError::ContextMismatch)
        );
        assert_eq!(
            transport.verify_at(
                NOW + 61,
                DiscoveryEndpointTransportOperationV1::Issue,
                &key(9).public_key_bytes(),
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::NotCurrentlyValid)
        );

        let mut future = transport.clone();
        future.issued_at = NOW + DISCOVERY_ENDPOINT_TRANSPORT_FUTURE_SKEW_SECS + 2;
        future.expires_at = future.issued_at + 10;
        assert_eq!(
            future.verify_at(
                NOW,
                DiscoveryEndpointTransportOperationV1::Issue,
                &key(9).public_key_bytes(),
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::NotCurrentlyValid)
        );

        let mut bad_signature = transport;
        bad_signature.signature[0] ^= 1;
        assert_eq!(
            bad_signature.verify_at(
                NOW + 1,
                DiscoveryEndpointTransportOperationV1::Issue,
                &key(9).public_key_bytes(),
                &[0x44; 32],
            ),
            Err(DiscoveryEndpointProofError::SignatureRejected)
        );
    }

    #[test]
    fn authenticated_transport_unknown_trailing_oversize_and_reserved_fail_closed() {
        let transport = transport_fixture();
        let mut unknown = transport.encode();
        unknown[5] = 3;
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::decode(&unknown),
            Err(DiscoveryEndpointProofError::Unsupported)
        );
        let mut wrong_version = transport.encode();
        wrong_version[4] = 2;
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::decode(&wrong_version),
            Err(DiscoveryEndpointProofError::Unsupported)
        );
        let mut trailing = transport.encode();
        trailing.push(0);
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::decode(&trailing),
            Err(DiscoveryEndpointProofError::Malformed)
        );
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::decode(&vec![
                0;
                DISCOVERY_ENDPOINT_TRANSPORT_MAX_FRAME_BYTES_V1
                    + 1
            ]),
            Err(DiscoveryEndpointProofError::Malformed)
        );

        let request_id = [0x11; 32];
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::sign(
                DiscoveryEndpointTransportOperationV1::Issue,
                request_id,
                [0x22; 32],
                [0x33; 32],
                [0; 32],
                NOW,
                NOW + 60,
                &stage_c_inner(
                    DiscoveryEndpointTransportOperationV1::Issue,
                    request_id,
                    &[0x55; 65],
                ),
                &key(9),
            ),
            Err(DiscoveryEndpointProofError::Malformed)
        );
        let oversized_inner = stage_c_inner(
            DiscoveryEndpointTransportOperationV1::Issue,
            request_id,
            &vec![0; DISCOVERY_ENDPOINT_TRANSPORT_MAX_INNER_BYTES_V1],
        );
        assert!(oversized_inner.len() > DISCOVERY_ENDPOINT_TRANSPORT_MAX_INNER_BYTES_V1);
        assert_eq!(
            DiscoveryEndpointAuthenticatedTransportV1::sign(
                DiscoveryEndpointTransportOperationV1::Issue,
                request_id,
                [0x22; 32],
                [0x33; 32],
                [0x44; 32],
                NOW,
                NOW + 60,
                &oversized_inner,
                &key(9),
            ),
            Err(DiscoveryEndpointProofError::Malformed)
        );
    }

    #[test]
    fn tamper_unknown_version_and_trailing_data_fail_closed() {
        let (challenge, proof, context) = fixture();
        let mut tampered = challenge.encode();
        tampered[40] ^= 1;
        let decoded = DiscoveryEndpointChallengeV1::decode(&tampered).expect("structural decode");
        assert_eq!(
            decoded.verify_at(NOW + 1, &context),
            Err(DiscoveryEndpointProofError::SignatureRejected)
        );

        let mut wrong_version = proof.encode();
        wrong_version[4] = 2;
        assert_eq!(
            DiscoveryEndpointProofV1::decode(&wrong_version),
            Err(DiscoveryEndpointProofError::Unsupported)
        );
        let mut trailing = proof.encode();
        trailing.push(0);
        assert_eq!(
            DiscoveryEndpointProofV1::decode(&trailing),
            Err(DiscoveryEndpointProofError::Malformed)
        );
        let mut sentinel_signature = proof.encode();
        let signature_start = sentinel_signature.len() - 64;
        sentinel_signature[signature_start..].fill(0);
        assert_eq!(
            DiscoveryEndpointProofV1::decode(&sentinel_signature),
            Err(DiscoveryEndpointProofError::Malformed)
        );
    }

    #[test]
    fn ttl_sentinel_and_canonical_endpoint_rules_fail_closed() {
        let challenger = key(7);
        let target = key(9);
        let endpoint = canonical_public_endpoint_commitment("[2606:4700:4700::1111]:51820")
            .expect("canonical public IPv6");
        assert_eq!(
            DiscoveryEndpointChallengeV1::issue(
                target.public_key_bytes(),
                [1; 32],
                endpoint,
                [2; 32],
                [3; 32],
                NOW,
                NOW + DISCOVERY_ENDPOINT_CHALLENGE_MAX_TTL_SECS + 1,
                &challenger,
            ),
            Err(DiscoveryEndpointProofError::Malformed)
        );
        assert_eq!(
            DiscoveryEndpointChallengeV1::issue(
                target.public_key_bytes(),
                [0; 32],
                endpoint,
                [2; 32],
                [3; 32],
                NOW,
                NOW + 1,
                &challenger,
            ),
            Err(DiscoveryEndpointProofError::Malformed)
        );
        assert_eq!(
            canonical_public_endpoint_commitment("127.0.0.1:51820"),
            Err(DiscoveryEndpointProofError::Malformed)
        );
        assert_eq!(
            canonical_public_endpoint_commitment("[::ffff:8.8.8.8]:51820"),
            Err(DiscoveryEndpointProofError::Malformed)
        );
        assert_eq!(
            canonical_public_endpoint_commitment("[2606:4700:4700:0:0:0:0:1111]:51820"),
            Err(DiscoveryEndpointProofError::Malformed)
        );

        let (challenge, _, context) = fixture();
        assert_eq!(
            challenge.verify_at(NOW + 121, &context),
            Err(DiscoveryEndpointProofError::NotCurrentlyValid)
        );
        assert_eq!(
            DiscoveryEndpointChallengeV1::issue(
                target.public_key_bytes(),
                [1; 32],
                endpoint,
                [u8::MAX; 32],
                [3; 32],
                NOW,
                NOW + 1,
                &challenger,
            ),
            Err(DiscoveryEndpointProofError::Malformed)
        );
    }

    #[test]
    fn proof_cannot_cross_challenger_contexts_or_challenges() {
        let (challenge, proof, context) = fixture();
        let other_context = [0x45; 32];
        assert_eq!(
            proof.verify_for_challenge(&challenge, NOW + 1, &other_context),
            Err(DiscoveryEndpointProofError::ContextMismatch)
        );

        let challenger = key(7);
        let target = key(9);
        let other = DiscoveryEndpointChallengeV1::issue(
            target.public_key_bytes(),
            challenge.descriptor_commitment(),
            challenge.endpoint_commitment(),
            [0x34; 32],
            context,
            NOW,
            NOW + 120,
            &challenger,
        )
        .expect("other challenge");
        assert_eq!(
            proof.verify_for_challenge(&other, NOW + 1, &context),
            Err(DiscoveryEndpointProofError::ContextMismatch)
        );
    }
}
