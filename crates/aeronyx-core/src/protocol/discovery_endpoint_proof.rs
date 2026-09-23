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
