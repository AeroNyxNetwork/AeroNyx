//! Canonical observer attestations for verified endpoint-possession evidence.
//!
//! One attestation proves only that one named observer verified one exact
//! endpoint proof. It grants no directory, routing, ranking, quorum, consensus,
//! reputation, or economic authority.

use std::fmt;
use std::net::{IpAddr, SocketAddr};

use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor};
use crate::protocol::discovery_endpoint_proof::{
    canonical_public_endpoint_commitment, DiscoveryEndpointChallengeV1, DiscoveryEndpointProofV1,
};

// [PERMISSIONLESS-ENDPOINT-ATTESTATION 2026-09-24 by Codex] Freeze this
// authority-neutral wire independently from discovery and MemChain transports.
const FRAME_MAGIC: [u8; 4] = *b"ADAT";
const FRAME_HEADER_BYTES: usize = 8;
const FRAME_KIND_ATTESTATION: u8 = 1;
const BODY_BYTES: usize = 281;
const UNSIGNED_BODY_BYTES: usize = BODY_BYTES - 64;
const SIGNATURE_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointEvidenceAttestationV1\0";
const COMMITMENT_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointEvidenceAttestationCommitmentV1\0";
const EVIDENCE_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointEvidenceV1\0";

/// Frozen endpoint-evidence attestation version.
pub const DISCOVERY_ENDPOINT_ATTESTATION_VERSION_V1: u8 = 1;
/// Exact canonical encoded length of one V1 attestation.
pub const DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1: usize = 289;
/// Maximum lifetime of one V1 attestation, in seconds.
pub const DISCOVERY_ENDPOINT_ATTESTATION_MAX_TTL_SECS: u64 = 24 * 60 * 60;
/// Maximum accepted positive clock skew for the observation time.
pub const DISCOVERY_ENDPOINT_ATTESTATION_FUTURE_SKEW_SECS: u64 = 30;

/// Derives the exact public IP socket committed by a signed descriptor's
/// endpoint. Bare canonical sockets and credential-free HTTP(S) authorities
/// share one ADEA commitment; URL paths, queries, fragments, DNS, and
/// noncanonical IP spellings never enter the signed evidence domain.
///
/// # Errors
/// Returns a coarse malformed error for any ambiguous or non-public target.
// [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Keep URL parsing
// and the existing SocketAddr commitment distinct; the attestation wire and
// domain transcript do not change.
pub fn canonical_attested_public_endpoint_socket_v1(
    endpoint: &str,
) -> Result<SocketAddr, DiscoveryEndpointAttestationError> {
    if endpoint.is_empty() || endpoint.len() > 80 || endpoint.trim() != endpoint {
        return Err(DiscoveryEndpointAttestationError::Malformed);
    }
    let (authority, default_port) = if let Some(authority) = endpoint.strip_prefix("http://") {
        (authority, Some(80))
    } else if let Some(authority) = endpoint.strip_prefix("https://") {
        (authority, Some(443))
    } else {
        (endpoint, None)
    };
    if authority.is_empty()
        || authority
            .bytes()
            .any(|byte| matches!(byte, b'/' | b'?' | b'#' | b'@' | b'\\'))
    {
        return Err(DiscoveryEndpointAttestationError::Malformed);
    }
    let socket = if let Ok(socket) = authority.parse::<SocketAddr>() {
        if socket.to_string() != authority {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        }
        socket
    } else {
        let Some(default_port) = default_port else {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        };
        let host = if let Some(inner) = authority
            .strip_prefix('[')
            .and_then(|value| value.strip_suffix(']'))
        {
            inner
        } else {
            authority
        };
        let ip: IpAddr = host
            .parse()
            .map_err(|_| DiscoveryEndpointAttestationError::Malformed)?;
        let canonical_host = match ip {
            IpAddr::V4(_) => ip.to_string(),
            IpAddr::V6(_) => format!("[{ip}]"),
        };
        if canonical_host != authority {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        }
        SocketAddr::new(ip, default_port)
    };
    canonical_public_endpoint_commitment(&socket.to_string())
        .map_err(|_| DiscoveryEndpointAttestationError::Malformed)?;
    Ok(socket)
}

/// Closed purpose domain for endpoint evidence attestations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DiscoveryEndpointAttestationPurposeV1 {
    /// Reports verification of one endpoint-possession proof.
    EndpointPossessionObservation = 1,
}

impl DiscoveryEndpointAttestationPurposeV1 {
    #[allow(clippy::missing_const_for_fn)]
    fn from_u8(value: u8) -> Result<Self, DiscoveryEndpointAttestationError> {
        match value {
            1 => Ok(Self::EndpointPossessionObservation),
            _ => Err(DiscoveryEndpointAttestationError::Unsupported),
        }
    }
}

/// Coarse endpoint-attestation validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryEndpointAttestationError {
    /// A length, field, timestamp, or canonical representation is malformed.
    Malformed,
    /// The version, kind, or purpose is unsupported.
    Unsupported,
    /// The attestation is outside its accepted time window.
    NotCurrentlyValid,
    /// A caller-selected exact binding differs.
    ContextMismatch,
    /// An Ed25519 signature did not verify.
    SignatureRejected,
}

impl fmt::Display for DiscoveryEndpointAttestationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Malformed => "discovery endpoint attestation is malformed",
            Self::Unsupported => "discovery endpoint attestation is unsupported",
            Self::NotCurrentlyValid => "discovery endpoint attestation is not currently valid",
            Self::ContextMismatch => "discovery endpoint attestation context does not match",
            Self::SignatureRejected => "discovery endpoint attestation signature was rejected",
        })
    }
}

impl std::error::Error for DiscoveryEndpointAttestationError {}

/// One observer-signed statement about exact endpoint-possession evidence.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointEvidenceAttestationV1 {
    subject_node_id: [u8; 32],
    descriptor_sequence: u64,
    descriptor_hash: [u8; 32],
    endpoint_commitment: [u8; 32],
    evidence_commitment: [u8; 32],
    observer_node_id: [u8; 32],
    observed_at: u64,
    expires_at: u64,
    context: [u8; 32],
    purpose: DiscoveryEndpointAttestationPurposeV1,
    signature: [u8; 64],
}

impl fmt::Debug for DiscoveryEndpointEvidenceAttestationV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointEvidenceAttestationV1")
            .field("purpose", &self.purpose)
            .field("observed_at", &self.observed_at)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

impl DiscoveryEndpointEvidenceAttestationV1 {
    /// Signs an attestation after verifying the exact challenge/proof pair.
    ///
    /// # Errors
    /// Returns a coarse error when proof, descriptor, observer, context, time,
    /// or canonical evidence validation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_from_verified_proof(
        signed_descriptor: &SignedNodeDescriptor,
        challenge: &DiscoveryEndpointChallengeV1,
        proof: &DiscoveryEndpointProofV1,
        context: [u8; 32],
        purpose: DiscoveryEndpointAttestationPurposeV1,
        observed_at: u64,
        expires_at: u64,
        observer: &IdentityKeyPair,
    ) -> Result<Self, DiscoveryEndpointAttestationError> {
        proof
            .verify_for_challenge(challenge, observed_at, &context)
            .map_err(|_| DiscoveryEndpointAttestationError::ContextMismatch)?;
        let descriptor = DirectoryDescriptorCommitmentV1::from_signed_descriptor(signed_descriptor)
            .map_err(|_| DiscoveryEndpointAttestationError::SignatureRejected)?;
        let descriptor_endpoint = signed_descriptor
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(DiscoveryEndpointAttestationError::Malformed)?;
        let descriptor_socket = canonical_attested_public_endpoint_socket_v1(descriptor_endpoint)?;
        let descriptor_endpoint_commitment =
            canonical_public_endpoint_commitment(&descriptor_socket.to_string())
                .map_err(|_| DiscoveryEndpointAttestationError::Malformed)?;
        let observer_node_id = observer.public_key_bytes();
        if observer_node_id != challenge.challenger_node_id()
            || descriptor.node_id != challenge.target_node_id()
            || descriptor.descriptor_hash != challenge.descriptor_commitment()
            || descriptor_endpoint_commitment != challenge.endpoint_commitment()
        {
            return Err(DiscoveryEndpointAttestationError::ContextMismatch);
        }
        let mut attestation = Self {
            subject_node_id: descriptor.node_id,
            descriptor_sequence: descriptor.sequence,
            descriptor_hash: descriptor.descriptor_hash,
            endpoint_commitment: challenge.endpoint_commitment(),
            evidence_commitment: discovery_endpoint_evidence_commitment_v1(challenge, proof),
            observer_node_id,
            observed_at,
            expires_at,
            context,
            purpose,
            signature: [0; 64],
        };
        attestation.validate_claims()?;
        validate_time(observed_at, expires_at, observed_at)?;
        attestation.signature = observer.sign(&attestation.signing_bytes());
        Ok(attestation)
    }

    /// Verifies signature, freshness, and every caller-selected exact binding.
    ///
    /// # Errors
    /// Returns a coarse error when any claim, binding, time, or signature fails.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_at(
        &self,
        now: u64,
        expected_observer: &[u8; 32],
        expected_descriptor: &DirectoryDescriptorCommitmentV1,
        expected_endpoint_commitment: &[u8; 32],
        expected_evidence_commitment: &[u8; 32],
        expected_context: &[u8; 32],
        expected_purpose: DiscoveryEndpointAttestationPurposeV1,
    ) -> Result<(), DiscoveryEndpointAttestationError> {
        self.validate_claims()?;
        validate_time(self.observed_at, self.expires_at, now)?;
        if self.observer_node_id != *expected_observer
            || self.subject_node_id != expected_descriptor.node_id
            || self.descriptor_sequence != expected_descriptor.sequence
            || self.descriptor_hash != expected_descriptor.descriptor_hash
            || self.endpoint_commitment != *expected_endpoint_commitment
            || self.evidence_commitment != *expected_evidence_commitment
            || self.context != *expected_context
            || self.purpose != expected_purpose
        {
            return Err(DiscoveryEndpointAttestationError::ContextMismatch);
        }
        verify_signature(
            &self.observer_node_id,
            &self.signing_bytes(),
            &self.signature,
        )
    }

    /// Encodes the exact fixed-width canonical frame.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1);
        bytes.extend_from_slice(&FRAME_MAGIC);
        bytes.push(DISCOVERY_ENDPOINT_ATTESTATION_VERSION_V1);
        bytes.push(FRAME_KIND_ATTESTATION);
        bytes.extend_from_slice(&281u16.to_be_bytes());
        bytes.extend_from_slice(&self.unsigned_body());
        bytes.extend_from_slice(&self.signature);
        bytes
    }

    /// Decodes one exact canonical V1 frame.
    ///
    /// # Errors
    /// Returns a coarse error for unsupported, malformed, trailing, or
    /// non-canonical input.
    pub fn decode(bytes: &[u8]) -> Result<Self, DiscoveryEndpointAttestationError> {
        if bytes.len() != DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1
            || bytes.get(..4) != Some(FRAME_MAGIC.as_slice())
        {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        }
        if bytes[4] != DISCOVERY_ENDPOINT_ATTESTATION_VERSION_V1
            || bytes[5] != FRAME_KIND_ATTESTATION
        {
            return Err(DiscoveryEndpointAttestationError::Unsupported);
        }
        if usize::from(u16::from_be_bytes([bytes[6], bytes[7]])) != BODY_BYTES {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        }
        let body = &bytes[FRAME_HEADER_BYTES..];
        let mut offset = 0;
        let attestation = Self {
            subject_node_id: take_array(body, &mut offset)?,
            descriptor_sequence: take_u64(body, &mut offset)?,
            descriptor_hash: take_array(body, &mut offset)?,
            endpoint_commitment: take_array(body, &mut offset)?,
            evidence_commitment: take_array(body, &mut offset)?,
            observer_node_id: take_array(body, &mut offset)?,
            observed_at: take_u64(body, &mut offset)?,
            expires_at: take_u64(body, &mut offset)?,
            context: take_array(body, &mut offset)?,
            purpose: DiscoveryEndpointAttestationPurposeV1::from_u8(take_u8(body, &mut offset)?)?,
            signature: take_array(body, &mut offset)?,
        };
        if offset != body.len() || is_reserved(&attestation.signature) {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        }
        attestation.validate_claims()?;
        if attestation.encode() != bytes {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        }
        Ok(attestation)
    }

    /// Returns the subject node identity.
    #[must_use]
    pub const fn subject_node_id(&self) -> [u8; 32] {
        self.subject_node_id
    }
    /// Returns the exact descriptor sequence.
    #[must_use]
    pub const fn descriptor_sequence(&self) -> u64 {
        self.descriptor_sequence
    }
    /// Returns the exact descriptor hash.
    #[must_use]
    pub const fn descriptor_hash(&self) -> [u8; 32] {
        self.descriptor_hash
    }
    /// Returns the endpoint commitment without endpoint text.
    #[must_use]
    pub const fn endpoint_commitment(&self) -> [u8; 32] {
        self.endpoint_commitment
    }
    /// Returns the canonical challenge/proof evidence commitment.
    #[must_use]
    pub const fn evidence_commitment(&self) -> [u8; 32] {
        self.evidence_commitment
    }
    /// Returns the observer node identity.
    #[must_use]
    pub const fn observer_node_id(&self) -> [u8; 32] {
        self.observer_node_id
    }
    /// Returns the observation time.
    #[must_use]
    pub const fn observed_at(&self) -> u64 {
        self.observed_at
    }
    /// Returns the expiry time.
    #[must_use]
    pub const fn expires_at(&self) -> u64 {
        self.expires_at
    }
    /// Returns the anti-replay context.
    #[must_use]
    pub const fn context(&self) -> [u8; 32] {
        self.context
    }
    /// Returns the closed attestation purpose.
    #[must_use]
    pub const fn purpose(&self) -> DiscoveryEndpointAttestationPurposeV1 {
        self.purpose
    }

    /// Returns a commitment to the exact signed canonical frame.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        domain_hash(COMMITMENT_DOMAIN, &self.encode())
    }

    fn validate_claims(&self) -> Result<(), DiscoveryEndpointAttestationError> {
        if [
            &self.subject_node_id,
            &self.descriptor_hash,
            &self.endpoint_commitment,
            &self.evidence_commitment,
            &self.observer_node_id,
            &self.context,
        ]
        .into_iter()
        .any(is_reserved)
            || self.descriptor_sequence == 0
            || self.observed_at == 0
            || self.expires_at <= self.observed_at
            || self.expires_at - self.observed_at > DISCOVERY_ENDPOINT_ATTESTATION_MAX_TTL_SECS
        {
            return Err(DiscoveryEndpointAttestationError::Malformed);
        }
        Ok(())
    }

    fn unsigned_body(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(UNSIGNED_BODY_BYTES);
        bytes.extend_from_slice(&self.subject_node_id);
        bytes.extend_from_slice(&self.descriptor_sequence.to_be_bytes());
        bytes.extend_from_slice(&self.descriptor_hash);
        bytes.extend_from_slice(&self.endpoint_commitment);
        bytes.extend_from_slice(&self.evidence_commitment);
        bytes.extend_from_slice(&self.observer_node_id);
        bytes.extend_from_slice(&self.observed_at.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at.to_be_bytes());
        bytes.extend_from_slice(&self.context);
        bytes.push(self.purpose as u8);
        bytes
    }

    fn signing_bytes(&self) -> Vec<u8> {
        domain_bytes(SIGNATURE_DOMAIN, &self.unsigned_body())
    }
}

/// Derives the frozen commitment to one exact canonical challenge/proof pair.
#[must_use]
pub fn discovery_endpoint_evidence_commitment_v1(
    challenge: &DiscoveryEndpointChallengeV1,
    proof: &DiscoveryEndpointProofV1,
) -> [u8; 32] {
    let challenge = challenge.encode();
    let proof = proof.encode();
    let mut body = Vec::with_capacity(8 + challenge.len() + proof.len());
    body.extend_from_slice(
        &u32::try_from(challenge.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    body.extend_from_slice(&challenge);
    body.extend_from_slice(&u32::try_from(proof.len()).unwrap_or(u32::MAX).to_be_bytes());
    body.extend_from_slice(&proof);
    domain_hash(EVIDENCE_DOMAIN, &body)
}

const fn validate_time(
    observed_at: u64,
    expires_at: u64,
    now: u64,
) -> Result<(), DiscoveryEndpointAttestationError> {
    if observed_at > now.saturating_add(DISCOVERY_ENDPOINT_ATTESTATION_FUTURE_SKEW_SECS)
        || now > expires_at
    {
        Err(DiscoveryEndpointAttestationError::NotCurrentlyValid)
    } else {
        Ok(())
    }
}

fn verify_signature(
    public_key: &[u8; 32],
    message: &[u8],
    signature: &[u8; 64],
) -> Result<(), DiscoveryEndpointAttestationError> {
    IdentityPublicKey::from_bytes(public_key)
        .map_err(|_| DiscoveryEndpointAttestationError::SignatureRejected)?
        .verify(message, signature)
        .map_err(|_| DiscoveryEndpointAttestationError::SignatureRejected)
}

fn take_array<const N: usize>(
    bytes: &[u8],
    offset: &mut usize,
) -> Result<[u8; N], DiscoveryEndpointAttestationError> {
    let end = offset
        .checked_add(N)
        .ok_or(DiscoveryEndpointAttestationError::Malformed)?;
    let slice = bytes
        .get(*offset..end)
        .ok_or(DiscoveryEndpointAttestationError::Malformed)?;
    let mut value = [0; N];
    value.copy_from_slice(slice);
    *offset = end;
    Ok(value)
}
fn take_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, DiscoveryEndpointAttestationError> {
    Ok(u64::from_be_bytes(take_array(bytes, offset)?))
}
fn take_u8(bytes: &[u8], offset: &mut usize) -> Result<u8, DiscoveryEndpointAttestationError> {
    let value = *bytes
        .get(*offset)
        .ok_or(DiscoveryEndpointAttestationError::Malformed)?;
    *offset += 1;
    Ok(value)
}
fn domain_bytes(domain: &[u8], body: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(domain.len() + 4 + body.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_be_bytes());
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::protocol::discovery::NodeDescriptor;

    const NOW: u64 = 1_780_000_000;

    struct Fixture {
        observer: IdentityKeyPair,
        subject: IdentityKeyPair,
        signed_descriptor: SignedNodeDescriptor,
        descriptor: DirectoryDescriptorCommitmentV1,
        challenge: DiscoveryEndpointChallengeV1,
        proof: DiscoveryEndpointProofV1,
        context: [u8; 32],
        endpoint: [u8; 32],
        evidence: [u8; 32],
        attestation: DiscoveryEndpointEvidenceAttestationV1,
    }

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).unwrap()
    }
    fn fixture() -> Fixture {
        let observer = key(0x11);
        let subject = key(0x22);
        let context = [0x33; 32];
        let endpoint = canonical_public_endpoint_commitment("8.8.8.8:51820").unwrap();
        let mut raw =
            NodeDescriptor::new(subject.public_key_bytes(), 7, NOW - 60, NOW + 3600, "test");
        raw.public_endpoint = Some("8.8.8.8:51820".into());
        let signed_descriptor = SignedNodeDescriptor::sign(raw, &subject).unwrap();
        let descriptor =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&signed_descriptor).unwrap();
        let challenge = DiscoveryEndpointChallengeV1::issue(
            subject.public_key_bytes(),
            descriptor.descriptor_hash,
            endpoint,
            [0x44; 32],
            context,
            NOW,
            NOW + 120,
            &observer,
        )
        .unwrap();
        let proof =
            DiscoveryEndpointProofV1::respond(&challenge, &context, NOW + 1, &subject).unwrap();
        let evidence = discovery_endpoint_evidence_commitment_v1(&challenge, &proof);
        let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
            &signed_descriptor,
            &challenge,
            &proof,
            context,
            DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
            NOW + 1,
            NOW + 3601,
            &observer,
        )
        .unwrap();
        Fixture {
            observer,
            subject,
            signed_descriptor,
            descriptor,
            challenge,
            proof,
            context,
            endpoint,
            evidence,
            attestation,
        }
    }
    fn verify(
        f: &Fixture,
        value: &DiscoveryEndpointEvidenceAttestationV1,
        now: u64,
    ) -> Result<(), DiscoveryEndpointAttestationError> {
        value.verify_at(
            now,
            &f.observer.public_key_bytes(),
            &f.descriptor,
            &f.endpoint,
            &f.evidence,
            &f.context,
            DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
        )
    }

    #[test]
    fn signed_bare_http_and_https_public_ip_descriptors_share_socket_commitment() {
        let f = fixture();
        let expected = canonical_public_endpoint_commitment("8.8.8.8:51820").unwrap();
        for endpoint in [
            "8.8.8.8:51820",
            "http://8.8.8.8:51820",
            "https://8.8.8.8:51820",
        ] {
            let socket = canonical_attested_public_endpoint_socket_v1(endpoint).unwrap();
            assert_eq!(socket.to_string(), "8.8.8.8:51820");
            let mut body = f.signed_descriptor.descriptor.clone();
            body.public_endpoint = Some(endpoint.to_string());
            let signed = SignedNodeDescriptor::sign(body, &f.subject).unwrap();
            let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&signed).unwrap();
            let challenge = DiscoveryEndpointChallengeV1::issue(
                f.subject.public_key_bytes(),
                pin.descriptor_hash,
                expected,
                [0x57; 32],
                f.context,
                NOW,
                NOW + 120,
                &f.observer,
            )
            .unwrap();
            let proof =
                DiscoveryEndpointProofV1::respond(&challenge, &f.context, NOW + 1, &f.subject)
                    .unwrap();
            let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &signed,
                &challenge,
                &proof,
                f.context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                NOW + 1,
                NOW + 120,
                &f.observer,
            )
            .unwrap();
            assert_eq!(attestation.endpoint_commitment(), expected);
            assert!(attestation
                .verify_at(
                    NOW + 1,
                    &f.observer.public_key_bytes(),
                    &pin,
                    &expected,
                    &discovery_endpoint_evidence_commitment_v1(&challenge, &proof),
                    &f.context,
                    DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                )
                .is_ok());
        }
    }

    #[test]
    fn descriptor_endpoint_parser_rejects_ambiguous_or_substituted_targets() {
        for endpoint in [
            "http://example.com:51820",
            "http://127.0.0.1:51820",
            "https://user@8.8.8.8:51820",
            "https://8.8.8.8:51820/path",
            "https://8.8.8.8:51820?x=1",
            "https://8.8.8.8:51820#x",
            "https://8.8.8.8:051820",
            "HTTP://8.8.8.8:51820",
            "https://8.8.8.8:51820/",
            "8.8.8.8:0",
        ] {
            assert_eq!(
                canonical_attested_public_endpoint_socket_v1(endpoint),
                Err(DiscoveryEndpointAttestationError::Malformed)
            );
        }
        assert_eq!(
            canonical_attested_public_endpoint_socket_v1("http://8.8.8.8")
                .unwrap()
                .port(),
            80
        );
        assert_eq!(
            canonical_attested_public_endpoint_socket_v1("https://8.8.8.8")
                .unwrap()
                .port(),
            443
        );
        let f = fixture();
        let mut body = f.signed_descriptor.descriptor.clone();
        body.public_endpoint = Some("https://9.9.9.9:51820".to_string());
        let signed = SignedNodeDescriptor::sign(body, &f.subject).unwrap();
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&signed).unwrap();
        let challenge = DiscoveryEndpointChallengeV1::issue(
            f.subject.public_key_bytes(),
            pin.descriptor_hash,
            f.endpoint,
            [0x58; 32],
            f.context,
            NOW,
            NOW + 120,
            &f.observer,
        )
        .unwrap();
        let proof =
            DiscoveryEndpointProofV1::respond(&challenge, &f.context, NOW + 1, &f.subject).unwrap();
        assert_eq!(
            DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &signed,
                &challenge,
                &proof,
                f.context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                NOW + 1,
                NOW + 120,
                &f.observer,
            ),
            Err(DiscoveryEndpointAttestationError::ContextMismatch)
        );
    }

    #[test]
    fn golden_frame_and_commitments_are_frozen() {
        let f = fixture();
        let encoded = f.attestation.encode();
        assert_eq!(encoded.len(), 289);
        assert_eq!(
            hex::encode(Sha256::digest(&encoded)),
            "85014c3584338b329cfe3e117538a44d92ca9c0b89abb00e35e7907939e07091"
        );
        assert_eq!(
            hex::encode(f.evidence),
            "9552a3d2b36575c20302d80625e20713600ede3b8f904ac2b7723e7250598604"
        );
        assert_eq!(
            hex::encode(f.attestation.commitment()),
            "e97a3b3300f8063908818b2a28863a5623e6b0777f4360acc50cacd49ecd30c8"
        );
        let decoded = DiscoveryEndpointEvidenceAttestationV1::decode(&encoded).unwrap();
        assert_eq!(decoded, f.attestation);
        assert!(verify(&f, &decoded, NOW + 1).is_ok());
    }

    #[test]
    fn malformed_header_length_trailing_and_purpose_reject() {
        let encoded = fixture().attestation.encode();
        for index in [0usize, 4, 5, 6, 7] {
            let mut v = encoded.clone();
            v[index] ^= 1;
            assert!(DiscoveryEndpointEvidenceAttestationV1::decode(&v).is_err());
        }
        assert!(DiscoveryEndpointEvidenceAttestationV1::decode(&encoded[..288]).is_err());
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(DiscoveryEndpointEvidenceAttestationV1::decode(&trailing).is_err());
        let mut purpose = encoded;
        purpose[FRAME_HEADER_BYTES + UNSIGNED_BODY_BYTES - 1] = 2;
        assert_eq!(
            DiscoveryEndpointEvidenceAttestationV1::decode(&purpose),
            Err(DiscoveryEndpointAttestationError::Unsupported)
        );
    }

    #[test]
    fn issue_rejects_wrong_observer_subject_context_and_proof() {
        let f = fixture();
        let wrong = key(0x55);
        assert!(
            DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &f.signed_descriptor,
                &f.challenge,
                &f.proof,
                f.context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                NOW + 1,
                NOW + 60,
                &wrong
            )
            .is_err()
        );
        let other_subject = key(0x77);
        let mut wrong_raw = NodeDescriptor::new(
            other_subject.public_key_bytes(),
            8,
            NOW - 60,
            NOW + 3600,
            "test",
        );
        wrong_raw.public_endpoint = Some("8.8.8.8:51820".into());
        let wrong_descriptor = SignedNodeDescriptor::sign(wrong_raw, &other_subject).unwrap();
        assert!(
            DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &wrong_descriptor,
                &f.challenge,
                &f.proof,
                f.context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                NOW + 1,
                NOW + 60,
                &f.observer
            )
            .is_err()
        );
        assert!(
            DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &f.signed_descriptor,
                &f.challenge,
                &f.proof,
                [0x66; 32],
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                NOW + 1,
                NOW + 60,
                &f.observer
            )
            .is_err()
        );
        assert!(
            DiscoveryEndpointProofV1::respond(&f.challenge, &f.context, NOW + 1, &wrong).is_err()
        );
    }

    #[test]
    fn time_reserved_and_exact_binding_boundaries_fail_closed() {
        let f = fixture();
        assert!(verify(&f, &f.attestation, NOW + 3601).is_ok());
        assert!(verify(&f, &f.attestation, NOW + 3602).is_err());
        let mut future = f.attestation.clone();
        future.observed_at = NOW + 32;
        future.expires_at = NOW + 60;
        future.signature = f.observer.sign(&future.signing_bytes());
        assert!(verify(&f, &future, NOW + 1).is_err());
        let mut descriptor = f.descriptor;
        descriptor.sequence += 1;
        assert!(f
            .attestation
            .verify_at(
                NOW + 1,
                &f.observer.public_key_bytes(),
                &descriptor,
                &f.endpoint,
                &f.evidence,
                &f.context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation
            )
            .is_err());
        for fill in [0u8, u8::MAX] {
            for field in 0..6 {
                let mut v = f.attestation.clone();
                match field {
                    0 => v.subject_node_id = [fill; 32],
                    1 => v.descriptor_hash = [fill; 32],
                    2 => v.endpoint_commitment = [fill; 32],
                    3 => v.evidence_commitment = [fill; 32],
                    4 => v.observer_node_id = [fill; 32],
                    _ => v.context = [fill; 32],
                }
                assert!(DiscoveryEndpointEvidenceAttestationV1::decode(&v.encode()).is_err());
            }
        }
    }

    #[test]
    fn wrong_signer_and_every_unsigned_byte_flip_reject() {
        let f = fixture();
        let wrong = key(0x88);
        let mut resigned = f.attestation.clone();
        resigned.signature = wrong.sign(&resigned.signing_bytes());
        assert_eq!(
            verify(&f, &resigned, NOW + 1),
            Err(DiscoveryEndpointAttestationError::SignatureRejected)
        );
        let encoded = f.attestation.encode();
        for index in FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + UNSIGNED_BODY_BYTES {
            let mut v = encoded.clone();
            v[index] ^= 1;
            if let Ok(decoded) = DiscoveryEndpointEvidenceAttestationV1::decode(&v) {
                assert!(verify(&f, &decoded, NOW + 1).is_err());
            }
        }
    }

    #[test]
    fn exact_pair_changes_evidence_commitment() {
        let f = fixture();
        let subject = key(0x99);
        let challenge = DiscoveryEndpointChallengeV1::issue(
            subject.public_key_bytes(),
            [0xaa; 32],
            f.endpoint,
            [0xbb; 32],
            f.context,
            NOW,
            NOW + 120,
            &f.observer,
        )
        .unwrap();
        let proof =
            DiscoveryEndpointProofV1::respond(&challenge, &f.context, NOW + 1, &subject).unwrap();
        assert_ne!(
            discovery_endpoint_evidence_commitment_v1(&challenge, &proof),
            f.evidence
        );
        assert_ne!(f.subject.public_key_bytes(), subject.public_key_bytes());
    }
}
