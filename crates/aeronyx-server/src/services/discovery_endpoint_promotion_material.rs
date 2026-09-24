// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_promotion_material.rs
// ============================================================================
//! Private resolution of one exact endpoint-promotion candidate.
//!
//! This module converts no data into routeability. It only combines one
//! caller-held signed descriptor with a current F.7 readiness capability and
//! one exact retained attestation. There is no list, prefix, page, ranking,
//! peer-store, API, or networking surface.

use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeCapability, NodeProtocolFeature, SignedNodeDescriptor,
};
use aeronyx_core::protocol::discovery_endpoint_attestation::canonical_attested_public_endpoint_socket_v1;
use aeronyx_core::protocol::discovery_endpoint_proof::canonical_public_endpoint_commitment;

use super::discovery_endpoint_attestation_inbox::{
    DiscoveryEndpointAttestationInboxError, SqliteDiscoveryEndpointAttestationInbox,
};
use super::discovery_endpoint_quarantine_revocation::{
    DiscoveryEndpointPromotionReadiness, DiscoveryEndpointQuarantineRevocationError,
    SqliteDiscoveryEndpointQuarantineRevocationRegistry,
};

/// Coarse resolution failures that expose no node, endpoint, or commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DiscoveryEndpointPromotionMaterialError {
    #[error("endpoint promotion material rejected")]
    Rejected,
    #[error("endpoint promotion material state corrupt")]
    Corrupt,
    #[error("endpoint promotion material unavailable")]
    Unavailable,
}

/// Optional signed descriptor features required by the future private caller.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointPromotionFeatureRequirement {
    capability: Option<NodeCapability>,
    protocol_feature: Option<NodeProtocolFeature>,
}

impl DiscoveryEndpointPromotionFeatureRequirement {
    pub(crate) const fn new(
        capability: Option<NodeCapability>,
        protocol_feature: Option<NodeProtocolFeature>,
    ) -> Self {
        Self {
            capability,
            protocol_feature,
        }
    }

    fn is_satisfied_by(self, descriptor: &SignedNodeDescriptor) -> bool {
        descriptor.descriptor.policy.public_discovery
            && self.capability.map_or(true, |required| {
                descriptor.descriptor.capabilities.contains(&required)
            })
            && self.protocol_feature.map_or(true, |required| {
                descriptor.descriptor.advertises_protocol_feature(required)
            })
    }
}

/// Exact private material prepared for a future atomic promotion consumer.
///
/// Deliberately does not implement `Debug`; the descriptor contains node and
/// endpoint material that must not enter logs or generic diagnostics.
// [PERMISSIONLESS-ENDPOINT-PROMOTION-MATERIAL 2026-09-24 by Codex] Construction
// is private to the resolver after two current F.7 checks and one exact ADAT.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct VerifiedPromotionMaterial {
    descriptor: SignedNodeDescriptor,
    descriptor_commitment: DirectoryDescriptorCommitmentV1,
    endpoint_commitment: [u8; 32],
    group_commitment: [u8; 32],
    readiness_commitment: [u8; 32],
    resolved_at: u64,
    valid_until: u64,
}

impl VerifiedPromotionMaterial {
    #[cfg(test)]
    pub(crate) fn test_only_from_descriptor(
        descriptor: SignedNodeDescriptor,
        resolved_at: u64,
        valid_until: u64,
    ) -> Option<Self> {
        let descriptor_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).ok()?;
        Some(Self {
            descriptor,
            descriptor_commitment,
            endpoint_commitment: [1; 32],
            group_commitment: [2; 32],
            readiness_commitment: [3; 32],
            resolved_at,
            valid_until,
        })
    }

    pub(crate) const fn descriptor(&self) -> &SignedNodeDescriptor {
        &self.descriptor
    }

    pub(crate) const fn descriptor_commitment(&self) -> DirectoryDescriptorCommitmentV1 {
        self.descriptor_commitment
    }

    pub(crate) const fn endpoint_commitment(&self) -> [u8; 32] {
        self.endpoint_commitment
    }

    pub(crate) const fn group_commitment(&self) -> [u8; 32] {
        self.group_commitment
    }

    pub(crate) const fn readiness_commitment(&self) -> [u8; 32] {
        self.readiness_commitment
    }

    pub(crate) const fn resolved_at(&self) -> u64 {
        self.resolved_at
    }

    pub(crate) const fn valid_until(&self) -> u64 {
        self.valid_until
    }
}

/// Resolves exactly one caller-supplied descriptor without exposing a candidate
/// enumeration surface.
pub(crate) struct DiscoveryEndpointPromotionMaterialResolver<'a> {
    inbox: &'a SqliteDiscoveryEndpointAttestationInbox,
    revocations: &'a SqliteDiscoveryEndpointQuarantineRevocationRegistry,
}

impl<'a> DiscoveryEndpointPromotionMaterialResolver<'a> {
    pub(crate) const fn new(
        inbox: &'a SqliteDiscoveryEndpointAttestationInbox,
        revocations: &'a SqliteDiscoveryEndpointQuarantineRevocationRegistry,
    ) -> Self {
        Self { inbox, revocations }
    }

    pub(crate) fn resolve_at(
        &self,
        readiness: &DiscoveryEndpointPromotionReadiness,
        descriptor: &SignedNodeDescriptor,
        features: DiscoveryEndpointPromotionFeatureRequirement,
        now: u64,
    ) -> Result<Option<VerifiedPromotionMaterial>, DiscoveryEndpointPromotionMaterialError> {
        self.resolve_with_hook_at(readiness, descriptor, features, now, || {})
    }

    fn resolve_with_hook_at<F>(
        &self,
        readiness: &DiscoveryEndpointPromotionReadiness,
        descriptor: &SignedNodeDescriptor,
        features: DiscoveryEndpointPromotionFeatureRequirement,
        now: u64,
        before_final_readiness_check: F,
    ) -> Result<Option<VerifiedPromotionMaterial>, DiscoveryEndpointPromotionMaterialError>
    where
        F: FnOnce(),
    {
        if now == 0
            || !self
                .revocations
                .verify_promotion_readiness_at(readiness, now)
                .map_err(map_revocation_error)?
        {
            return Ok(None);
        }
        if descriptor.verify_at(now).is_err() || !features.is_satisfied_by(descriptor) {
            return Ok(None);
        }
        let Ok(descriptor_commitment) =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
        else {
            return Ok(None);
        };
        let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
            return Ok(None);
        };
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Resolve
        // the signed HTTP authority through the same canonical public socket
        // as ADEA and the direct dialer, never by hashing URL text as a socket.
        let Ok(socket) = canonical_attested_public_endpoint_socket_v1(endpoint) else {
            return Ok(None);
        };
        let Ok(endpoint_commitment) = canonical_public_endpoint_commitment(&socket.to_string())
        else {
            return Ok(None);
        };
        if descriptor_commitment.sequence != readiness.descriptor_sequence()
            || !self
                .inbox
                .contains_exact_candidate_at(
                    descriptor_commitment.node_id,
                    descriptor_commitment.sequence,
                    descriptor_commitment.descriptor_hash,
                    endpoint_commitment,
                    readiness.group_commitment(),
                    now,
                )
                .map_err(map_inbox_error)?
        {
            return Ok(None);
        }

        before_final_readiness_check();
        if !self
            .revocations
            .verify_promotion_readiness_at(readiness, now)
            .map_err(map_revocation_error)?
        {
            return Ok(None);
        }
        Ok(Some(VerifiedPromotionMaterial {
            descriptor: descriptor.clone(),
            descriptor_commitment,
            endpoint_commitment,
            group_commitment: readiness.group_commitment(),
            readiness_commitment: readiness.readiness_commitment(),
            resolved_at: now,
            valid_until: readiness
                .valid_until()
                .min(descriptor.descriptor.expires_at),
        }))
    }
}

const fn map_inbox_error(
    error: DiscoveryEndpointAttestationInboxError,
) -> DiscoveryEndpointPromotionMaterialError {
    match error {
        DiscoveryEndpointAttestationInboxError::Rejected => {
            DiscoveryEndpointPromotionMaterialError::Rejected
        }
        DiscoveryEndpointAttestationInboxError::Corrupt
        | DiscoveryEndpointAttestationInboxError::UnsupportedSchema => {
            DiscoveryEndpointPromotionMaterialError::Corrupt
        }
        DiscoveryEndpointAttestationInboxError::Unavailable => {
            DiscoveryEndpointPromotionMaterialError::Unavailable
        }
    }
}

const fn map_revocation_error(
    error: DiscoveryEndpointQuarantineRevocationError,
) -> DiscoveryEndpointPromotionMaterialError {
    match error {
        DiscoveryEndpointQuarantineRevocationError::Rejected => {
            DiscoveryEndpointPromotionMaterialError::Rejected
        }
        DiscoveryEndpointQuarantineRevocationError::Corrupt
        | DiscoveryEndpointQuarantineRevocationError::UnsupportedSchema => {
            DiscoveryEndpointPromotionMaterialError::Corrupt
        }
        DiscoveryEndpointQuarantineRevocationError::Unavailable => {
            DiscoveryEndpointPromotionMaterialError::Unavailable
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{NodeDescriptor, NodePolicy};
    use aeronyx_core::protocol::discovery_endpoint_attestation::{
        DiscoveryEndpointAttestationPurposeV1, DiscoveryEndpointEvidenceAttestationV1,
    };
    use aeronyx_core::protocol::discovery_endpoint_proof::{
        DiscoveryEndpointChallengeV1, DiscoveryEndpointProofV1,
    };
    use rusqlite::Connection;
    use tempfile::TempDir;

    use crate::services::discovery_endpoint_attestation_inbox::{
        DiscoveryEndpointAttestationInboxConfig, DiscoveryEndpointAttestationRecordOutcome,
        VerifiedDiscoveryEndpointAttestationV1,
    };
    use crate::services::discovery_endpoint_eligibility::{
        evaluate_endpoint_candidate, DiscoveryEndpointEligibilityPolicy,
        DiscoveryEndpointStakePolicyMode,
    };
    use crate::services::discovery_endpoint_quarantine::{
        quarantine_admission_commitment, DiscoveryEndpointQuarantineConfig,
        SqliteDiscoveryEndpointQuarantineRegistry,
    };
    use crate::services::discovery_endpoint_quarantine_observation::{
        DiscoveryEndpointObservationDirection, DiscoveryEndpointQuarantineChallengeOutcome,
        DiscoveryEndpointQuarantineEvidenceRequest,
        DiscoveryEndpointQuarantineEvidenceVerificationError,
        DiscoveryEndpointQuarantineEvidenceVerifier, DiscoveryEndpointQuarantineObservationConfig,
        DiscoveryEndpointSatisfiedQuarantineEvidence,
        SqliteDiscoveryEndpointQuarantineObservationRegistry,
    };
    use crate::services::discovery_endpoint_quarantine_revocation::{
        DiscoveryEndpointNegativeEvidenceVerificationError,
        DiscoveryEndpointNegativeEvidenceVerifier, DiscoveryEndpointNegativeObservationRequest,
        DiscoveryEndpointQuarantinePositiveOutcome, DiscoveryEndpointQuarantineRevocationConfig,
    };

    const NOW: u64 = 2_000_000_000;
    const ENDPOINT: &str = "8.8.8.8:51820";

    struct AcceptObservation;

    impl DiscoveryEndpointQuarantineEvidenceVerifier for AcceptObservation {
        fn verify(
            &self,
            _request: &DiscoveryEndpointQuarantineEvidenceRequest,
        ) -> Result<(), DiscoveryEndpointQuarantineEvidenceVerificationError> {
            Ok(())
        }
    }

    struct AcceptNegative;

    impl DiscoveryEndpointNegativeEvidenceVerifier for AcceptNegative {
        fn verify(
            &self,
            _request: &DiscoveryEndpointNegativeObservationRequest,
        ) -> Result<(), DiscoveryEndpointNegativeEvidenceVerificationError> {
            Ok(())
        }
    }

    struct Fixture {
        _directory: TempDir,
        inbox_config: DiscoveryEndpointAttestationInboxConfig,
        revocation_config: DiscoveryEndpointQuarantineRevocationConfig,
        inbox: SqliteDiscoveryEndpointAttestationInbox,
        revocations: SqliteDiscoveryEndpointQuarantineRevocationRegistry,
        descriptor: SignedNodeDescriptor,
        readiness: DiscoveryEndpointPromotionReadiness,
        evidence: DiscoveryEndpointSatisfiedQuarantineEvidence,
    }

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed identity")
    }

    fn tempdir() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("external test root");
        TempDir::new_in("target/test-temp").expect("tempdir")
    }

    fn build_fixture(public_discovery: bool) -> Fixture {
        build_fixture_with_endpoint(public_discovery, ENDPOINT)
    }

    fn build_fixture_with_endpoint(public_discovery: bool, endpoint_text: &str) -> Fixture {
        let directory = tempdir();
        let target = key(0x21);
        let mut descriptor =
            NodeDescriptor::new(target.public_key_bytes(), 7, NOW - 60, NOW + 600, "1.0.0")
                .with_protocol_features([NodeProtocolFeature::AnonymousMailboxV1]);
        descriptor.public_endpoint = Some(endpoint_text.to_string());
        descriptor.capabilities.push(NodeCapability::ChatRelay);
        descriptor.policy = NodePolicy {
            public_discovery,
            ..NodePolicy::default()
        };
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let descriptor_pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .expect("descriptor commitment");
        let socket = canonical_attested_public_endpoint_socket_v1(endpoint_text)
            .expect("canonical endpoint");
        let endpoint_commitment =
            canonical_public_endpoint_commitment(&socket.to_string()).expect("endpoint commitment");
        let inbox_config = DiscoveryEndpointAttestationInboxConfig {
            db_path: directory.path().join("attestation.sqlite3"),
            max_entries: 8,
            max_logical_bytes: 8 * 1024,
            retention_ttl_secs: 180,
            cleanup_batch_size: 8,
        };
        let inbox = SqliteDiscoveryEndpointAttestationInbox::open(inbox_config.clone())
            .expect("attestation inbox");
        for (observer_seed, nonce_seed, context_seed, observed_at) in
            [(0x31, 0x41, 0x51, NOW), (0x32, 0x42, 0x52, NOW + 1)]
        {
            let observer = key(observer_seed);
            let context = [context_seed; 32];
            let challenge = DiscoveryEndpointChallengeV1::issue(
                target.public_key_bytes(),
                descriptor_pin.descriptor_hash,
                endpoint_commitment,
                [nonce_seed; 32],
                context,
                observed_at,
                NOW + 120,
                &observer,
            )
            .expect("challenge");
            let proof =
                DiscoveryEndpointProofV1::respond(&challenge, &context, observed_at, &target)
                    .expect("proof");
            let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &descriptor,
                &challenge,
                &proof,
                context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                observed_at,
                NOW + 120,
                &observer,
            )
            .expect("attestation");
            let frame = attestation.encode();
            let verified =
                VerifiedDiscoveryEndpointAttestationV1::verify(&frame, observed_at, context)
                    .expect("verified attestation");
            assert_eq!(
                inbox
                    .record_verified_at(&verified, observed_at)
                    .expect("retain attestation"),
                DiscoveryEndpointAttestationRecordOutcome::Inserted
            );
        }

        let facts = inbox
            .candidate_facts_at(NOW + 2, 60, 8)
            .expect("candidate facts")
            .into_iter()
            .next()
            .expect("candidate");
        let admission = evaluate_endpoint_candidate(
            &facts,
            DiscoveryEndpointEligibilityPolicy::new(
                2,
                60,
                1,
                DiscoveryEndpointStakePolicyMode::Disabled,
            )
            .expect("eligibility policy"),
            NOW + 2,
            None,
        )
        .into_quarantine_admission()
        .expect("quarantine admission");
        let admission_commitment = quarantine_admission_commitment(&admission);
        let quarantine =
            SqliteDiscoveryEndpointQuarantineRegistry::open(DiscoveryEndpointQuarantineConfig {
                db_path: directory.path().join("quarantine.sqlite3"),
                max_entries: 4,
                cleanup_batch_size: 4,
            })
            .expect("quarantine");
        quarantine
            .record_at(admission, NOW + 2)
            .expect("retain quarantine admission");
        let fresh = quarantine
            .fresh_admission_at(admission_commitment, NOW + 2)
            .expect("fresh admission lookup")
            .expect("fresh admission");
        let observations = SqliteDiscoveryEndpointQuarantineObservationRegistry::open(
            DiscoveryEndpointQuarantineObservationConfig {
                db_path: directory.path().join("observation.sqlite3"),
                max_challenges: 4,
                max_attempts_per_challenge: 4,
                challenge_ttl_secs: 60,
                minimum_observation_span_secs: 5,
                cleanup_batch_size: 4,
            },
        )
        .expect("observation registry");
        let challenge = match observations
            .begin_at(fresh, [0x61; 32], [0x62; 32], NOW + 2, NOW + 2)
            .expect("begin observation")
        {
            DiscoveryEndpointQuarantineChallengeOutcome::Issued(challenge) => challenge,
            other => panic!("unexpected observation outcome {other:?}"),
        };
        observations
            .observe_at(
                challenge,
                [0x63; 32],
                [0x64; 32],
                DiscoveryEndpointObservationDirection::OutboundChallenge,
                [0x65; 32],
                NOW + 3,
                NOW + 3,
                Some(&AcceptObservation),
            )
            .expect("outbound observation");
        observations
            .observe_at(
                challenge,
                [0x66; 32],
                [0x67; 32],
                DiscoveryEndpointObservationDirection::InboundProof,
                [0x68; 32],
                NOW + 8,
                NOW + 8,
                Some(&AcceptObservation),
            )
            .expect("inbound observation");
        let evidence = observations
            .satisfied_evidence_at(challenge, NOW + 8)
            .expect("satisfied lookup")
            .expect("satisfied evidence");
        let revocation_config = DiscoveryEndpointQuarantineRevocationConfig {
            db_path: directory.path().join("revocation.sqlite3"),
            max_states: 4,
            max_negative_per_state: 4,
            negative_ttl_secs: 60,
            cleanup_batch_size: 4,
        };
        let revocations =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(revocation_config.clone())
                .expect("revocation registry");
        assert_eq!(
            revocations
                .retain_positive_at(evidence, NOW + 8)
                .expect("retain positive"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        let readiness = revocations
            .promotion_readiness_at(fresh, evidence, NOW + 8)
            .expect("readiness evaluation")
            .expect("ready");
        Fixture {
            _directory: directory,
            inbox_config,
            revocation_config,
            inbox,
            revocations,
            descriptor,
            readiness,
            evidence,
        }
    }

    fn exact_features() -> DiscoveryEndpointPromotionFeatureRequirement {
        DiscoveryEndpointPromotionFeatureRequirement::new(
            Some(NodeCapability::ChatRelay),
            Some(NodeProtocolFeature::AnonymousMailboxV1),
        )
    }

    #[test]
    fn exact_material_resolves_deterministically_and_survives_restart() {
        let fixture = build_fixture(true);
        let resolver =
            DiscoveryEndpointPromotionMaterialResolver::new(&fixture.inbox, &fixture.revocations);
        let first = resolver
            .resolve_at(
                &fixture.readiness,
                &fixture.descriptor,
                exact_features(),
                NOW + 8,
            )
            .expect("resolve")
            .expect("material");
        let repeated = resolver
            .resolve_at(
                &fixture.readiness,
                &fixture.descriptor,
                exact_features(),
                NOW + 8,
            )
            .expect("repeat")
            .expect("repeated material");
        assert_eq!(first.descriptor(), &fixture.descriptor);
        assert!(first == repeated);
        assert_eq!(first.descriptor_commitment().sequence, 7);
        assert_eq!(
            first.endpoint_commitment(),
            canonical_public_endpoint_commitment(ENDPOINT).expect("endpoint")
        );
        assert_eq!(
            first.group_commitment(),
            fixture.readiness.group_commitment()
        );
        assert_eq!(
            first.readiness_commitment(),
            fixture.readiness.readiness_commitment()
        );
        assert_eq!(first.resolved_at(), NOW + 8);
        assert_eq!(first.valid_until(), fixture.readiness.valid_until());

        let Fixture {
            _directory,
            inbox_config,
            revocation_config,
            inbox,
            revocations,
            descriptor,
            readiness,
            evidence: _,
        } = fixture;
        drop(inbox);
        drop(revocations);
        let inbox =
            SqliteDiscoveryEndpointAttestationInbox::open(inbox_config).expect("restart inbox");
        let revocations =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(revocation_config)
                .expect("restart revocations");
        let restarted = DiscoveryEndpointPromotionMaterialResolver::new(&inbox, &revocations)
            .resolve_at(&readiness, &descriptor, exact_features(), NOW + 9)
            .expect("restart resolution")
            .expect("restart material");
        assert_eq!(restarted.descriptor(), &descriptor);
    }

    #[test]
    fn signed_https_descriptor_resolves_only_its_exact_public_socket() {
        let fixture = build_fixture_with_endpoint(true, "https://8.8.8.8:51820");
        let resolver =
            DiscoveryEndpointPromotionMaterialResolver::new(&fixture.inbox, &fixture.revocations);
        let material = resolver
            .resolve_at(
                &fixture.readiness,
                &fixture.descriptor,
                exact_features(),
                NOW + 8,
            )
            .expect("resolve https descriptor")
            .expect("exact material");
        assert_eq!(
            material.endpoint_commitment(),
            canonical_public_endpoint_commitment("8.8.8.8:51820").expect("socket")
        );
        for endpoint in ["https://9.9.9.9:51820", "https://8.8.8.8:51821"] {
            let mut body = fixture.descriptor.descriptor.clone();
            body.public_endpoint = Some(endpoint.to_string());
            let altered = SignedNodeDescriptor::sign(body, &key(0x21)).expect("signed alternate");
            assert!(resolver
                .resolve_at(&fixture.readiness, &altered, exact_features(), NOW + 8,)
                .expect("alternate rejected")
                .is_none());
        }
    }

    #[test]
    fn exact_descriptor_feature_endpoint_and_presence_mismatches_fail_closed() {
        let fixture = build_fixture(true);
        let resolver =
            DiscoveryEndpointPromotionMaterialResolver::new(&fixture.inbox, &fixture.revocations);
        assert!(resolver
            .resolve_at(
                &fixture.readiness,
                &fixture.descriptor,
                DiscoveryEndpointPromotionFeatureRequirement::new(
                    Some(NodeCapability::PrivacyRelay),
                    Some(NodeProtocolFeature::AnonymousMailboxV1),
                ),
                NOW + 8,
            )
            .expect("capability mismatch")
            .is_none());
        assert!(resolver
            .resolve_at(
                &fixture.readiness,
                &fixture.descriptor,
                DiscoveryEndpointPromotionFeatureRequirement::new(
                    Some(NodeCapability::ChatRelay),
                    Some(NodeProtocolFeature::DirectPeerRelayAuthV2),
                ),
                NOW + 8,
            )
            .expect("feature mismatch")
            .is_none());

        let replacement_key = key(0x21);
        let mut replacement_body = fixture.descriptor.descriptor.clone();
        replacement_body.public_endpoint = Some("8.8.4.4:51820".to_string());
        let replacement =
            SignedNodeDescriptor::sign(replacement_body, &replacement_key).expect("replacement");
        assert!(resolver
            .resolve_at(&fixture.readiness, &replacement, exact_features(), NOW + 8,)
            .expect("descriptor replacement")
            .is_none());

        let empty_directory = tempdir();
        let empty = SqliteDiscoveryEndpointAttestationInbox::open(
            DiscoveryEndpointAttestationInboxConfig {
                db_path: empty_directory.path().join("empty.sqlite3"),
                max_entries: 4,
                max_logical_bytes: 4 * 1024,
                retention_ttl_secs: 60,
                cleanup_batch_size: 4,
            },
        )
        .expect("empty inbox");
        assert!(
            DiscoveryEndpointPromotionMaterialResolver::new(&empty, &fixture.revocations)
                .resolve_at(
                    &fixture.readiness,
                    &fixture.descriptor,
                    exact_features(),
                    NOW + 8,
                )
                .expect("missing material")
                .is_none()
        );

        let hidden = build_fixture(false);
        assert!(DiscoveryEndpointPromotionMaterialResolver::new(
            &hidden.inbox,
            &hidden.revocations
        )
        .resolve_at(
            &hidden.readiness,
            &hidden.descriptor,
            DiscoveryEndpointPromotionFeatureRequirement::new(None, None),
            NOW + 8,
        )
        .expect("public discovery requirement")
        .is_none());
    }

    #[test]
    fn revocation_between_private_read_and_final_check_blocks_material() {
        let fixture = build_fixture(true);
        let resolver =
            DiscoveryEndpointPromotionMaterialResolver::new(&fixture.inbox, &fixture.revocations);
        let material = resolver
            .resolve_with_hook_at(
                &fixture.readiness,
                &fixture.descriptor,
                exact_features(),
                NOW + 8,
                || {
                    fixture
                        .revocations
                        .record_negative_at(
                            fixture.evidence,
                            [0x71; 32],
                            [0x72; 32],
                            NOW + 8,
                            NOW + 8,
                            Some(&AcceptNegative),
                        )
                        .expect("revoke during resolution");
                },
            )
            .expect("race result");
        assert!(material.is_none());
    }

    #[test]
    fn corrupt_frame_expired_readiness_and_routeability_surfaces_fail_closed() {
        let fixture = build_fixture(true);
        let mutation = Connection::open(&fixture.inbox_config.db_path).expect("mutation handle");
        mutation
            .execute(
                "UPDATE discovery_endpoint_attestation_inbox_v1 SET frame=zeroblob(?1)",
                [i64::try_from(
                    aeronyx_core::protocol::DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1,
                )
                .expect("frame size")],
            )
            .expect("corrupt frame");
        let resolver =
            DiscoveryEndpointPromotionMaterialResolver::new(&fixture.inbox, &fixture.revocations);
        assert!(matches!(
            resolver.resolve_at(
                &fixture.readiness,
                &fixture.descriptor,
                exact_features(),
                NOW + 8,
            ),
            Err(DiscoveryEndpointPromotionMaterialError::Corrupt)
        ));
        assert!(resolver
            .resolve_at(
                &fixture.readiness,
                &fixture.descriptor,
                exact_features(),
                fixture.readiness.valid_until().saturating_add(1),
            )
            .expect("expired readiness")
            .is_none());

        let source = include_str!("discovery_endpoint_promotion_material.rs");
        for forbidden in [
            ["Peer", "Store"].concat(),
            [".", "upsert"].concat(),
            ["services::", "routing"].concat(),
            ["server", ".rs"].concat(),
        ] {
            assert!(!source.contains(&forbidden));
        }
        assert!(!source.contains(&["impl fmt::Debug for Verified", "PromotionMaterial"].concat()));
    }
}
