// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_promotion_coordinator.rs
// ============================================================================
//! Bounded, default-off public-endpoint possession probe for Stage-A candidates.
//!
//! A candidate's signed descriptor or its candidate-initiated ADEA exchange
//! cannot prove that its advertised endpoint accepts inbound traffic. The
//! observer therefore sends an unpredictable signed challenge *to that exact
//! endpoint* and accepts only a target-signed response to the same challenge.
//! This is endpoint control at one instant, not route delivery, Sybil
//! independence, economic eligibility, or permission to skip quarantine.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeCapability, NodeDiscoveryMessage, SignedNodeDescriptor,
};
use aeronyx_core::protocol::discovery_endpoint_attestation::{
    canonical_attested_public_endpoint_socket_v1, discovery_endpoint_evidence_commitment_v1,
    DiscoveryEndpointAttestationPurposeV1, DiscoveryEndpointEvidenceAttestationV1,
};
use aeronyx_core::protocol::discovery_endpoint_proof::{
    canonical_public_endpoint_commitment, DiscoveryEndpointChallengeV1, DiscoveryEndpointProofV1,
    DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1, DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1,
};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::Router;
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use super::discovery_endpoint_attestation_inbox::{
    DiscoveryEndpointAttestationRecordOutcome, SqliteDiscoveryEndpointAttestationInbox,
    VerifiedDiscoveryEndpointAttestationV1,
};
use super::discovery_endpoint_eligibility::{
    evaluate_endpoint_candidate, DiscoveryEndpointEligibilityPolicy,
    DiscoveryEndpointStakePolicyMode,
};
use super::discovery_endpoint_promotion_material::{
    DiscoveryEndpointPromotionFeatureRequirement, DiscoveryEndpointPromotionMaterialResolver,
};
use super::discovery_endpoint_quarantine::{
    quarantine_admission_commitment, DiscoveryEndpointFreshQuarantineAdmission,
    DiscoveryEndpointQuarantineConfig, DiscoveryEndpointQuarantineRecordOutcome,
    SqliteDiscoveryEndpointQuarantineRegistry,
};
use super::discovery_endpoint_quarantine_observation::{
    DiscoveryEndpointObservationDirection, DiscoveryEndpointQuarantineChallenge,
    DiscoveryEndpointQuarantineChallengeOutcome, DiscoveryEndpointQuarantineEvidenceRequest,
    DiscoveryEndpointQuarantineEvidenceVerificationError,
    DiscoveryEndpointQuarantineEvidenceVerifier, DiscoveryEndpointQuarantineObservationConfig,
    DiscoveryEndpointQuarantineObservationOutcome,
    SqliteDiscoveryEndpointQuarantineObservationRegistry,
};
use super::discovery_endpoint_quarantine_revocation::{
    DiscoveryEndpointPromotionProbationOutcome, DiscoveryEndpointQuarantinePositiveOutcome,
    DiscoveryEndpointQuarantineRevocationConfig,
    SqliteDiscoveryEndpointQuarantineRevocationRegistry,
};
use super::peer_store::PeerStore;
use crate::api::public_node_router::public_endpoint_flow_context;
use crate::api::{
    canonical_peer_http_url, peer_endpoint_is_public_ip, privacy_safe_peer_http_client_builder,
};

const RESPOND_PATH: &str = "/api/discovery/endpoint-proof/respond";
const MAX_PARALLEL_RESPONDERS: usize = 32;
const PROBE_TIMEOUT: Duration = Duration::from_secs(4);
const MINIMUM_OBSERVATION_SPAN_SECS: u64 = 5;
const PROMOTION_POLICY_VERSION: u64 = 1;
const MAX_FACT_AGE_SECS: u64 = 120;
const PROMOTION_CAPACITY: usize = 256;
const ATTESTATION_GOSSIP_FANOUT: usize = 2;
const OBSERVATION_CONTEXT_DOMAIN: &[u8] = b"AeroNyx/PermissionlessEndpointObservationContextV1\0";

/// Coarse failure with no node id, endpoint, descriptor, or evidence payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum EndpointPossessionError {
    #[error("endpoint possession rejected")]
    Rejected,
    #[error("endpoint possession unavailable")]
    Unavailable,
    #[error("endpoint possession state corrupt")]
    Corrupt,
}

/// This composition owns no identity graph or central enrollment. All four
/// repositories are private and individually fail closed on unknown schemas.
pub(crate) struct PermissionlessPromotionCoordinator {
    peer_store: Arc<PeerStore>,
    inbox: Arc<SqliteDiscoveryEndpointAttestationInbox>,
    quarantine: SqliteDiscoveryEndpointQuarantineRegistry,
    observations: SqliteDiscoveryEndpointQuarantineObservationRegistry,
    revocations: SqliteDiscoveryEndpointQuarantineRevocationRegistry,
    observer: Arc<IdentityKeyPair>,
    client: reqwest::Client,
    attestation_gossip_cursor: AtomicU64,
}

#[derive(Clone, Copy)]
struct PendingVerifiedObservation {
    challenge: DiscoveryEndpointQuarantineChallenge,
    fresh: DiscoveryEndpointFreshQuarantineAdmission,
}

/// Selects at most two already-verified public peers for one opaque fact.
///
/// [PERMISSIONLESS-ATTESTATION-ROTATION 2026-09-25 by Codex] The caller's
/// cursor advances independently of node ids, and filtering precedes the
/// transport budget. For a stable eligible view, repeated live-process rounds
/// cover every peer without using the candidate's key as a public locator or
/// log field.
fn select_attestation_gossip_targets(
    mut identities: Vec<([u8; 32], String)>,
    observer_id: [u8; 32],
    candidate_id: [u8; 32],
    round: u64,
) -> Vec<reqwest::Url> {
    identities.sort_unstable_by_key(|(node_id, _)| *node_id);
    let mut seen_urls = HashSet::new();
    let mut targets = identities
        .into_iter()
        .filter(|(node_id, endpoint)| {
            *node_id != observer_id
                && *node_id != candidate_id
                && peer_endpoint_is_public_ip(endpoint)
        })
        .filter_map(|(_, endpoint)| {
            canonical_peer_http_url(&endpoint, "/api/discovery/gossip").ok()
        })
        .filter(|url| seen_urls.insert(url.clone()))
        .collect::<Vec<_>>();
    if !targets.is_empty() {
        let start = (round as usize) % targets.len();
        targets.rotate_left(start);
        targets.truncate(ATTESTATION_GOSSIP_FANOUT);
    }
    targets
}

impl PermissionlessPromotionCoordinator {
    /// Opens all durable gates before the public responder or scheduler starts.
    /// The caller must run this synchronous SQLite operation in spawn_blocking.
    pub(crate) fn open(
        prefix: &str,
        peer_store: Arc<PeerStore>,
        inbox: Arc<SqliteDiscoveryEndpointAttestationInbox>,
        observer: Arc<IdentityKeyPair>,
    ) -> Result<Self, EndpointPossessionError> {
        if prefix.trim().is_empty() {
            return Err(EndpointPossessionError::Rejected);
        }
        let db_path = |kind: &str| PathBuf::from(format!("{prefix}.{kind}.sqlite3"));
        let quarantine =
            SqliteDiscoveryEndpointQuarantineRegistry::open(DiscoveryEndpointQuarantineConfig {
                db_path: db_path("quarantine"),
                max_entries: PROMOTION_CAPACITY,
                cleanup_batch_size: 32,
            })
            .map_err(|_| EndpointPossessionError::Unavailable)?;
        let observations = SqliteDiscoveryEndpointQuarantineObservationRegistry::open(
            DiscoveryEndpointQuarantineObservationConfig {
                db_path: db_path("observations"),
                max_challenges: PROMOTION_CAPACITY,
                max_attempts_per_challenge: 4,
                challenge_ttl_secs: 90,
                minimum_observation_span_secs: MINIMUM_OBSERVATION_SPAN_SECS,
                cleanup_batch_size: 32,
            },
        )
        .map_err(|_| EndpointPossessionError::Unavailable)?;
        let revocations = SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(
            DiscoveryEndpointQuarantineRevocationConfig {
                db_path: db_path("revocations"),
                max_states: PROMOTION_CAPACITY,
                max_negative_per_state: 4,
                negative_ttl_secs: 300,
                cleanup_batch_size: 32,
            },
        )
        .map_err(|_| EndpointPossessionError::Unavailable)?;
        // A redirect could substitute a different host after the exact
        // descriptor URL was checked. The URL is also compared after send.
        let client = privacy_safe_peer_http_client_builder()
            .timeout(PROBE_TIMEOUT)
            .build()
            .map_err(|_| EndpointPossessionError::Unavailable)?;
        Ok(Self {
            peer_store,
            inbox,
            quarantine,
            observations,
            revocations,
            observer,
            client,
            // [PERMISSIONLESS-ATTESTATION-ROTATION 2026-09-25 by Codex]
            // A restart may repeat a bounded round, but never grants route
            // authority. Wall-clock seeding varies the initial offset when
            // restart times differ; cursor persistence is not claimed.
            attestation_gossip_cursor: AtomicU64::new(unix_now_secs()),
        })
    }

    /// Runs one candidate through two real endpoint round trips. Other nodes'
    /// signed attestations arrive through the existing gossip inbox; this node
    /// never manufactures a second observer identity to satisfy the threshold.
    pub(crate) async fn advance_one(
        self: &Arc<Self>,
    ) -> Result<Option<SignedNodeDescriptor>, EndpointPossessionError> {
        let now = unix_now_secs();
        let Some(descriptor) = self
            .peer_store
            .permissionless_candidate_batch(now, 1)
            .into_iter()
            .next()
        else {
            return Ok(None);
        };
        let peer_store = Arc::clone(&self.peer_store);
        let node_id = descriptor.node_id();
        if !tokio::task::spawn_blocking(move || {
            peer_store.prepare_permissionless_promotion_capacity_for(&node_id, now)
        })
        .await
        .map_err(|_| EndpointPossessionError::Unavailable)?
        {
            // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] A
            // saturated historic gate budget sends no new network probe.
            return Ok(None);
        }
        let admission_probe =
            probe_exact_public_endpoint(&self.client, &self.observer, &descriptor, now).await?;
        let attestation = self
            .retain_local_attestation(descriptor.clone(), admission_probe.clone())
            .await?;
        let coordinator = Arc::clone(self);
        let probe_for_begin = admission_probe.clone();
        let descriptor_for_begin = descriptor.clone();
        let begin_at = unix_now_secs();
        let Some(pending) = tokio::task::spawn_blocking(move || {
            coordinator.begin_verified_observation_at(
                &descriptor_for_begin,
                &probe_for_begin,
                begin_at,
            )
        })
        .await
        .map_err(|_| EndpointPossessionError::Unavailable)??
        else {
            self.gossip_attestation(&descriptor, attestation).await;
            return Ok(None);
        };
        self.gossip_attestation(&descriptor, attestation).await;
        let first =
            probe_exact_public_endpoint(&self.client, &self.observer, &descriptor, unix_now_secs())
                .await?;
        let coordinator = Arc::clone(self);
        let first_for_record = first.clone();
        tokio::task::spawn_blocking(move || {
            coordinator.retain_verified_observation(
                pending.challenge,
                &first_for_record,
                DiscoveryEndpointObservationDirection::OutboundChallenge,
                first_for_record.observed_at,
            )
        })
        .await
        .map_err(|_| EndpointPossessionError::Unavailable)??;
        tokio::time::sleep(Duration::from_secs(MINIMUM_OBSERVATION_SPAN_SECS)).await;
        let second =
            probe_exact_public_endpoint(&self.client, &self.observer, &descriptor, unix_now_secs())
                .await?;
        let coordinator = Arc::clone(self);
        let promoted_descriptor = descriptor.clone();
        tokio::task::spawn_blocking(move || {
            coordinator.complete_verified_observation(&descriptor, &first, &second, pending)
        })
        .await
        .map_err(|_| EndpointPossessionError::Unavailable)??;
        Ok(Some(promoted_descriptor))
    }

    async fn retain_local_attestation(
        self: &Arc<Self>,
        descriptor: SignedNodeDescriptor,
        proof: VerifiedEndpointPossession,
    ) -> Result<Vec<u8>, EndpointPossessionError> {
        let coordinator = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            coordinator.verify_possession(&descriptor, &proof)?;
            let context = public_endpoint_flow_context(coordinator.observer.public_key_bytes());
            let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &descriptor,
                &proof.challenge,
                &proof.proof,
                context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                proof.observed_at,
                proof.observed_at.saturating_add(MAX_FACT_AGE_SECS),
                &coordinator.observer,
            )
            .map_err(|_| EndpointPossessionError::Rejected)?;
            let frame = attestation.encode();
            let verified =
                VerifiedDiscoveryEndpointAttestationV1::verify(&frame, proof.observed_at, context)
                    .map_err(|_| EndpointPossessionError::Rejected)?;
            match coordinator
                .inbox
                .record_verified_at(&verified, proof.observed_at)
            {
                Ok(
                    DiscoveryEndpointAttestationRecordOutcome::Inserted
                    | DiscoveryEndpointAttestationRecordOutcome::Existing,
                ) => Ok(frame),
                Ok(DiscoveryEndpointAttestationRecordOutcome::Conflict) => coordinator
                    .inbox
                    .exact_frame_for_slot_at(verified.slot_commitment(), context, proof.observed_at)
                    .map_err(|_| EndpointPossessionError::Unavailable)?
                    .ok_or(EndpointPossessionError::Corrupt),
                Ok(DiscoveryEndpointAttestationRecordOutcome::AtCapacity) | Err(_) => {
                    Err(EndpointPossessionError::Unavailable)
                }
            }
        })
        .await
        .map_err(|_| EndpointPossessionError::Unavailable)?
    }

    async fn gossip_attestation(&self, candidate: &SignedNodeDescriptor, frame: Vec<u8>) {
        let message = NodeDiscoveryMessage::EndpointEvidenceAttestationV1 {
            attestation_frame: frame,
        };
        let observer_id = self.observer.public_key_bytes();
        // [PERMISSIONLESS-ATTESTATION-ROTATION 2026-09-25 by Codex] Filter
        // invalid/colliding public transports before consuming the two-peer
        // budget, then rotate the complete verified view each retry. An
        // attestation is only evidence; gossip never promotes a route.
        let round = self
            .attestation_gossip_cursor
            .fetch_add(1, Ordering::Relaxed);
        let targets = select_attestation_gossip_targets(
            self.peer_store
                .valid_public_endpoint_identities(unix_now_secs()),
            observer_id,
            candidate.node_id(),
            round,
        );
        for url in targets {
            let _ = self.client.post(url).json(&message).send().await;
        }
    }

    fn verify_possession(
        &self,
        descriptor: &SignedNodeDescriptor,
        observed: &VerifiedEndpointPossession,
    ) -> Result<(), EndpointPossessionError> {
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|_| EndpointPossessionError::Rejected)?;
        let endpoint = descriptor
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(EndpointPossessionError::Rejected)?;
        let (_, commitment) = exact_probe_target(endpoint)?;
        let challenge = &observed.challenge;
        let context = public_endpoint_flow_context(self.observer.public_key_bytes());
        if challenge.target_node_id() != pin.node_id
            || challenge.descriptor_commitment() != pin.descriptor_hash
            || challenge.endpoint_commitment() != commitment
            || challenge.challenger_node_id() != self.observer.public_key_bytes()
        {
            return Err(EndpointPossessionError::Rejected);
        }
        observed
            .proof
            .verify_for_challenge(challenge, observed.observed_at, &context)
            .map_err(|_| EndpointPossessionError::Rejected)
    }

    /// Starts quarantine observation only after a real local probe and at
    /// least one *other* signed observer are current. No synthetic observer
    /// is created to meet the threshold.
    fn begin_verified_observation_at(
        &self,
        descriptor: &SignedNodeDescriptor,
        admission_probe: &VerifiedEndpointPossession,
        now: u64,
    ) -> Result<Option<PendingVerifiedObservation>, EndpointPossessionError> {
        self.verify_possession(descriptor, admission_probe)?;
        if descriptor.verify_at(now).is_err() {
            return Err(EndpointPossessionError::Rejected);
        }
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|_| EndpointPossessionError::Rejected)?;
        let (_, endpoint_commitment) = exact_probe_target(
            descriptor
                .descriptor
                .public_endpoint
                .as_deref()
                .ok_or(EndpointPossessionError::Rejected)?,
        )?;
        let Some(facts) = self
            .inbox
            .candidate_facts_for_exact_at(
                pin.node_id,
                pin.sequence,
                pin.descriptor_hash,
                endpoint_commitment,
                now,
                MAX_FACT_AGE_SECS,
            )
            .map_err(|_| EndpointPossessionError::Unavailable)?
        else {
            return Ok(None);
        };
        let policy = DiscoveryEndpointEligibilityPolicy::new(
            2,
            MAX_FACT_AGE_SECS,
            PROMOTION_POLICY_VERSION,
            DiscoveryEndpointStakePolicyMode::Disabled,
        )
        .map_err(|_| EndpointPossessionError::Rejected)?;
        let Some(admission) =
            evaluate_endpoint_candidate(&facts, policy, now, None).into_quarantine_admission()
        else {
            return Ok(None);
        };
        match self.quarantine.record_at(admission, now) {
            Ok(
                DiscoveryEndpointQuarantineRecordOutcome::Inserted
                | DiscoveryEndpointQuarantineRecordOutcome::Existing,
            ) => {}
            Ok(_) => return Err(EndpointPossessionError::Rejected),
            Err(_) => return Err(EndpointPossessionError::Unavailable),
        }
        let fresh = self
            .quarantine
            .fresh_admission_at(quarantine_admission_commitment(&admission), now)
            .map_err(|_| EndpointPossessionError::Unavailable)?
            .ok_or(EndpointPossessionError::Rejected)?;
        let mut nonce = [0u8; 32];
        OsRng
            .try_fill_bytes(&mut nonce)
            .map_err(|_| EndpointPossessionError::Unavailable)?;
        let challenge = match self
            .observations
            .begin_at(
                fresh,
                nonce,
                public_endpoint_flow_context(self.observer.public_key_bytes()),
                now,
                now,
            )
            .map_err(|_| EndpointPossessionError::Unavailable)?
        {
            DiscoveryEndpointQuarantineChallengeOutcome::Issued(challenge)
            | DiscoveryEndpointQuarantineChallengeOutcome::Existing(challenge) => challenge,
            _ => return Err(EndpointPossessionError::Rejected),
        };
        Ok(Some(PendingVerifiedObservation { challenge, fresh }))
    }

    /// Finishes only the exact already-started challenge after a second real
    /// endpoint probe separated by the configured minimum wall-clock span.
    fn complete_verified_observation(
        &self,
        descriptor: &SignedNodeDescriptor,
        first: &VerifiedEndpointPossession,
        second: &VerifiedEndpointPossession,
        pending: PendingVerifiedObservation,
    ) -> Result<(), EndpointPossessionError> {
        self.verify_possession(descriptor, first)?;
        self.verify_possession(descriptor, second)?;
        let now = second.observed_at;
        if first.observed_at > now
            || now.saturating_sub(first.observed_at) < MINIMUM_OBSERVATION_SPAN_SECS
            || descriptor.verify_at(now).is_err()
        {
            return Err(EndpointPossessionError::Rejected);
        }
        self.retain_verified_observation(
            pending.challenge,
            second,
            DiscoveryEndpointObservationDirection::InboundProof,
            now,
        )?;
        #[cfg(test)]
        eprintln!("four-node: inbound retained");
        let evidence = self
            .observations
            .satisfied_evidence_at(pending.challenge, now)
            .map_err(|_| EndpointPossessionError::Unavailable)?
            .ok_or(EndpointPossessionError::Rejected)?;
        #[cfg(test)]
        eprintln!("four-node: evidence satisfied");
        match self.revocations.retain_positive_at(evidence, now) {
            Ok(
                DiscoveryEndpointQuarantinePositiveOutcome::Retained
                | DiscoveryEndpointQuarantinePositiveOutcome::Existing,
            ) => {}
            Ok(_) => return Err(EndpointPossessionError::Rejected),
            Err(_) => return Err(EndpointPossessionError::Unavailable),
        }
        #[cfg(test)]
        eprintln!("four-node: positive retained");
        let readiness = self
            .revocations
            .promotion_readiness_at(pending.fresh, evidence, now)
            .map_err(|_| EndpointPossessionError::Unavailable)?
            .ok_or(EndpointPossessionError::Rejected)?;
        #[cfg(test)]
        eprintln!("four-node: readiness");
        let material =
            DiscoveryEndpointPromotionMaterialResolver::new(&self.inbox, &self.revocations)
                .resolve_at(
                    &readiness,
                    descriptor,
                    DiscoveryEndpointPromotionFeatureRequirement::new(
                        Some(NodeCapability::ChatRelay),
                        None,
                    ),
                    now,
                )
                .map_err(|_| EndpointPossessionError::Unavailable)?
                .ok_or(EndpointPossessionError::Rejected)?;
        #[cfg(test)]
        eprintln!("four-node: material");
        match self
            .revocations
            .retain_promotion_probation_at(&material, &readiness, now)
        {
            Ok(
                DiscoveryEndpointPromotionProbationOutcome::Inserted
                | DiscoveryEndpointPromotionProbationOutcome::Existing
                | DiscoveryEndpointPromotionProbationOutcome::Replaced,
            ) => {}
            Ok(_) => return Err(EndpointPossessionError::Rejected),
            Err(_) => return Err(EndpointPossessionError::Unavailable),
        }
        #[cfg(test)]
        eprintln!("four-node: probation retained");
        if !self
            .revocations
            .contains_current_promotion_probation_at(&material, &readiness, now)
            .map_err(|_| EndpointPossessionError::Unavailable)?
        {
            return Err(EndpointPossessionError::Rejected);
        }
        self.peer_store
            .promote_permissionless_candidate(&material, now)
            .map_err(|_| EndpointPossessionError::Rejected)?;
        Ok(())
    }

    fn retain_verified_observation(
        &self,
        challenge: DiscoveryEndpointQuarantineChallenge,
        possession: &VerifiedEndpointPossession,
        direction: DiscoveryEndpointObservationDirection,
        now: u64,
    ) -> Result<(), EndpointPossessionError> {
        let evidence =
            discovery_endpoint_evidence_commitment_v1(&possession.challenge, &possession.proof);
        let attempt_id = evidence;
        let context = observation_context(&attempt_id, direction);
        let verifier = ExactPossessionObservationVerifier {
            evidence,
            attempt_id,
            context,
            direction,
        };
        match self
            .observations
            .observe_at(
                challenge,
                attempt_id,
                context,
                direction,
                evidence,
                possession.observed_at,
                now,
                Some(&verifier),
            )
            .map_err(|_| EndpointPossessionError::Unavailable)?
        {
            DiscoveryEndpointQuarantineObservationOutcome::Recorded(_)
            | DiscoveryEndpointQuarantineObservationOutcome::Existing(_) => Ok(()),
            _ => Err(EndpointPossessionError::Rejected),
        }
    }
}

struct ExactPossessionObservationVerifier {
    evidence: [u8; 32],
    attempt_id: [u8; 32],
    context: [u8; 32],
    direction: DiscoveryEndpointObservationDirection,
}

impl DiscoveryEndpointQuarantineEvidenceVerifier for ExactPossessionObservationVerifier {
    fn verify(
        &self,
        request: &DiscoveryEndpointQuarantineEvidenceRequest,
    ) -> Result<(), DiscoveryEndpointQuarantineEvidenceVerificationError> {
        if request.evidence_commitment == self.evidence
            && request.attempt_id == self.attempt_id
            && request.observer_context == self.context
            && request.direction == self.direction
        {
            Ok(())
        } else {
            Err(DiscoveryEndpointQuarantineEvidenceVerificationError::Invalid)
        }
    }
}

fn observation_context(
    attempt_id: &[u8; 32],
    direction: DiscoveryEndpointObservationDirection,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(OBSERVATION_CONTEXT_DOMAIN);
    hasher.update(attempt_id);
    hasher.update([direction as u8]);
    hasher.finalize().into()
}

/// The descriptor transports an HTTP(S) endpoint, while ADEA commits to a
/// canonical public IP socket. Both representations must resolve to the same
/// literal host and effective port; no DNS, proxy, or redirect is admitted.
// [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] This typed bridge
// closes the URL-vs-socket commitment mismatch without changing core wire.
fn exact_probe_target(endpoint: &str) -> Result<(reqwest::Url, [u8; 32]), EndpointPossessionError> {
    let attested_socket = canonical_attested_public_endpoint_socket_v1(endpoint)
        .map_err(|_| EndpointPossessionError::Rejected)?;
    if !peer_endpoint_is_public_ip(endpoint) {
        return Err(EndpointPossessionError::Rejected);
    }
    let url = canonical_peer_http_url(endpoint, RESPOND_PATH)
        .map_err(|_| EndpointPossessionError::Rejected)?;
    let host = url.host_str().ok_or(EndpointPossessionError::Rejected)?;
    let ip: IpAddr = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .map_err(|_| EndpointPossessionError::Rejected)?;
    let port = url
        .port_or_known_default()
        .ok_or(EndpointPossessionError::Rejected)?;
    let socket = SocketAddr::new(ip, port);
    if socket != attested_socket {
        return Err(EndpointPossessionError::Rejected);
    }
    let commitment = canonical_public_endpoint_commitment(&attested_socket.to_string())
        .map_err(|_| EndpointPossessionError::Rejected)?;
    Ok((url, commitment))
}

/// A verified challenge/proof pair delivered by one exact endpoint request.
/// Its fields are private so callers cannot supply a self-signed descriptor
/// as a replacement for direct network observation.
#[derive(Clone)]
pub(crate) struct VerifiedEndpointPossession {
    challenge: DiscoveryEndpointChallengeV1,
    proof: DiscoveryEndpointProofV1,
    observed_at: u64,
}

#[derive(Clone)]
struct ResponderState {
    identity: Arc<IdentityKeyPair>,
    permits: Arc<Semaphore>,
}

/// Mount only on the public node listener, and only under the explicit
/// permissionless-promotion opt-in. Ordinary local/node/VPN APIs are unchanged.
// [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] The responder is
// separate from candidate-initiated ADEA so endpoint reachability is observed
// by an outbound request rather than inferred from a claimed URL.
pub(crate) fn build_endpoint_possession_responder(identity: Arc<IdentityKeyPair>) -> Router {
    Router::new()
        .route(RESPOND_PATH, post(respond_to_endpoint_challenge))
        .layer(DefaultBodyLimit::max(
            DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1,
        ))
        .with_state(ResponderState {
            identity,
            permits: Arc::new(Semaphore::new(MAX_PARALLEL_RESPONDERS)),
        })
}

async fn respond_to_endpoint_challenge(
    State(state): State<ResponderState>,
    body: Bytes,
) -> Result<Vec<u8>, StatusCode> {
    if body.len() != DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1 {
        return Err(StatusCode::BAD_REQUEST);
    }
    let permit = state
        .permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
    let identity = state.identity;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let challenge =
            DiscoveryEndpointChallengeV1::decode(&body).map_err(|_| StatusCode::BAD_REQUEST)?;
        let now = unix_now_secs();
        let context = public_endpoint_flow_context(challenge.challenger_node_id());
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &context, now, &identity)
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        Ok(proof.encode())
    })
    .await
    .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
}

/// Performs exactly one bounded, no-redirect request to the signed public IP
/// literal in `descriptor`. The response is authenticated before any caller
/// may turn it into an attestation or durable observation.
pub(crate) async fn probe_exact_public_endpoint(
    client: &reqwest::Client,
    observer: &IdentityKeyPair,
    descriptor: &SignedNodeDescriptor,
    now: u64,
) -> Result<VerifiedEndpointPossession, EndpointPossessionError> {
    descriptor
        .verify_at(now)
        .map_err(|_| EndpointPossessionError::Rejected)?;
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or(EndpointPossessionError::Rejected)?;
    let (url, endpoint_commitment) = exact_probe_target(endpoint)?;
    let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
        .map_err(|_| EndpointPossessionError::Rejected)?;
    let mut nonce = [0u8; 32];
    OsRng
        .try_fill_bytes(&mut nonce)
        .map_err(|_| EndpointPossessionError::Unavailable)?;
    let context = public_endpoint_flow_context(observer.public_key_bytes());
    let expires_at = now
        .checked_add(PROBE_TIMEOUT.as_secs() + 10)
        .ok_or(EndpointPossessionError::Rejected)?;
    let challenge = DiscoveryEndpointChallengeV1::issue(
        pin.node_id,
        pin.descriptor_hash,
        endpoint_commitment,
        nonce,
        context,
        now,
        expires_at,
        observer,
    )
    .map_err(|_| EndpointPossessionError::Rejected)?;
    let mut response = client
        .post(url.clone())
        .timeout(PROBE_TIMEOUT)
        .body(challenge.encode())
        .send()
        .await
        .map_err(|_| EndpointPossessionError::Unavailable)?;
    if response.status() != reqwest::StatusCode::OK || response.url() != &url {
        return Err(EndpointPossessionError::Rejected);
    }
    // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] A hostile
    // endpoint must not force an unbounded response allocation before the
    // fixed-width codec gets a chance to reject it.
    let mut bytes = Vec::with_capacity(DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| EndpointPossessionError::Unavailable)?
    {
        let next = bytes
            .len()
            .checked_add(chunk.len())
            .ok_or(EndpointPossessionError::Rejected)?;
        if next > DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1 {
            return Err(EndpointPossessionError::Rejected);
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.len() != DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1 {
        return Err(EndpointPossessionError::Rejected);
    }
    let proof =
        DiscoveryEndpointProofV1::decode(&bytes).map_err(|_| EndpointPossessionError::Rejected)?;
    let observed_at = unix_now_secs();
    proof
        .verify_for_challenge(&challenge, observed_at, &context)
        .map_err(|_| EndpointPossessionError::Rejected)?;
    Ok(VerifiedEndpointPossession {
        challenge,
        proof,
        observed_at,
    })
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::discovery_endpoint_attestation_inbox::DiscoveryEndpointAttestationInboxConfig;
    use super::super::peer_store::PermissionlessNodeAdmissionOutcome;
    use super::*;
    use aeronyx_core::protocol::chat::{BlindRelayEnvelope, BlindRelaySuccessReceipt};
    use aeronyx_core::protocol::discovery::{NodeDescriptor, NodePolicy};
    use tempfile::TempDir;

    const NOW: u64 = 2_000_000_000;
    const ENDPOINT: &str = "https://8.8.8.8:8422";

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed key")
    }

    fn candidate(target: &IdentityKeyPair) -> SignedNodeDescriptor {
        let mut body = NodeDescriptor::new(
            target.public_key_bytes(),
            7,
            NOW - 1,
            NOW + 600,
            "1.0.0+anpf1-brsr1",
        );
        body.public_endpoint = Some(ENDPOINT.to_string());
        body.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
        body.policy = NodePolicy {
            public_discovery: true,
            ..NodePolicy::default()
        };
        SignedNodeDescriptor::sign(body, target).expect("candidate descriptor")
    }

    fn possession(
        descriptor: &SignedNodeDescriptor,
        target: &IdentityKeyPair,
        observer: &IdentityKeyPair,
        observed_at: u64,
        nonce: u8,
    ) -> VerifiedEndpointPossession {
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .expect("descriptor pin");
        let endpoint = exact_probe_target(ENDPOINT).expect("endpoint pin").1;
        let context = public_endpoint_flow_context(observer.public_key_bytes());
        let challenge = DiscoveryEndpointChallengeV1::issue(
            target.public_key_bytes(),
            pin.descriptor_hash,
            endpoint,
            [nonce; 32],
            context,
            observed_at,
            observed_at + 30,
            observer,
        )
        .expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &context, observed_at, target)
            .expect("target proof");
        VerifiedEndpointPossession {
            challenge,
            proof,
            observed_at,
        }
    }

    fn test_root() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("test root");
        TempDir::new_in("target/test-temp").expect("temporary directory")
    }

    #[test]
    fn attestation_gossip_retries_cover_verified_peers_without_widening_fanout() {
        // [PERMISSIONLESS-ATTESTATION-ROTATION 2026-09-25 by Codex] An
        // observer restart may repeat a round, but neither private endpoints,
        // duplicate transports, nor the candidate consume its two-peer cap.
        // No HTTP request or identity-bearing log is needed for this proof.
        let observer = [0x01; 32];
        let candidate = [0x02; 32];
        let mut identities = vec![
            (observer, "https://8.8.8.8:8422".to_string()),
            (candidate, "https://8.8.4.4:8422".to_string()),
            ([0x03; 32], "https://127.0.0.1:8422".to_string()),
            ([0x04; 32], "https://example.com:8422".to_string()),
            ([0x20; 32], "https://8.8.8.10:8422".to_string()),
        ];
        assert!(select_attestation_gossip_targets(
            identities[..4].to_vec(),
            observer,
            candidate,
            0,
        )
        .is_empty());
        for offset in 0..6u8 {
            identities.push((
                [0x10 + offset; 32],
                format!("https://8.8.8.{}:8422", 10 + offset),
            ));
        }
        identities.reverse();
        let mut covered = HashSet::new();
        for round in 0..6 {
            let selected =
                select_attestation_gossip_targets(identities.clone(), observer, candidate, round);
            assert_eq!(selected.len(), ATTESTATION_GOSSIP_FANOUT);
            for url in selected {
                assert!(url.as_str().ends_with("/api/discovery/gossip"));
                assert!(!url.as_str().contains("127.0.0.1"));
                assert!(!url.as_str().contains("example.com"));
                covered.insert(url);
            }
        }
        assert_eq!(covered.len(), 6, "all eligible transports get a turn");
        assert_eq!(
            select_attestation_gossip_targets(identities.clone(), observer, candidate, 0),
            select_attestation_gossip_targets(identities, observer, candidate, 0),
            "a repeated round after restart is bounded and deterministic"
        );
    }

    #[test]
    fn url_to_socket_bridge_binds_the_dialed_public_ip_and_port() {
        let (url, commitment) = exact_probe_target(ENDPOINT).expect("exact endpoint");
        assert_eq!(
            url.as_str(),
            "https://8.8.8.8:8422/api/discovery/endpoint-proof/respond"
        );
        assert_eq!(
            commitment,
            canonical_public_endpoint_commitment("8.8.8.8:8422").expect("socket commitment")
        );
        assert_ne!(
            commitment,
            exact_probe_target("https://8.8.8.8:8423")
                .expect("different port")
                .1
        );
        assert_ne!(
            commitment,
            exact_probe_target("https://9.9.9.9:8422")
                .expect("different host")
                .1
        );
        for invalid in [
            "https://example.com:8422",
            "https://127.0.0.1:8422",
            "https://user:pass@8.8.8.8:8422",
            "https://proxy.example/https://8.8.8.8:8422",
            "ftp://8.8.8.8:8422",
        ] {
            assert!(exact_probe_target(invalid).is_err());
        }
    }

    #[tokio::test]
    async fn possession_http_client_does_not_follow_redirects() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local test listener");
        let addr = listener.local_addr().expect("local address");
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] One raw
        // response makes redirect behavior deterministic without depending
        // on a second HTTP server route or accepting a substituted target.
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("one request");
            let mut request = [0u8; 1024];
            socket.read(&mut request).await.expect("read request");
            socket
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: /alternate\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .expect("redirect response");
        });
        let client = privacy_safe_peer_http_client_builder()
            .timeout(PROBE_TIMEOUT)
            .build()
            .expect("probe client");
        let response = client
            .post(format!("http://{addr}/source"))
            .send()
            .await
            .expect("redirect response");
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        assert_eq!(response.url().path(), "/source");
        server.await.expect("one responder task");
    }

    // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Four distinct
    // identities: T is the candidate, O1/O2 independently attest, and R owns
    // the route store. No socket, alternate target, or network fallback exists.
    #[tokio::test]
    async fn four_node_exact_candidate_needs_two_observers_two_probes_and_fresh_route() {
        let root = test_root();
        let target = key(0x21);
        let observer_one = Arc::new(key(0x31));
        let observer_two = key(0x32);
        let route_consumer = key(0x41);
        let descriptor = candidate(&target);
        let store = Arc::new(PeerStore::new());
        store.enable_untrusted_discovery_candidate_mode();
        assert_eq!(
            store.admit_permissionless_descriptor(descriptor.clone(), NOW),
            PermissionlessNodeAdmissionOutcome::Admitted,
        );
        assert!(store.get_valid(&target.public_key_bytes(), NOW).is_none());
        let inbox = Arc::new(
            SqliteDiscoveryEndpointAttestationInbox::open(
                DiscoveryEndpointAttestationInboxConfig {
                    db_path: root.path().join("inbox.sqlite3"),
                    max_entries: 32,
                    max_logical_bytes: 32 * 1024,
                    retention_ttl_secs: 180,
                    cleanup_batch_size: 8,
                },
            )
            .expect("inbox"),
        );
        let runtime = Arc::new(
            PermissionlessPromotionCoordinator::open(
                root.path().join("promotion").to_str().expect("prefix"),
                Arc::clone(&store),
                Arc::clone(&inbox),
                Arc::clone(&observer_one),
            )
            .expect("coordinator"),
        );
        eprintln!("four-node: opened");
        let first_probe = possession(&descriptor, &target, &observer_one, NOW, 0x51);
        let mut altered_body = descriptor.descriptor.clone();
        altered_body.sequence += 1;
        let altered =
            SignedNodeDescriptor::sign(altered_body, &target).expect("altered signed descriptor");
        assert_eq!(
            runtime.verify_possession(&altered, &first_probe),
            Err(EndpointPossessionError::Rejected)
        );
        let first_frame = runtime
            .retain_local_attestation(descriptor.clone(), first_probe.clone())
            .await
            .expect("first attestation");
        eprintln!("four-node: first attestation");
        assert!(runtime
            .begin_verified_observation_at(&descriptor, &first_probe, NOW + 1,)
            .expect("one observer query")
            .is_none());
        // A second local probe reuses the exact signed frame. It never burns
        // the one-per-observer slot or invents a second observer.
        let retry_probe = possession(&descriptor, &target, &observer_one, NOW + 1, 0x52);
        assert_eq!(
            runtime
                .retain_local_attestation(descriptor.clone(), retry_probe,)
                .await
                .expect("exact slot retry"),
            first_frame
        );
        let other_probe = possession(&descriptor, &target, &observer_two, NOW + 1, 0x61);
        let other_context = public_endpoint_flow_context(observer_two.public_key_bytes());
        let other_attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
            &descriptor,
            &other_probe.challenge,
            &other_probe.proof,
            other_context,
            DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
            NOW + 1,
            NOW + 120,
            &observer_two,
        )
        .expect("other observer attestation");
        let other = VerifiedDiscoveryEndpointAttestationV1::verify(
            &other_attestation.encode(),
            NOW + 1,
            other_context,
        )
        .expect("verified other observer");
        assert_eq!(
            inbox
                .record_verified_at(&other, NOW + 1)
                .expect("retain other"),
            DiscoveryEndpointAttestationRecordOutcome::Inserted
        );
        eprintln!("four-node: second attestation");
        let pending = runtime
            .begin_verified_observation_at(&descriptor, &first_probe, NOW + 2)
            .expect("eligible candidate")
            .expect("durable challenge");
        eprintln!("four-node: challenge");
        let outbound = possession(&descriptor, &target, &observer_one, NOW + 3, 0x71);
        runtime
            .retain_verified_observation(
                pending.challenge,
                &outbound,
                DiscoveryEndpointObservationDirection::OutboundChallenge,
                NOW + 3,
            )
            .expect("first direction");
        eprintln!("four-node: first direction");
        assert!(store
            .get_valid(&target.public_key_bytes(), NOW + 3)
            .is_none());
        let inbound = possession(&descriptor, &target, &observer_one, NOW + 9, 0x72);
        runtime
            .complete_verified_observation(&descriptor, &outbound, &inbound, pending)
            .expect("promotion chain");
        eprintln!("four-node: promoted");
        assert!(store
            .get_valid(&target.public_key_bytes(), NOW + 9)
            .is_none());
        assert!(store.record_route_forward_success_for_descriptor(&descriptor, NOW + 10));
        assert!(store
            .get_valid(&target.public_key_bytes(), NOW + 10)
            .is_none());
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] R's
        // request and T's terminal receipt exercise the exact signed control
        // authority independently of the generic route-health counter.
        let route = BlindRelayEnvelope {
            route_id: [0x81; 16],
            next_hop: target.public_key_bytes(),
            ttl: 1,
            encrypted_blob: vec![0x91; 32],
            timestamp: NOW + 11,
            signature: [0; 64],
        }
        .sign_with(&route_consumer);
        let receipt = BlindRelaySuccessReceipt::terminal(
            &route,
            1,
            Some("terminal_next_hop"),
            None,
            None,
            NOW + 11,
            &target,
        );
        receipt
            .verify_expected(
                &route,
                true,
                false,
                1,
                Some("terminal_next_hop"),
                None,
                None,
                &target.public_key_bytes(),
            )
            .expect("R-to-T signed control receipt");
        assert!(store.record_permissionless_promotion_probe_verified(&descriptor, NOW + 11));
        assert_eq!(
            store.get_valid(&target.public_key_bytes(), NOW + 11),
            Some(descriptor.clone())
        );
        assert!(store
            .get_valid(&target.public_key_bytes(), NOW + 121)
            .is_none());
        // Restart cannot reconstruct authority from the descriptor-only cache.
        let restarted = PeerStore::new();
        restarted.enable_untrusted_discovery_candidate_mode();
        assert!(restarted
            .get_valid(&target.public_key_bytes(), NOW + 10)
            .is_none());
    }
}
