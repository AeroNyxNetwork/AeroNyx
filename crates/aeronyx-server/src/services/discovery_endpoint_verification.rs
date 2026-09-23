// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_verification.rs
// ============================================================================
//! Bounded in-memory orchestration for Stage A discovery endpoint proofs.
//!
//! This service issues exact target-bound challenges and atomically consumes
//! one valid proof. It does not perform network I/O, persist evidence, promote
//! peers, alter discovery state, or expose a public API.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::{
    canonical_public_endpoint_commitment, DiscoveryEndpointChallengeV1, DiscoveryEndpointProofV1,
    DISCOVERY_ENDPOINT_CHALLENGE_MAX_TTL_SECS,
};
use rand::{rngs::OsRng, RngCore};

/// Hard upper bound for pending and consumed challenge records.
pub const DISCOVERY_ENDPOINT_VERIFICATION_MAX_ENTRIES: usize = 65_536;

/// Immutable capacity and lifetime policy for endpoint verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveryEndpointVerificationConfig {
    /// Maximum retained challenge records, including consumed records.
    pub max_entries: usize,
    /// Challenge lifetime in seconds.
    pub challenge_ttl_secs: u64,
}

impl DiscoveryEndpointVerificationConfig {
    /// Validates a bounded service policy.
    ///
    /// # Errors
    /// Returns [`DiscoveryEndpointVerificationError::InvalidConfig`] for zero
    /// or protocol-incompatible limits.
    pub const fn validate(self) -> Result<Self, DiscoveryEndpointVerificationError> {
        if self.max_entries == 0
            || self.max_entries > DISCOVERY_ENDPOINT_VERIFICATION_MAX_ENTRIES
            || self.challenge_ttl_secs == 0
            || self.challenge_ttl_secs > DISCOVERY_ENDPOINT_CHALLENGE_MAX_TTL_SECS
        {
            return Err(DiscoveryEndpointVerificationError::InvalidConfig);
        }
        Ok(self)
    }
}

/// One source-private, exact challenge issuance request.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointChallengeRequestV1 {
    request_id: [u8; 32],
    target_node_id: [u8; 32],
    descriptor_commitment: [u8; 32],
    endpoint_commitment: [u8; 32],
    challenger_context: [u8; 32],
}

impl fmt::Debug for DiscoveryEndpointChallengeRequestV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointChallengeRequestV1")
            .finish_non_exhaustive()
    }
}

impl DiscoveryEndpointChallengeRequestV1 {
    /// Builds one request from an opaque id and canonical public IP endpoint.
    ///
    /// # Errors
    /// Returns [`DiscoveryEndpointVerificationError::Rejected`] for reserved
    /// request material or a non-canonical/non-public endpoint.
    pub fn new(
        request_id: [u8; 32],
        target_node_id: [u8; 32],
        descriptor_commitment: [u8; 32],
        canonical_public_endpoint: &str,
        challenger_context: [u8; 32],
    ) -> Result<Self, DiscoveryEndpointVerificationError> {
        if is_reserved(&request_id) {
            return Err(DiscoveryEndpointVerificationError::Rejected);
        }
        let endpoint_commitment = canonical_public_endpoint_commitment(canonical_public_endpoint)
            .map_err(|_| DiscoveryEndpointVerificationError::Rejected)?;
        Ok(Self {
            request_id,
            target_node_id,
            descriptor_commitment,
            endpoint_commitment,
            challenger_context,
        })
    }

    /// Returns the opaque id used for exact replay and proof consumption.
    #[must_use]
    pub const fn request_id(&self) -> [u8; 32] {
        self.request_id
    }
}

/// Coarse issuance result that redacts challenge bytes from `Debug`.
#[derive(Clone, PartialEq, Eq)]
pub enum DiscoveryEndpointChallengeIssueOutcome {
    /// A new challenge was retained before being returned.
    Issued {
        /// Exact canonical challenge bytes retained by the service.
        challenge_frame: Vec<u8>,
    },
    /// The exact request was replayed and returned its original bytes.
    Existing {
        /// The originally retained canonical challenge bytes.
        challenge_frame: Vec<u8>,
    },
    /// The same request id was already bound to different claims.
    Conflict,
    /// The matching retained request has expired.
    Expired,
    /// No new challenge can be retained under the configured bound.
    AtCapacity,
}

impl fmt::Debug for DiscoveryEndpointChallengeIssueOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Issued { .. } => "DiscoveryEndpointChallengeIssueOutcome::Issued(..)",
            Self::Existing { .. } => "DiscoveryEndpointChallengeIssueOutcome::Existing(..)",
            Self::Conflict => "DiscoveryEndpointChallengeIssueOutcome::Conflict",
            Self::Expired => "DiscoveryEndpointChallengeIssueOutcome::Expired",
            Self::AtCapacity => "DiscoveryEndpointChallengeIssueOutcome::AtCapacity",
        })
    }
}

/// Coarse proof-consumption result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryEndpointProofConsumeOutcome {
    /// This exact proof consumed the challenge for the first time.
    Accepted,
    /// The exact previously accepted proof was replayed.
    Existing,
    /// The request id, challenge, or consumed proof conflicts.
    Conflict,
    /// The retained challenge expired before first acceptance.
    Expired,
    /// No challenge exists for this opaque request id.
    NotFound,
    /// The challenge or proof failed canonical or cryptographic validation.
    Rejected,
}

/// Coarse operational failures without ids, endpoints, keys, or frame bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryEndpointVerificationError {
    /// Service bounds are invalid.
    InvalidConfig,
    /// Caller input cannot be admitted.
    Rejected,
    /// Internal synchronized state is unavailable.
    Unavailable,
}

impl fmt::Display for DiscoveryEndpointVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfig => "endpoint verification config is invalid",
            Self::Rejected => "endpoint verification input was rejected",
            Self::Unavailable => "endpoint verification service is unavailable",
        })
    }
}

impl std::error::Error for DiscoveryEndpointVerificationError {}

#[derive(Clone, PartialEq, Eq)]
struct RetainedChallenge {
    request: DiscoveryEndpointChallengeRequestV1,
    challenge_frame: Vec<u8>,
    challenge_commitment: [u8; 32],
    expires_at: u64,
    consumed_proof: Option<Vec<u8>>,
}

impl fmt::Debug for RetainedChallenge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedChallenge")
            .field("expires_at", &self.expires_at)
            .field("consumed", &self.consumed_proof.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct VerificationState {
    entries: HashMap<[u8; 32], RetainedChallenge>,
}

/// Thread-safe, bounded endpoint challenge issuer and one-time proof consumer.
pub struct DiscoveryEndpointVerificationService {
    challenger: IdentityKeyPair,
    config: DiscoveryEndpointVerificationConfig,
    state: Mutex<VerificationState>,
}

impl fmt::Debug for DiscoveryEndpointVerificationService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointVerificationService")
            .field("max_entries", &self.config.max_entries)
            .field("challenge_ttl_secs", &self.config.challenge_ttl_secs)
            .finish_non_exhaustive()
    }
}

impl DiscoveryEndpointVerificationService {
    /// Constructs an empty verifier using the node's challenger identity.
    ///
    /// # Errors
    /// Returns [`DiscoveryEndpointVerificationError::InvalidConfig`] when the
    /// requested capacity or lifetime exceeds protocol bounds.
    pub fn new(
        challenger: IdentityKeyPair,
        config: DiscoveryEndpointVerificationConfig,
    ) -> Result<Self, DiscoveryEndpointVerificationError> {
        Ok(Self {
            challenger,
            config: config.validate()?,
            state: Mutex::new(VerificationState::default()),
        })
    }

    /// Atomically issues or replays one exact challenge.
    ///
    /// Exact replay/conflict is checked before capacity. New issuance removes
    /// expired unrelated records before applying the configured bound.
    ///
    /// # Errors
    /// Returns a coarse error for invalid claims, timestamp overflow, or an
    /// unavailable synchronized state.
    #[allow(clippy::significant_drop_tightening)]
    pub fn issue_at(
        &self,
        request: DiscoveryEndpointChallengeRequestV1,
        now: u64,
    ) -> Result<DiscoveryEndpointChallengeIssueOutcome, DiscoveryEndpointVerificationError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| DiscoveryEndpointVerificationError::Unavailable)?;

        if let Some(existing) = state.entries.get(&request.request_id) {
            if existing.request != request {
                return Ok(DiscoveryEndpointChallengeIssueOutcome::Conflict);
            }
            if now > existing.expires_at {
                return Ok(DiscoveryEndpointChallengeIssueOutcome::Expired);
            }
            return Ok(DiscoveryEndpointChallengeIssueOutcome::Existing {
                challenge_frame: existing.challenge_frame.clone(),
            });
        }

        state.entries.retain(|_, entry| entry.expires_at >= now);
        if state.entries.len() >= self.config.max_entries {
            return Ok(DiscoveryEndpointChallengeIssueOutcome::AtCapacity);
        }

        let expires_at = now
            .checked_add(self.config.challenge_ttl_secs)
            .ok_or(DiscoveryEndpointVerificationError::Rejected)?;
        let challenge = DiscoveryEndpointChallengeV1::issue(
            request.target_node_id,
            request.descriptor_commitment,
            request.endpoint_commitment,
            fresh_nonce(),
            request.challenger_context,
            now,
            expires_at,
            &self.challenger,
        )
        .map_err(|_| DiscoveryEndpointVerificationError::Rejected)?;
        let challenge_frame = challenge.encode();
        state.entries.insert(
            request.request_id,
            RetainedChallenge {
                request,
                challenge_frame: challenge_frame.clone(),
                challenge_commitment: challenge.commitment(),
                expires_at,
                consumed_proof: None,
            },
        );
        Ok(DiscoveryEndpointChallengeIssueOutcome::Issued { challenge_frame })
    }

    /// Atomically verifies and consumes one exact challenge proof.
    ///
    /// Exact accepted-proof replay is returned before freshness checks and
    /// never creates a second effect. Invalid proofs leave the challenge open
    /// for a later valid response.
    ///
    /// # Errors
    /// Returns [`DiscoveryEndpointVerificationError::Unavailable`] only when
    /// synchronized state cannot be accessed.
    #[allow(clippy::significant_drop_tightening)]
    pub fn verify_and_consume_at(
        &self,
        request_id: [u8; 32],
        challenge_frame: &[u8],
        proof_frame: &[u8],
        now: u64,
    ) -> Result<DiscoveryEndpointProofConsumeOutcome, DiscoveryEndpointVerificationError> {
        if is_reserved(&request_id) {
            return Ok(DiscoveryEndpointProofConsumeOutcome::Rejected);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| DiscoveryEndpointVerificationError::Unavailable)?;
        let Some(entry) = state.entries.get_mut(&request_id) else {
            return Ok(DiscoveryEndpointProofConsumeOutcome::NotFound);
        };
        if entry.challenge_frame != challenge_frame {
            return Ok(DiscoveryEndpointProofConsumeOutcome::Conflict);
        }
        if let Some(existing_proof) = &entry.consumed_proof {
            return Ok(if existing_proof == proof_frame {
                DiscoveryEndpointProofConsumeOutcome::Existing
            } else {
                DiscoveryEndpointProofConsumeOutcome::Conflict
            });
        }
        if now > entry.expires_at {
            return Ok(DiscoveryEndpointProofConsumeOutcome::Expired);
        }
        let Ok(decoded_challenge) = DiscoveryEndpointChallengeV1::decode(challenge_frame) else {
            return Ok(DiscoveryEndpointProofConsumeOutcome::Rejected);
        };
        let Ok(decoded_proof) = DiscoveryEndpointProofV1::decode(proof_frame) else {
            return Ok(DiscoveryEndpointProofConsumeOutcome::Rejected);
        };
        if entry.challenge_commitment != decoded_challenge.commitment() {
            return Ok(DiscoveryEndpointProofConsumeOutcome::Conflict);
        }
        if decoded_proof
            .verify_for_challenge(&decoded_challenge, now, &entry.request.challenger_context)
            .is_err()
        {
            return Ok(DiscoveryEndpointProofConsumeOutcome::Rejected);
        }
        entry.consumed_proof = Some(proof_frame.to_vec());
        Ok(DiscoveryEndpointProofConsumeOutcome::Accepted)
    }

    /// Removes at most `limit` expired challenge records.
    ///
    /// # Errors
    /// Returns [`DiscoveryEndpointVerificationError::Unavailable`] if the
    /// synchronized state is poisoned.
    #[allow(clippy::significant_drop_tightening)]
    pub fn cleanup_expired_at(
        &self,
        now: u64,
        limit: usize,
    ) -> Result<usize, DiscoveryEndpointVerificationError> {
        if limit == 0 {
            return Ok(0);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| DiscoveryEndpointVerificationError::Unavailable)?;
        let expired: Vec<[u8; 32]> = state
            .entries
            .iter()
            .filter_map(|(request_id, entry)| (entry.expires_at < now).then_some(*request_id))
            .take(limit)
            .collect();
        for request_id in &expired {
            state.entries.remove(request_id);
        }
        Ok(expired.len())
    }

    #[cfg(test)]
    fn retained_count(&self) -> usize {
        self.state.lock().map_or(0, |state| state.entries.len())
    }
}

fn fresh_nonce() -> [u8; 32] {
    loop {
        let mut nonce = [0; 32];
        OsRng.fill_bytes(&mut nonce);
        if !is_reserved(&nonce) {
            return nonce;
        }
    }
}

fn is_reserved<const N: usize>(value: &[u8; N]) -> bool {
    value.iter().all(|byte| *byte == 0) || value.iter().all(|byte| *byte == u8::MAX)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_100_000;

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed identity")
    }

    fn service(max_entries: usize, ttl: u64) -> DiscoveryEndpointVerificationService {
        DiscoveryEndpointVerificationService::new(
            key(3),
            DiscoveryEndpointVerificationConfig {
                max_entries,
                challenge_ttl_secs: ttl,
            },
        )
        .expect("valid service")
    }

    fn make_request(
        request_byte: u8,
        target: &IdentityKeyPair,
        context_byte: u8,
    ) -> DiscoveryEndpointChallengeRequestV1 {
        DiscoveryEndpointChallengeRequestV1::new(
            [request_byte; 32],
            target.public_key_bytes(),
            [0x22; 32],
            "8.8.8.8:51820",
            [context_byte; 32],
        )
        .expect("valid request")
    }

    fn issued_frame(outcome: DiscoveryEndpointChallengeIssueOutcome) -> Vec<u8> {
        match outcome {
            DiscoveryEndpointChallengeIssueOutcome::Issued { challenge_frame } => challenge_frame,
            other => panic!("expected issued challenge, got {other:?}"),
        }
    }

    #[test]
    fn issue_verify_consume_and_exact_replay_are_single_effect() {
        let service = service(4, 60);
        let target = key(9);
        let request = make_request(1, &target, 0x44);
        let frame = issued_frame(service.issue_at(request.clone(), NOW).expect("issue"));
        let replay = service.issue_at(request.clone(), NOW + 1).expect("replay");
        assert_eq!(
            replay,
            DiscoveryEndpointChallengeIssueOutcome::Existing {
                challenge_frame: frame.clone()
            }
        );

        let challenge = DiscoveryEndpointChallengeV1::decode(&frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(
            &challenge,
            &request.challenger_context,
            NOW + 2,
            &target,
        )
        .expect("proof")
        .encode();
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &proof, NOW + 2)
                .expect("consume"),
            DiscoveryEndpointProofConsumeOutcome::Accepted
        );
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &proof, NOW + 3)
                .expect("exact replay"),
            DiscoveryEndpointProofConsumeOutcome::Existing
        );
        assert_eq!(service.retained_count(), 1);
    }

    #[test]
    fn changed_issue_or_consumed_proof_conflicts() {
        let service = service(4, 60);
        let target = key(9);
        let request = make_request(2, &target, 0x44);
        let frame = issued_frame(service.issue_at(request.clone(), NOW).expect("issue"));
        let changed = DiscoveryEndpointChallengeRequestV1::new(
            request.request_id,
            target.public_key_bytes(),
            [0x23; 32],
            "8.8.8.8:51820",
            request.challenger_context,
        )
        .expect("changed request");
        assert_eq!(
            service.issue_at(changed, NOW + 1).expect("conflict"),
            DiscoveryEndpointChallengeIssueOutcome::Conflict
        );

        let challenge = DiscoveryEndpointChallengeV1::decode(&frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(
            &challenge,
            &request.challenger_context,
            NOW + 2,
            &target,
        )
        .expect("proof")
        .encode();
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &proof, NOW + 2)
                .expect("consume"),
            DiscoveryEndpointProofConsumeOutcome::Accepted
        );
        let changed_proof = DiscoveryEndpointProofV1::respond(
            &challenge,
            &request.challenger_context,
            NOW + 3,
            &target,
        )
        .expect("second valid proof")
        .encode();
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &changed_proof, NOW + 3)
                .expect("conflicting replay"),
            DiscoveryEndpointProofConsumeOutcome::Conflict
        );
    }

    #[test]
    fn tamper_and_cross_context_do_not_consume() {
        let service = service(4, 60);
        let target = key(9);
        let request = make_request(3, &target, 0x44);
        let frame = issued_frame(service.issue_at(request.clone(), NOW).expect("issue"));
        let challenge = DiscoveryEndpointChallengeV1::decode(&frame).expect("challenge");
        let wrong_context = [0x45; 32];
        assert!(
            DiscoveryEndpointProofV1::respond(&challenge, &wrong_context, NOW + 1, &target)
                .is_err()
        );

        let other_request = make_request(8, &target, 0x45);
        let other_frame = issued_frame(
            service
                .issue_at(other_request.clone(), NOW)
                .expect("other issue"),
        );
        let other_challenge =
            DiscoveryEndpointChallengeV1::decode(&other_frame).expect("other challenge");
        let cross_context_proof = DiscoveryEndpointProofV1::respond(
            &other_challenge,
            &other_request.challenger_context,
            NOW + 1,
            &target,
        )
        .expect("other proof")
        .encode();
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &cross_context_proof, NOW + 1,)
                .expect("cross-context proof"),
            DiscoveryEndpointProofConsumeOutcome::Rejected
        );

        let proof = DiscoveryEndpointProofV1::respond(
            &challenge,
            &request.challenger_context,
            NOW + 1,
            &target,
        )
        .expect("proof")
        .encode();
        let mut tampered = proof.clone();
        tampered[100] ^= 1;
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &tampered, NOW + 1)
                .expect("tamper"),
            DiscoveryEndpointProofConsumeOutcome::Rejected
        );
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &proof, NOW + 2)
                .expect("valid retry"),
            DiscoveryEndpointProofConsumeOutcome::Accepted
        );
    }

    #[test]
    fn capacity_is_fail_closed_and_cleanup_is_bounded() {
        let service = service(2, 10);
        let target = key(9);
        let first = make_request(4, &target, 0x41);
        let second = make_request(5, &target, 0x42);
        let third = make_request(6, &target, 0x43);
        assert!(matches!(
            service.issue_at(first.clone(), NOW).expect("first"),
            DiscoveryEndpointChallengeIssueOutcome::Issued { .. }
        ));
        assert!(matches!(
            service.issue_at(second, NOW).expect("second"),
            DiscoveryEndpointChallengeIssueOutcome::Issued { .. }
        ));
        assert!(matches!(
            service
                .issue_at(first.clone(), NOW + 1)
                .expect("replay before capacity"),
            DiscoveryEndpointChallengeIssueOutcome::Existing { .. }
        ));
        assert_eq!(
            service.issue_at(third.clone(), NOW).expect("capacity"),
            DiscoveryEndpointChallengeIssueOutcome::AtCapacity
        );
        assert_eq!(
            service.issue_at(first, NOW + 11).expect("expired exact"),
            DiscoveryEndpointChallengeIssueOutcome::Expired
        );
        assert_eq!(service.cleanup_expired_at(NOW + 11, 1).expect("cleanup"), 1);
        assert_eq!(service.retained_count(), 1);
        assert_eq!(
            service
                .cleanup_expired_at(NOW + 11, 0)
                .expect("zero cleanup"),
            0
        );
        assert!(matches!(
            service.issue_at(third, NOW + 11).expect("after cleanup"),
            DiscoveryEndpointChallengeIssueOutcome::Issued { .. }
        ));
    }

    #[test]
    fn expired_proof_and_privacy_safe_debug_are_coarse() {
        let service = service(2, 10);
        let target = key(9);
        let request = make_request(7, &target, 0x44);
        let frame = issued_frame(service.issue_at(request.clone(), NOW).expect("issue"));
        let challenge = DiscoveryEndpointChallengeV1::decode(&frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(
            &challenge,
            &request.challenger_context,
            NOW + 1,
            &target,
        )
        .expect("proof")
        .encode();
        assert_eq!(
            service
                .verify_and_consume_at(request.request_id, &frame, &proof, NOW + 11)
                .expect("expired"),
            DiscoveryEndpointProofConsumeOutcome::Expired
        );

        let request_debug = format!("{request:?}");
        let outcome_debug = format!(
            "{:?}",
            DiscoveryEndpointChallengeIssueOutcome::Issued {
                challenge_frame: frame.clone()
            }
        );
        let service_debug = format!("{service:?}");
        for forbidden in [
            hex::encode(request.request_id),
            hex::encode(request.target_node_id),
            hex::encode(request.endpoint_commitment),
            hex::encode(&frame),
            "8.8.8.8".to_string(),
        ] {
            assert!(!request_debug.contains(&forbidden));
            assert!(!outcome_debug.contains(&forbidden));
            assert!(!service_debug.contains(&forbidden));
        }
    }
}
