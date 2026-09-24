// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_eligibility.rs
// ============================================================================
//! Pure quarantine-eligibility policy for verified endpoint attestations.
//!
//! Cryptographic validity remains owned by the attestation inbox. This module
//! can only classify de-identified facts as eligible for further quarantine;
//! it has no peer-store, promotion, ranking, routing, or network authority.
// [PERMISSIONLESS-ENDPOINT-ELIGIBILITY 2026-09-24 by Codex] Keep external
// stake policy separate from ADAT truth and fail closed when it is required.

use std::fmt;

use super::discovery_endpoint_attestation_inbox::DiscoveryEndpointCandidateFacts;

const MAX_MINIMUM_DISTINCT_OBSERVERS: usize = 64;
const MAX_EVIDENCE_AGE_SECS: u64 = 7 * 24 * 60 * 60;

/// Explicit external stake-policy mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointStakePolicyMode {
    /// Endpoint evidence is evaluated without an external economic decision.
    Disabled,
    /// A candidate-bound external decision is mandatory.
    Required,
}

/// Validated policy for one eligibility evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointEligibilityPolicy {
    minimum_distinct_observers: usize,
    maximum_evidence_age_secs: u64,
    policy_version: u64,
    stake_policy: DiscoveryEndpointStakePolicyMode,
}

impl DiscoveryEndpointEligibilityPolicy {
    pub(crate) fn new(
        minimum_distinct_observers: usize,
        maximum_evidence_age_secs: u64,
        policy_version: u64,
        stake_policy: DiscoveryEndpointStakePolicyMode,
    ) -> Result<Self, DiscoveryEndpointEligibilityPolicyError> {
        if !(2..=MAX_MINIMUM_DISTINCT_OBSERVERS).contains(&minimum_distinct_observers)
            || maximum_evidence_age_secs == 0
            || maximum_evidence_age_secs > MAX_EVIDENCE_AGE_SECS
            || policy_version == 0
        {
            return Err(DiscoveryEndpointEligibilityPolicyError::InvalidPolicy);
        }
        Ok(Self {
            minimum_distinct_observers,
            maximum_evidence_age_secs,
            policy_version,
            stake_policy,
        })
    }

    pub(crate) const fn maximum_evidence_age_secs(self) -> u64 {
        self.maximum_evidence_age_secs
    }
}

/// Coarse policy-construction failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DiscoveryEndpointEligibilityPolicyError {
    /// Threshold, age, or version would weaken the frozen policy floor.
    #[error("endpoint eligibility policy invalid")]
    InvalidPolicy,
}

/// Coarse reason that a candidate remains ineligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointIneligibilityReason {
    InsufficientObservers,
    FreshnessOverlapMissing,
    EvidenceStale,
    StakeDecisionMissing,
    StakeDecisionInvalid,
    StakeDecisionUnavailable,
}

/// The only outcomes emitted by this service.
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointEligibilityDecision {
    Ineligible(DiscoveryEndpointIneligibilityReason),
    EligibleForQuarantine(DiscoveryEndpointQuarantineAdmission),
}

/// Unforgeable-by-construction input accepted by the quarantine registry.
// [PERMISSIONLESS-ENDPOINT-QUARANTINE-ADMISSION 2026-09-24 by Codex] Keep
// construction private to the evaluator so storage cannot accept raw facts.
// [PERMISSIONLESS-ENDPOINT-PROMOTION-READINESS 2026-09-24 by Codex] Preserve
// the signed descriptor sequence before any later readiness decision.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineAdmission {
    group_commitment: [u8; 32],
    descriptor_sequence: u64,
    policy_version: u64,
    valid_until: u64,
}

impl DiscoveryEndpointQuarantineAdmission {
    pub(crate) const fn group_commitment(&self) -> [u8; 32] {
        self.group_commitment
    }

    pub(crate) const fn descriptor_sequence(&self) -> u64 {
        self.descriptor_sequence
    }

    pub(crate) const fn policy_version(&self) -> u64 {
        self.policy_version
    }

    pub(crate) const fn valid_until(&self) -> u64 {
        self.valid_until
    }
}

impl fmt::Debug for DiscoveryEndpointQuarantineAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointQuarantineAdmission")
            .field("policy_version", &self.policy_version)
            .field("valid_until", &self.valid_until)
            .finish_non_exhaustive()
    }
}

impl DiscoveryEndpointEligibilityDecision {
    pub(crate) fn into_quarantine_admission(self) -> Option<DiscoveryEndpointQuarantineAdmission> {
        match self {
            Self::EligibleForQuarantine(admission) => Some(admission),
            Self::Ineligible(_) => None,
        }
    }
}

impl fmt::Debug for DiscoveryEndpointEligibilityDecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ineligible(reason) => formatter.debug_tuple("Ineligible").field(reason).finish(),
            Self::EligibleForQuarantine(admission) => formatter
                .debug_struct("EligibleForQuarantine")
                .field("policy_version", &admission.policy_version)
                .field("valid_until", &admission.valid_until)
                .finish_non_exhaustive(),
        }
    }
}

/// Candidate-bound question sent to an injected external stake verifier.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointStakePolicyRequest {
    pub(crate) group_commitment: [u8; 32],
    pub(crate) policy_version: u64,
    pub(crate) evaluated_at: u64,
}

impl fmt::Debug for DiscoveryEndpointStakePolicyRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointStakePolicyRequest")
            .field("policy_version", &self.policy_version)
            .field("evaluated_at", &self.evaluated_at)
            .finish_non_exhaustive()
    }
}

/// Candidate-bound result returned by a separately authenticated verifier.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointStakePolicyDecision {
    pub(crate) group_commitment: [u8; 32],
    pub(crate) policy_version: u64,
    pub(crate) satisfied: bool,
    pub(crate) valid_until: u64,
}

impl fmt::Debug for DiscoveryEndpointStakePolicyDecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointStakePolicyDecision")
            .field("policy_version", &self.policy_version)
            .field("satisfied", &self.satisfied)
            .field("valid_until", &self.valid_until)
            .finish_non_exhaustive()
    }
}

/// Coarse external-verifier failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointStakePolicyVerificationError {
    Invalid,
    Unavailable,
}

/// Replaceable trust boundary for an external attested stake decision.
pub(crate) trait DiscoveryEndpointStakePolicyVerifier: Send + Sync {
    fn verify(
        &self,
        request: DiscoveryEndpointStakePolicyRequest,
    ) -> Result<DiscoveryEndpointStakePolicyDecision, DiscoveryEndpointStakePolicyVerificationError>;
}

/// Evaluates one factual candidate without mutating any repository.
pub(crate) fn evaluate_endpoint_candidate(
    facts: &DiscoveryEndpointCandidateFacts,
    policy: DiscoveryEndpointEligibilityPolicy,
    now: u64,
    stake_verifier: Option<&dyn DiscoveryEndpointStakePolicyVerifier>,
) -> DiscoveryEndpointEligibilityDecision {
    if facts.distinct_observers < policy.minimum_distinct_observers {
        return DiscoveryEndpointEligibilityDecision::Ineligible(
            DiscoveryEndpointIneligibilityReason::InsufficientObservers,
        );
    }
    if facts.overlap_started_at > facts.overlap_expires_at || facts.overlap_expires_at < now {
        return DiscoveryEndpointEligibilityDecision::Ineligible(
            DiscoveryEndpointIneligibilityReason::FreshnessOverlapMissing,
        );
    }
    if facts.newest_observed_at > now
        || now.saturating_sub(facts.newest_observed_at) > policy.maximum_evidence_age_secs
    {
        return DiscoveryEndpointEligibilityDecision::Ineligible(
            DiscoveryEndpointIneligibilityReason::EvidenceStale,
        );
    }

    let mut valid_until = facts.overlap_expires_at.min(facts.newest_expires_at);
    if policy.stake_policy == DiscoveryEndpointStakePolicyMode::Required {
        let Some(verifier) = stake_verifier else {
            return DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::StakeDecisionMissing,
            );
        };
        let request = DiscoveryEndpointStakePolicyRequest {
            group_commitment: facts.group_commitment,
            policy_version: policy.policy_version,
            evaluated_at: now,
        };
        let decision = match verifier.verify(request) {
            Ok(decision) => decision,
            Err(DiscoveryEndpointStakePolicyVerificationError::Invalid) => {
                return DiscoveryEndpointEligibilityDecision::Ineligible(
                    DiscoveryEndpointIneligibilityReason::StakeDecisionInvalid,
                );
            }
            Err(DiscoveryEndpointStakePolicyVerificationError::Unavailable) => {
                return DiscoveryEndpointEligibilityDecision::Ineligible(
                    DiscoveryEndpointIneligibilityReason::StakeDecisionUnavailable,
                );
            }
        };
        if decision.group_commitment != facts.group_commitment
            || decision.policy_version != policy.policy_version
            || !decision.satisfied
            || decision.valid_until < now
        {
            return DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::StakeDecisionInvalid,
            );
        }
        valid_until = valid_until.min(decision.valid_until);
    }

    DiscoveryEndpointEligibilityDecision::EligibleForQuarantine(
        DiscoveryEndpointQuarantineAdmission {
            group_commitment: facts.group_commitment,
            descriptor_sequence: facts.descriptor_sequence,
            policy_version: policy.policy_version,
            valid_until,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 2_000_000_000;

    fn facts(observers: usize) -> DiscoveryEndpointCandidateFacts {
        DiscoveryEndpointCandidateFacts {
            group_commitment: [0x31; 32],
            descriptor_sequence: 7,
            distinct_observers: observers,
            overlap_started_at: NOW - 5,
            overlap_expires_at: NOW + 30,
            newest_observed_at: NOW - 5,
            newest_expires_at: NOW + 40,
        }
    }

    fn policy(
        threshold: usize,
        version: u64,
        stake: DiscoveryEndpointStakePolicyMode,
    ) -> DiscoveryEndpointEligibilityPolicy {
        DiscoveryEndpointEligibilityPolicy::new(threshold, 60, version, stake).expect("policy")
    }

    struct FixedStake(
        Result<DiscoveryEndpointStakePolicyDecision, DiscoveryEndpointStakePolicyVerificationError>,
    );

    impl DiscoveryEndpointStakePolicyVerifier for FixedStake {
        fn verify(
            &self,
            _request: DiscoveryEndpointStakePolicyRequest,
        ) -> Result<
            DiscoveryEndpointStakePolicyDecision,
            DiscoveryEndpointStakePolicyVerificationError,
        > {
            self.0
        }
    }

    #[test]
    fn observer_floor_overlap_age_and_policy_version_are_fail_closed() {
        assert_eq!(
            policy(2, 1, DiscoveryEndpointStakePolicyMode::Disabled).maximum_evidence_age_secs(),
            60
        );
        assert_eq!(
            evaluate_endpoint_candidate(
                &facts(1),
                policy(2, 1, DiscoveryEndpointStakePolicyMode::Disabled),
                NOW,
                None,
            ),
            DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::InsufficientObservers
            )
        );
        assert!(matches!(
            evaluate_endpoint_candidate(
                &facts(2),
                policy(2, 4, DiscoveryEndpointStakePolicyMode::Disabled),
                NOW,
                None,
            ),
            DiscoveryEndpointEligibilityDecision::EligibleForQuarantine(
                DiscoveryEndpointQuarantineAdmission {
                    policy_version: 4,
                    ..
                }
            )
        ));
        assert_eq!(
            evaluate_endpoint_candidate(
                &facts(2),
                policy(3, 4, DiscoveryEndpointStakePolicyMode::Disabled),
                NOW,
                None,
            ),
            DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::InsufficientObservers
            )
        );
        let mut disjoint = facts(2);
        disjoint.overlap_started_at = NOW + 1;
        disjoint.overlap_expires_at = NOW;
        assert!(matches!(
            evaluate_endpoint_candidate(
                &disjoint,
                policy(2, 1, DiscoveryEndpointStakePolicyMode::Disabled),
                NOW,
                None,
            ),
            DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::FreshnessOverlapMissing
            )
        ));
        let mut stale = facts(2);
        stale.newest_observed_at = NOW - 61;
        assert!(matches!(
            evaluate_endpoint_candidate(
                &stale,
                policy(2, 1, DiscoveryEndpointStakePolicyMode::Disabled),
                NOW,
                None,
            ),
            DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::EvidenceStale
            )
        ));
        assert!(DiscoveryEndpointEligibilityPolicy::new(
            1,
            60,
            1,
            DiscoveryEndpointStakePolicyMode::Disabled
        )
        .is_err());
        assert!(DiscoveryEndpointEligibilityPolicy::new(
            2,
            60,
            0,
            DiscoveryEndpointStakePolicyMode::Disabled
        )
        .is_err());
    }

    #[test]
    fn required_stake_is_candidate_bound_and_unavailable_fails_closed() {
        let required = policy(2, 9, DiscoveryEndpointStakePolicyMode::Required);
        assert!(matches!(
            evaluate_endpoint_candidate(&facts(2), required, NOW, None),
            DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::StakeDecisionMissing
            )
        ));
        let unavailable = FixedStake(Err(
            DiscoveryEndpointStakePolicyVerificationError::Unavailable,
        ));
        assert!(matches!(
            evaluate_endpoint_candidate(&facts(2), required, NOW, Some(&unavailable)),
            DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::StakeDecisionUnavailable
            )
        ));
        let invalid_verifier =
            FixedStake(Err(DiscoveryEndpointStakePolicyVerificationError::Invalid));
        assert!(matches!(
            evaluate_endpoint_candidate(&facts(2), required, NOW, Some(&invalid_verifier)),
            DiscoveryEndpointEligibilityDecision::Ineligible(
                DiscoveryEndpointIneligibilityReason::StakeDecisionInvalid
            )
        ));
        for invalid in [
            DiscoveryEndpointStakePolicyDecision {
                group_commitment: [0x32; 32],
                policy_version: 9,
                satisfied: true,
                valid_until: NOW + 20,
            },
            DiscoveryEndpointStakePolicyDecision {
                group_commitment: [0x31; 32],
                policy_version: 8,
                satisfied: true,
                valid_until: NOW + 20,
            },
            DiscoveryEndpointStakePolicyDecision {
                group_commitment: [0x31; 32],
                policy_version: 9,
                satisfied: false,
                valid_until: NOW + 20,
            },
            DiscoveryEndpointStakePolicyDecision {
                group_commitment: [0x31; 32],
                policy_version: 9,
                satisfied: true,
                valid_until: NOW - 1,
            },
        ] {
            assert!(matches!(
                evaluate_endpoint_candidate(
                    &facts(2),
                    required,
                    NOW,
                    Some(&FixedStake(Ok(invalid)))
                ),
                DiscoveryEndpointEligibilityDecision::Ineligible(
                    DiscoveryEndpointIneligibilityReason::StakeDecisionInvalid
                )
            ));
        }
        let valid = FixedStake(Ok(DiscoveryEndpointStakePolicyDecision {
            group_commitment: [0x31; 32],
            policy_version: 9,
            satisfied: true,
            valid_until: NOW + 20,
        }));
        assert_eq!(
            evaluate_endpoint_candidate(&facts(2), required, NOW, Some(&valid)),
            DiscoveryEndpointEligibilityDecision::EligibleForQuarantine(
                DiscoveryEndpointQuarantineAdmission {
                    group_commitment: [0x31; 32],
                    descriptor_sequence: 7,
                    policy_version: 9,
                    valid_until: NOW + 20,
                }
            )
        );
    }

    #[test]
    fn debug_surfaces_redact_group_commitments() {
        let sentinel = hex::encode([0x31; 32]);
        let decision = evaluate_endpoint_candidate(
            &facts(2),
            policy(2, 1, DiscoveryEndpointStakePolicyMode::Disabled),
            NOW,
            None,
        );
        let request = DiscoveryEndpointStakePolicyRequest {
            group_commitment: [0x31; 32],
            policy_version: 1,
            evaluated_at: NOW,
        };
        let stake = DiscoveryEndpointStakePolicyDecision {
            group_commitment: [0x31; 32],
            policy_version: 1,
            satisfied: true,
            valid_until: NOW + 1,
        };
        assert!(!format!("{decision:?}{request:?}{stake:?}").contains(&sentinel));
    }
}
