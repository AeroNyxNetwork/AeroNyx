// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/witness_outcome.rs
// ============================================
//! # Observation witness outcome values
//!
//! Owns witness targets, decisions, privacy-safe outcome buckets, and the
//! audited aggregate outcome counters and snapshot.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{DirectoryObservationCheckpointV1, DirectoryReplicaStoreError};

/// Audited mature checkpoint that has not reached its configured corroboration
/// target among the current operator-pinned witnesses.
///
/// The retained witness identities are public node signing keys required only
/// to avoid duplicate outbound requests. They must never be exposed by public
/// status or interpreted as voting weight, consensus membership, or finality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryObservationWitnessTarget {
    /// Canonical observer-signed checkpoint requiring more external evidence.
    pub checkpoint: DirectoryObservationCheckpointV1,
    /// Current pinned witnesses with an audited accepted receipt for this row.
    pub witnessed_by: Vec<[u8; 32]>,
    /// Required number of distinct pinned witness receipts.
    pub minimum_witnesses: usize,
}

/// Result of independently evaluating an external observation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessDecision {
    /// Every exact producer prefix exists locally and the root recomputes.
    Accepted,
    /// At least one exact referenced producer prefix is not retained locally.
    EvidenceUnavailable,
    /// Retained producer evidence conflicts or recomputes a different root.
    EvidenceConflict,
}

/// Stable privacy-safe result bucket for one outbound witness attempt.
///
/// The enum deliberately excludes peer identity, endpoint, request id,
/// signature, checkpoint hash, transport text, and response body data. New
/// variants require a schema migration and additive status-contract review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessOutcome {
    /// A canonical accepted receipt was verified and durably retained.
    Accepted,
    /// The witness does not yet retain every exact referenced producer prefix.
    EvidenceUnavailable,
    /// Locally retained evidence conflicts with the observed checkpoint.
    EvidenceConflict,
    /// The witness is not admitted, reachable, or serving the optional route.
    PeerUnavailable,
    /// The bounded outbound request failed before a verifiable frame arrived.
    TransportFailure,
    /// A received frame failed canonical contract or signature verification.
    VerificationFailure,
    /// A verified accepted receipt could not be durably retained.
    PersistenceFailure,
}

/// Aggregate counters for a bounded set of witness attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessOutcomeCounters {
    /// Canonical accepted receipts durably retained.
    pub accepted: u64,
    /// Witnesses missing at least one exact producer prefix.
    pub evidence_unavailable: u64,
    /// Witnesses whose retained evidence conflicts with the checkpoint.
    pub evidence_conflict: u64,
    /// Witnesses unavailable at admission, endpoint, or capability validation.
    pub peer_unavailable: u64,
    /// Bounded outbound transport failures.
    pub transport_failures: u64,
    /// Canonical contract or signature verification failures.
    pub verification_failures: u64,
    /// Verified receipts rejected by durable persistence.
    pub persistence_failures: u64,
}

impl DirectoryObservationWitnessOutcomeCounters {
    pub(super) fn from_outcomes(outcomes: &[DirectoryObservationWitnessOutcome]) -> Self {
        let mut counters = Self::default();
        for outcome in outcomes {
            counters.record(*outcome);
        }
        counters
    }

    pub(super) fn record(&mut self, outcome: DirectoryObservationWitnessOutcome) {
        let counter = match outcome {
            DirectoryObservationWitnessOutcome::Accepted => &mut self.accepted,
            DirectoryObservationWitnessOutcome::EvidenceUnavailable => {
                &mut self.evidence_unavailable
            }
            DirectoryObservationWitnessOutcome::EvidenceConflict => &mut self.evidence_conflict,
            DirectoryObservationWitnessOutcome::PeerUnavailable => &mut self.peer_unavailable,
            DirectoryObservationWitnessOutcome::TransportFailure => &mut self.transport_failures,
            DirectoryObservationWitnessOutcome::VerificationFailure => {
                &mut self.verification_failures
            }
            DirectoryObservationWitnessOutcome::PersistenceFailure => {
                &mut self.persistence_failures
            }
        };
        *counter = counter.saturating_add(1);
    }

    fn checked_add(self, other: Self) -> Result<Self, DirectoryReplicaStoreError> {
        let add = |left: u64, right: u64| {
            left.checked_add(right).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness outcome counter exhausted".to_string(),
                )
            })
        };
        Ok(Self {
            accepted: add(self.accepted, other.accepted)?,
            evidence_unavailable: add(self.evidence_unavailable, other.evidence_unavailable)?,
            evidence_conflict: add(self.evidence_conflict, other.evidence_conflict)?,
            peer_unavailable: add(self.peer_unavailable, other.peer_unavailable)?,
            transport_failures: add(self.transport_failures, other.transport_failures)?,
            verification_failures: add(self.verification_failures, other.verification_failures)?,
            persistence_failures: add(self.persistence_failures, other.persistence_failures)?,
        })
    }

    pub(super) const fn saturating_add(self, other: Self) -> Self {
        Self {
            accepted: self.accepted.saturating_add(other.accepted),
            evidence_unavailable: self
                .evidence_unavailable
                .saturating_add(other.evidence_unavailable),
            evidence_conflict: self
                .evidence_conflict
                .saturating_add(other.evidence_conflict),
            peer_unavailable: self.peer_unavailable.saturating_add(other.peer_unavailable),
            transport_failures: self
                .transport_failures
                .saturating_add(other.transport_failures),
            verification_failures: self
                .verification_failures
                .saturating_add(other.verification_failures),
            persistence_failures: self
                .persistence_failures
                .saturating_add(other.persistence_failures),
        }
    }

    /// Total attempts represented by these mutually exclusive buckets.
    #[must_use]
    pub const fn attempts(self) -> u64 {
        self.accepted
            .saturating_add(self.evidence_unavailable)
            .saturating_add(self.evidence_conflict)
            .saturating_add(self.peer_unavailable)
            .saturating_add(self.transport_failures)
            .saturating_add(self.verification_failures)
            .saturating_add(self.persistence_failures)
    }

    /// Non-accepted attempts represented by these buckets.
    #[must_use]
    pub const fn failures(self) -> u64 {
        self.attempts().saturating_sub(self.accepted)
    }
}

/// Audited aggregate witness telemetry retained across restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessOutcomeSnapshot {
    /// Completed bounded witness rounds.
    pub rounds: u64,
    /// Cumulative mutually exclusive attempt outcomes.
    pub totals: DirectoryObservationWitnessOutcomeCounters,
    /// Latest local checkpoint sequence evaluated by a witness round.
    pub last_checkpoint_sequence: u64,
    /// Timestamp of the latest completed witness round.
    pub last_round_at: Option<u64>,
    /// Latest round containing at least one accepted receipt.
    pub last_success_at: Option<u64>,
    /// Latest round containing at least one non-accepted attempt.
    pub last_failure_at: Option<u64>,
    /// Mutually exclusive outcomes from only the latest completed round.
    pub last_round: DirectoryObservationWitnessOutcomeCounters,
    /// Process-only failures while persisting this telemetry itself.
    /// Durable snapshots always keep this field at zero.
    pub telemetry_persistence_failures: u64,
}

impl DirectoryObservationWitnessOutcomeSnapshot {
    pub(super) fn next_durable_round(
        self,
        checkpoint_sequence: u64,
        observed_at: u64,
        round: DirectoryObservationWitnessOutcomeCounters,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        if checkpoint_sequence < self.last_checkpoint_sequence
            || self
                .last_round_at
                .is_some_and(|last_round_at| observed_at < last_round_at)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome round regressed".to_string(),
            ));
        }
        Ok(Self {
            rounds: self.rounds.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness outcome round counter exhausted".to_string(),
                )
            })?,
            totals: self.totals.checked_add(round)?,
            last_checkpoint_sequence: checkpoint_sequence,
            last_round_at: Some(observed_at),
            last_success_at: if round.accepted > 0 {
                Some(observed_at)
            } else {
                self.last_success_at
            },
            last_failure_at: if round.failures() > 0 {
                Some(observed_at)
            } else {
                self.last_failure_at
            },
            last_round: round,
            telemetry_persistence_failures: 0,
        })
    }
}
