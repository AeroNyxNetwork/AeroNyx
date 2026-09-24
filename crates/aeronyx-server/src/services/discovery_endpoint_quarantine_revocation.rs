// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_quarantine_revocation.rs
// ============================================================================
//! Durable negative-evidence, revocation, and policy-epoch boundary.
//!
//! Positive evidence is never permanent authority. This module consumes only
//! registry-minted satisfied-evidence capabilities and verifier-approved
//! negative observations. Its outputs remain quarantine policy state and have
//! no identity, descriptor, endpoint, ranking, routing, promotion, server, API,
//! configuration, or network authority.
// [PERMISSIONLESS-ENDPOINT-QUARANTINE-REVOCATION 2026-09-24 by Codex] Keep
// revocation irreversible for one admission and policy epochs monotonic.

use std::fmt;
#[cfg(unix)]
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use sha2::{Digest, Sha256};

use super::chat_relay_backup_certification::verify_sqlite_physical_integrity;
use super::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions,
};
use super::chat_relay_mailbox::{prepare_private_sqlite_target, verify_private_file};
use super::discovery_endpoint_quarantine::DiscoveryEndpointFreshQuarantineAdmission;
use super::discovery_endpoint_quarantine_observation::{
    satisfied_quarantine_evidence_commitment, DiscoveryEndpointSatisfiedQuarantineEvidence,
};

const SCHEMA_VERSION: i64 = 1;
const MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
const MAX_STATES: usize = 65_536;
const MAX_NEGATIVE_PER_STATE: usize = 64;
const MAX_NEGATIVE_TTL_SECS: u64 = 7 * 24 * 60 * 60;
const MAX_CLEANUP_BATCH: usize = 4_096;
const NEGATIVE_OBSERVATION_DOMAIN: &[u8] = b"AeroNyx/EndpointQuarantineNegativeObservationV1\0";
const PROMOTION_READINESS_DOMAIN: &[u8] = b"AeroNyx/EndpointPromotionReadinessV1\0";
const PROMOTION_READINESS_TTL_SECS: u64 = 120;

/// Bounded storage and retention policy.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineRevocationConfig {
    pub(crate) db_path: PathBuf,
    pub(crate) max_states: usize,
    pub(crate) max_negative_per_state: usize,
    pub(crate) negative_ttl_secs: u64,
    pub(crate) cleanup_batch_size: usize,
}

impl fmt::Debug for DiscoveryEndpointQuarantineRevocationConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointQuarantineRevocationConfig")
            .field("max_states", &self.max_states)
            .field("max_negative_per_state", &self.max_negative_per_state)
            .field("negative_ttl_secs", &self.negative_ttl_secs)
            .field("cleanup_batch_size", &self.cleanup_batch_size)
            .finish_non_exhaustive()
    }
}

/// Coarse failures with no candidate, observer, commitment, or path data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DiscoveryEndpointQuarantineRevocationError {
    #[error("endpoint quarantine revocation rejected")]
    Rejected,
    #[error("endpoint quarantine revocation schema unsupported")]
    UnsupportedSchema,
    #[error("endpoint quarantine revocation state corrupt")]
    Corrupt,
    #[error("endpoint quarantine revocation unavailable")]
    Unavailable,
}

/// Current policy state remains isolated from routeability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum DiscoveryEndpointQuarantinePolicyState {
    Positive = 1,
    Revoked = 2,
}

impl DiscoveryEndpointQuarantinePolicyState {
    const fn from_i64(value: i64) -> Result<Self, DiscoveryEndpointQuarantineRevocationError> {
        match value {
            1 => Ok(Self::Positive),
            2 => Ok(Self::Revoked),
            _ => Err(DiscoveryEndpointQuarantineRevocationError::Corrupt),
        }
    }
}

/// Positive admission result; none of these outcomes is routeable authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointQuarantinePositiveOutcome {
    Retained,
    Existing,
    Revoked,
    OldPolicyEpoch,
    Expired,
    AtCapacity,
    Conflict,
}

/// Exact negative observation sent to an injected verifier.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointNegativeObservationRequest {
    pub(crate) evidence_id: [u8; 32],
    pub(crate) positive_commitment: [u8; 32],
    pub(crate) admission_commitment: [u8; 32],
    pub(crate) challenge_id: [u8; 32],
    pub(crate) policy_epoch: u64,
    pub(crate) observer_context: [u8; 32],
    pub(crate) observed_at: u64,
    pub(crate) expires_at: u64,
}

impl fmt::Debug for DiscoveryEndpointNegativeObservationRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointNegativeObservationRequest")
            .field("policy_epoch", &self.policy_epoch)
            .field("observed_at", &self.observed_at)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

/// Coarse external-verifier failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointNegativeEvidenceVerificationError {
    Invalid,
    Unavailable,
}

/// Mandatory authentication boundary for negative evidence.
pub(crate) trait DiscoveryEndpointNegativeEvidenceVerifier: Send + Sync {
    fn verify(
        &self,
        request: &DiscoveryEndpointNegativeObservationRequest,
    ) -> Result<(), DiscoveryEndpointNegativeEvidenceVerificationError>;
}

/// Negative mutation outcome with exact replay and conflict semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointQuarantineNegativeOutcome {
    Revoked,
    Existing,
    Conflict,
    OldPolicyEpoch,
    Expired,
    AtCapacity,
    EvidenceMissing,
    EvidenceRejected,
    EvidenceUnavailable,
}

/// Bounded cleanup counts only; no candidate projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineRevocationCleanup {
    pub(crate) removed_states: usize,
    pub(crate) removed_negative_observations: usize,
}

/// Aggregate-only policy status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineRevocationSnapshot {
    pub(crate) current_policy_epoch: u64,
    pub(crate) retained_states: usize,
    pub(crate) current_positive_states: usize,
    pub(crate) revoked_states: usize,
    pub(crate) retained_negative_observations: usize,
}

/// Short-lived proof that every private quarantine gate agreed in one snapshot.
// [PERMISSIONLESS-ENDPOINT-PROMOTION-READINESS 2026-09-24 by Codex] This token
// is not a route, rank, peer-store mutation, or network capability. Only this
// registry can mint it after re-reading its mutable revocation state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointPromotionReadiness {
    readiness_commitment: [u8; 32],
    admission_commitment: [u8; 32],
    positive_commitment: [u8; 32],
    challenge_id: [u8; 32],
    group_commitment: [u8; 32],
    descriptor_sequence: u64,
    policy_epoch: u64,
    evaluated_at: u64,
    valid_until: u64,
}

impl DiscoveryEndpointPromotionReadiness {
    pub(crate) const fn readiness_commitment(&self) -> [u8; 32] {
        self.readiness_commitment
    }

    pub(crate) const fn admission_commitment(&self) -> [u8; 32] {
        self.admission_commitment
    }

    pub(crate) const fn group_commitment(&self) -> [u8; 32] {
        self.group_commitment
    }

    pub(crate) const fn descriptor_sequence(&self) -> u64 {
        self.descriptor_sequence
    }

    pub(crate) const fn policy_epoch(&self) -> u64 {
        self.policy_epoch
    }

    pub(crate) const fn evaluated_at(&self) -> u64 {
        self.evaluated_at
    }

    pub(crate) const fn valid_until(&self) -> u64 {
        self.valid_until
    }
}

impl fmt::Debug for DiscoveryEndpointPromotionReadiness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointPromotionReadiness")
            .field("descriptor_sequence", &self.descriptor_sequence)
            .field("policy_epoch", &self.policy_epoch)
            .field("evaluated_at", &self.evaluated_at)
            .field("valid_until", &self.valid_until)
            .finish_non_exhaustive()
    }
}

/// Dedicated durable revocation registry.
pub(crate) struct SqliteDiscoveryEndpointQuarantineRevocationRegistry {
    config: DiscoveryEndpointQuarantineRevocationConfig,
    connection: Mutex<Connection>,
    #[cfg(unix)]
    _database_parent: File,
}

impl fmt::Debug for SqliteDiscoveryEndpointQuarantineRevocationRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteDiscoveryEndpointQuarantineRevocationRegistry")
            .field("max_states", &self.config.max_states)
            .field(
                "max_negative_per_state",
                &self.config.max_negative_per_state,
            )
            .finish_non_exhaustive()
    }
}

impl SqliteDiscoveryEndpointQuarantineRevocationRegistry {
    pub(crate) fn open(
        config: DiscoveryEndpointQuarantineRevocationConfig,
    ) -> Result<Self, DiscoveryEndpointQuarantineRevocationError> {
        validate_config(&config)?;
        let target = prepare_private_sqlite_target(&config.db_path)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        verify_private_file(&target.resolved_path, false)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        restrict_private_sqlite_permissions(&target.resolved_path)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        verify_sqlite_physical_integrity(&connection, "endpoint_quarantine_revocation_startup")
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        configure_full_durability(&connection, MINIMUM_SYNCHRONOUS_LEVEL)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        initialize_schema(&mut connection)?;
        startup_audit(&connection, &config)?;
        Ok(Self {
            config,
            connection: Mutex::new(connection),
            #[cfg(unix)]
            _database_parent: target.parent,
        })
    }

    /// Retains one fresh positive capability under the monotonic epoch.
    pub(crate) fn retain_positive_at(
        &self,
        evidence: DiscoveryEndpointSatisfiedQuarantineEvidence,
        now: u64,
    ) -> Result<
        DiscoveryEndpointQuarantinePositiveOutcome,
        DiscoveryEndpointQuarantineRevocationError,
    > {
        validate_satisfied_evidence(&evidence, now)?;
        if evidence.valid_until() < now {
            return Ok(DiscoveryEndpointQuarantinePositiveOutcome::Expired);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        cleanup_tx(&tx, now, self.config.cleanup_batch_size)?;
        let (mut states, mut negatives, mut current_epoch) = load_meta(&tx)?;
        if evidence.policy_epoch() < current_epoch {
            return finish_positive(
                tx,
                DiscoveryEndpointQuarantinePositiveOutcome::OldPolicyEpoch,
            );
        }
        if evidence.policy_epoch() > current_epoch {
            (states, negatives, current_epoch) =
                advance_policy_epoch(&tx, evidence.policy_epoch())?;
        }
        if let Some(existing) = load_state(&tx, &evidence.admission_commitment())? {
            if existing.positive_commitment != evidence.evidence_commitment()
                || existing.challenge_id != evidence.challenge_id()
                || existing.policy_epoch != evidence.policy_epoch()
                || existing.valid_until != evidence.valid_until()
            {
                return finish_positive(tx, DiscoveryEndpointQuarantinePositiveOutcome::Conflict);
            }
            let outcome = if existing.state == DiscoveryEndpointQuarantinePolicyState::Revoked {
                DiscoveryEndpointQuarantinePositiveOutcome::Revoked
            } else {
                DiscoveryEndpointQuarantinePositiveOutcome::Existing
            };
            return finish_positive(tx, outcome);
        }
        if states >= self.config.max_states {
            return finish_positive(tx, DiscoveryEndpointQuarantinePositiveOutcome::AtCapacity);
        }
        tx.execute(
            "INSERT INTO discovery_endpoint_quarantine_policy_state_v1(
               admission_commitment,positive_commitment,challenge_id,policy_epoch,
               valid_until,state,revoked_at
             ) VALUES(?1,?2,?3,?4,?5,1,NULL)",
            params![
                &evidence.admission_commitment()[..],
                &evidence.evidence_commitment()[..],
                &evidence.challenge_id()[..],
                as_i64(evidence.policy_epoch())?,
                as_i64(evidence.valid_until())?,
            ],
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        update_meta(
            &tx,
            states
                .checked_add(1)
                .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            negatives,
            current_epoch,
        )?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        Ok(DiscoveryEndpointQuarantinePositiveOutcome::Retained)
    }

    /// Verifies and records one exact negative observation without holding the
    /// `SQLite` write lock across the injected verifier.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_negative_at(
        &self,
        evidence: DiscoveryEndpointSatisfiedQuarantineEvidence,
        evidence_id: [u8; 32],
        observer_context: [u8; 32],
        observed_at: u64,
        now: u64,
        verifier: Option<&dyn DiscoveryEndpointNegativeEvidenceVerifier>,
    ) -> Result<
        DiscoveryEndpointQuarantineNegativeOutcome,
        DiscoveryEndpointQuarantineRevocationError,
    > {
        validate_satisfied_evidence(&evidence, now)?;
        if now == 0
            || observed_at == 0
            || observed_at > now
            || evidence_id.iter().all(|byte| *byte == 0)
            || observer_context.iter().all(|byte| *byte == 0)
        {
            return Err(DiscoveryEndpointQuarantineRevocationError::Rejected);
        }
        let expires_at = observed_at
            .checked_add(self.config.negative_ttl_secs)
            .ok_or(DiscoveryEndpointQuarantineRevocationError::Rejected)?
            .min(evidence.valid_until());
        if evidence.valid_until() < now || observed_at > evidence.valid_until() || expires_at < now
        {
            return Ok(DiscoveryEndpointQuarantineNegativeOutcome::Expired);
        }
        let request = DiscoveryEndpointNegativeObservationRequest {
            evidence_id,
            positive_commitment: evidence.evidence_commitment(),
            admission_commitment: evidence.admission_commitment(),
            challenge_id: evidence.challenge_id(),
            policy_epoch: evidence.policy_epoch(),
            observer_context,
            observed_at,
            expires_at,
        };
        let exact_commitment = negative_observation_commitment(&request);
        {
            let mut connection = self
                .connection
                .lock()
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
            let tx = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
            if let Some(outcome) =
                preflight_negative(&tx, &request, &exact_commitment, now, &self.config, false)?
            {
                return finish_negative(tx, outcome);
            }
            tx.commit()
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        }
        let Some(verifier) = verifier else {
            return Ok(DiscoveryEndpointQuarantineNegativeOutcome::EvidenceMissing);
        };
        match verifier.verify(&request) {
            Ok(()) => {}
            Err(DiscoveryEndpointNegativeEvidenceVerificationError::Invalid) => {
                return Ok(DiscoveryEndpointQuarantineNegativeOutcome::EvidenceRejected);
            }
            Err(DiscoveryEndpointNegativeEvidenceVerificationError::Unavailable) => {
                return Ok(DiscoveryEndpointQuarantineNegativeOutcome::EvidenceUnavailable);
            }
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        cleanup_tx(&tx, now, self.config.cleanup_batch_size)?;
        if let Some(outcome) =
            preflight_negative(&tx, &request, &exact_commitment, now, &self.config, true)?
        {
            return finish_negative(tx, outcome);
        }
        let (mut states, mut negatives, mut current_epoch) = load_meta(&tx)?;
        if request.policy_epoch > current_epoch {
            (states, negatives, current_epoch) = advance_policy_epoch(&tx, request.policy_epoch)?;
        }
        if let Some(existing) = load_state(&tx, &request.admission_commitment)? {
            if existing.positive_commitment != request.positive_commitment
                || existing.challenge_id != request.challenge_id
                || existing.policy_epoch != request.policy_epoch
            {
                return finish_negative(tx, DiscoveryEndpointQuarantineNegativeOutcome::Conflict);
            }
            if existing.state != DiscoveryEndpointQuarantinePolicyState::Revoked {
                let changed = tx
                    .execute(
                        "UPDATE discovery_endpoint_quarantine_policy_state_v1
                         SET state=2,revoked_at=?1
                         WHERE admission_commitment=?2 AND state=1",
                        params![as_i64(now)?, &request.admission_commitment[..]],
                    )
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
                if changed != 1 {
                    return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
                }
            }
        } else {
            if states >= self.config.max_states {
                return finish_negative(tx, DiscoveryEndpointQuarantineNegativeOutcome::AtCapacity);
            }
            tx.execute(
                "INSERT INTO discovery_endpoint_quarantine_policy_state_v1(
                   admission_commitment,positive_commitment,challenge_id,policy_epoch,
                   valid_until,state,revoked_at
                 ) VALUES(?1,?2,?3,?4,?5,2,?6)",
                params![
                    &request.admission_commitment[..],
                    &request.positive_commitment[..],
                    &request.challenge_id[..],
                    as_i64(request.policy_epoch)?,
                    as_i64(evidence.valid_until())?,
                    as_i64(now)?,
                ],
            )
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
            states = states
                .checked_add(1)
                .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        }
        tx.execute(
            "INSERT INTO discovery_endpoint_quarantine_negative_v1(
               evidence_id,admission_commitment,negative_commitment,observer_context,
               policy_epoch,observed_at,expires_at
             ) VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                &request.evidence_id[..],
                &request.admission_commitment[..],
                &exact_commitment[..],
                &request.observer_context[..],
                as_i64(request.policy_epoch)?,
                as_i64(request.observed_at)?,
                as_i64(request.expires_at)?,
            ],
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        update_meta(
            &tx,
            states,
            negatives
                .checked_add(1)
                .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            current_epoch,
        )?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        Ok(DiscoveryEndpointQuarantineNegativeOutcome::Revoked)
    }

    /// Evaluates immutable F.4/F.5 capabilities against one current F.6
    /// snapshot. `None` is deliberately coarse and cannot disclose which gate
    /// rejected the candidate.
    pub(crate) fn promotion_readiness_at(
        &self,
        admission: DiscoveryEndpointFreshQuarantineAdmission,
        evidence: DiscoveryEndpointSatisfiedQuarantineEvidence,
        now: u64,
    ) -> Result<
        Option<DiscoveryEndpointPromotionReadiness>,
        DiscoveryEndpointQuarantineRevocationError,
    > {
        if now == 0 {
            return Err(DiscoveryEndpointQuarantineRevocationError::Rejected);
        }
        let policy_matches = admission.policy_version() == evidence.policy_epoch();
        if admission.admission_commitment() != evidence.admission_commitment()
            || !policy_matches
            || admission.valid_until() < now
            || evidence.valid_until() < now
            || admission.group_commitment().iter().all(|byte| *byte == 0)
            || validate_satisfied_evidence(&evidence, now).is_err()
        {
            return Ok(None);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let (_, _, current_epoch) = load_meta(&tx)?;
        let Some(state) = load_state(&tx, &admission.admission_commitment())? else {
            tx.commit()
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
            drop(connection);
            return Ok(None);
        };
        if !state_allows_readiness(
            &state,
            &evidence,
            current_epoch,
            negative_count(&tx, &admission.admission_commitment())?,
            now,
        ) {
            tx.commit()
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
            drop(connection);
            return Ok(None);
        }
        let valid_until = admission
            .valid_until()
            .min(evidence.valid_until())
            .min(state.valid_until)
            .min(
                now.checked_add(PROMOTION_READINESS_TTL_SECS)
                    .ok_or(DiscoveryEndpointQuarantineRevocationError::Rejected)?,
            );
        let mut readiness = DiscoveryEndpointPromotionReadiness {
            readiness_commitment: [0; 32],
            admission_commitment: admission.admission_commitment(),
            positive_commitment: evidence.evidence_commitment(),
            challenge_id: evidence.challenge_id(),
            group_commitment: admission.group_commitment(),
            descriptor_sequence: admission.descriptor_sequence(),
            policy_epoch: current_epoch,
            evaluated_at: now,
            valid_until,
        };
        readiness.readiness_commitment = promotion_readiness_commitment(&readiness);
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        drop(connection);
        Ok(Some(readiness))
    }

    /// Revalidates a previously minted token against current mutable policy.
    /// Revocation or an epoch advance therefore invalidates an old token even
    /// before its short wall-clock expiry.
    pub(crate) fn verify_promotion_readiness_at(
        &self,
        readiness: &DiscoveryEndpointPromotionReadiness,
        now: u64,
    ) -> Result<bool, DiscoveryEndpointQuarantineRevocationError> {
        if now == 0
            || readiness.evaluated_at == 0
            || readiness.policy_epoch == 0
            || readiness.admission_commitment.iter().all(|byte| *byte == 0)
            || readiness.positive_commitment.iter().all(|byte| *byte == 0)
            || readiness.challenge_id.iter().all(|byte| *byte == 0)
            || readiness.group_commitment.iter().all(|byte| *byte == 0)
            || readiness.evaluated_at > now
            || readiness.valid_until < readiness.evaluated_at
            || readiness.valid_until < now
            || readiness.valid_until
                > readiness
                    .evaluated_at
                    .saturating_add(PROMOTION_READINESS_TTL_SECS)
            || readiness.readiness_commitment != promotion_readiness_commitment(readiness)
        {
            return Ok(false);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let (_, _, current_epoch) = load_meta(&tx)?;
        let state = load_state(&tx, &readiness.admission_commitment)?;
        let valid = current_epoch == readiness.policy_epoch
            && matches!(
                state,
                Some(stored)
                    if stored.state == DiscoveryEndpointQuarantinePolicyState::Positive
                        && stored.positive_commitment == readiness.positive_commitment
                        && stored.challenge_id == readiness.challenge_id
                        && stored.policy_epoch == readiness.policy_epoch
                        && stored.valid_until >= now
            )
            && negative_count(&tx, &readiness.admission_commitment)? == 0;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        drop(connection);
        Ok(valid)
    }

    pub(crate) fn cleanup_expired_at(
        &self,
        now: u64,
        limit: usize,
    ) -> Result<
        DiscoveryEndpointQuarantineRevocationCleanup,
        DiscoveryEndpointQuarantineRevocationError,
    > {
        if now == 0 || limit == 0 || limit > MAX_CLEANUP_BATCH {
            return Err(DiscoveryEndpointQuarantineRevocationError::Rejected);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let cleanup = cleanup_tx(&tx, now, limit)?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        Ok(cleanup)
    }

    pub(crate) fn snapshot_at(
        &self,
        now: u64,
    ) -> Result<
        DiscoveryEndpointQuarantineRevocationSnapshot,
        DiscoveryEndpointQuarantineRevocationError,
    > {
        if now == 0 {
            return Err(DiscoveryEndpointQuarantineRevocationError::Rejected);
        }
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let (states, negatives, current_epoch) = load_meta_connection(&connection)?;
        let positive: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM discovery_endpoint_quarantine_policy_state_v1
                 WHERE state=1 AND policy_epoch=?1 AND valid_until>=?2",
                params![as_i64(current_epoch)?, as_i64(now)?],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        let revoked: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM discovery_endpoint_quarantine_policy_state_v1 WHERE state=2",
                [],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        Ok(DiscoveryEndpointQuarantineRevocationSnapshot {
            current_policy_epoch: current_epoch,
            retained_states: states,
            current_positive_states: usize::try_from(positive)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            revoked_states: usize::try_from(revoked)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            retained_negative_observations: negatives,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StoredPolicyState {
    positive_commitment: [u8; 32],
    challenge_id: [u8; 32],
    policy_epoch: u64,
    valid_until: u64,
    state: DiscoveryEndpointQuarantinePolicyState,
}

fn validate_satisfied_evidence(
    evidence: &DiscoveryEndpointSatisfiedQuarantineEvidence,
    now: u64,
) -> Result<(), DiscoveryEndpointQuarantineRevocationError> {
    if now == 0
        || evidence.policy_epoch() == 0
        || evidence.evidence_commitment().iter().all(|byte| *byte == 0)
        || evidence
            .admission_commitment()
            .iter()
            .all(|byte| *byte == 0)
        || evidence.challenge_id().iter().all(|byte| *byte == 0)
        || evidence.evidence_commitment()
            != satisfied_quarantine_evidence_commitment(
                &evidence.admission_commitment(),
                &evidence.challenge_id(),
                evidence.policy_epoch(),
                evidence.valid_until(),
            )
    {
        return Err(DiscoveryEndpointQuarantineRevocationError::Rejected);
    }
    Ok(())
}

fn state_allows_readiness(
    state: &StoredPolicyState,
    evidence: &DiscoveryEndpointSatisfiedQuarantineEvidence,
    current_epoch: u64,
    negative_observations: usize,
    now: u64,
) -> bool {
    state.state == DiscoveryEndpointQuarantinePolicyState::Positive
        && negative_observations == 0
        && current_epoch == evidence.policy_epoch()
        && state.policy_epoch == current_epoch
        && state.positive_commitment == evidence.evidence_commitment()
        && state.challenge_id == evidence.challenge_id()
        && state.valid_until == evidence.valid_until()
        && state.valid_until >= now
}

fn promotion_readiness_commitment(readiness: &DiscoveryEndpointPromotionReadiness) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(PROMOTION_READINESS_DOMAIN);
    hash.update(readiness.admission_commitment);
    hash.update(readiness.positive_commitment);
    hash.update(readiness.challenge_id);
    hash.update(readiness.group_commitment);
    hash.update(readiness.descriptor_sequence.to_be_bytes());
    hash.update(readiness.policy_epoch.to_be_bytes());
    hash.update(readiness.evaluated_at.to_be_bytes());
    hash.update(readiness.valid_until.to_be_bytes());
    hash.finalize().into()
}

fn preflight_negative(
    tx: &Transaction<'_>,
    request: &DiscoveryEndpointNegativeObservationRequest,
    exact_commitment: &[u8; 32],
    now: u64,
    config: &DiscoveryEndpointQuarantineRevocationConfig,
    enforce_capacity: bool,
) -> Result<
    Option<DiscoveryEndpointQuarantineNegativeOutcome>,
    DiscoveryEndpointQuarantineRevocationError,
> {
    if let Some(existing) = load_negative_commitment(tx, &request.evidence_id)? {
        let outcome = if existing == *exact_commitment {
            DiscoveryEndpointQuarantineNegativeOutcome::Existing
        } else {
            DiscoveryEndpointQuarantineNegativeOutcome::Conflict
        };
        return Ok(Some(outcome));
    }
    let (_, _, current_epoch) = load_meta(tx)?;
    if request.policy_epoch < current_epoch {
        return Ok(Some(
            DiscoveryEndpointQuarantineNegativeOutcome::OldPolicyEpoch,
        ));
    }
    if request.expires_at < now {
        return Ok(Some(DiscoveryEndpointQuarantineNegativeOutcome::Expired));
    }
    if enforce_capacity
        && negative_count(tx, &request.admission_commitment)? >= config.max_negative_per_state
    {
        return Ok(Some(DiscoveryEndpointQuarantineNegativeOutcome::AtCapacity));
    }
    Ok(None)
}

fn validate_config(
    config: &DiscoveryEndpointQuarantineRevocationConfig,
) -> Result<(), DiscoveryEndpointQuarantineRevocationError> {
    if config.db_path.as_os_str().is_empty()
        || config.db_path == Path::new(":memory:")
        || config.max_states == 0
        || config.max_states > MAX_STATES
        || config.max_negative_per_state == 0
        || config.max_negative_per_state > MAX_NEGATIVE_PER_STATE
        || config.negative_ttl_secs == 0
        || config.negative_ttl_secs > MAX_NEGATIVE_TTL_SECS
        || config.cleanup_batch_size == 0
        || config.cleanup_batch_size > MAX_CLEANUP_BATCH
    {
        return Err(DiscoveryEndpointQuarantineRevocationError::Rejected);
    }
    config
        .max_states
        .checked_mul(config.max_negative_per_state)
        .ok_or(DiscoveryEndpointQuarantineRevocationError::Rejected)?;
    Ok(())
}

fn initialize_schema(
    connection: &mut Connection,
) -> Result<(), DiscoveryEndpointQuarantineRevocationError> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let version: i64 = tx
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    if version == 0 {
        let foreign: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
        if foreign != 0 {
            return Err(DiscoveryEndpointQuarantineRevocationError::UnsupportedSchema);
        }
        tx.execute_batch(
            "CREATE TABLE discovery_endpoint_quarantine_revocation_meta_v1(
               singleton INTEGER PRIMARY KEY CHECK(singleton=1),states INTEGER NOT NULL,
               negatives INTEGER NOT NULL,current_policy_epoch INTEGER NOT NULL
             );
             INSERT INTO discovery_endpoint_quarantine_revocation_meta_v1 VALUES(1,0,0,0);
             CREATE TABLE discovery_endpoint_quarantine_policy_state_v1(
               admission_commitment BLOB PRIMARY KEY CHECK(length(admission_commitment)=32),
               positive_commitment BLOB NOT NULL UNIQUE CHECK(length(positive_commitment)=32),
               challenge_id BLOB NOT NULL CHECK(length(challenge_id)=32),
               policy_epoch INTEGER NOT NULL,valid_until INTEGER NOT NULL,
               state INTEGER NOT NULL CHECK(state IN (1,2)),revoked_at INTEGER
             );
             CREATE INDEX discovery_endpoint_quarantine_policy_expiry_v1
               ON discovery_endpoint_quarantine_policy_state_v1(valid_until,admission_commitment);
             CREATE TABLE discovery_endpoint_quarantine_negative_v1(
               evidence_id BLOB PRIMARY KEY CHECK(length(evidence_id)=32),
               admission_commitment BLOB NOT NULL CHECK(length(admission_commitment)=32),
               negative_commitment BLOB NOT NULL UNIQUE CHECK(length(negative_commitment)=32),
               observer_context BLOB NOT NULL CHECK(length(observer_context)=32),
               policy_epoch INTEGER NOT NULL,observed_at INTEGER NOT NULL,expires_at INTEGER NOT NULL,
               FOREIGN KEY(admission_commitment) REFERENCES discovery_endpoint_quarantine_policy_state_v1(admission_commitment) ON DELETE CASCADE
             );
             CREATE INDEX discovery_endpoint_quarantine_negative_expiry_v1
               ON discovery_endpoint_quarantine_negative_v1(expires_at,evidence_id);
             PRAGMA user_version=1;",
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    } else if version != SCHEMA_VERSION {
        return Err(DiscoveryEndpointQuarantineRevocationError::UnsupportedSchema);
    }
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)
}

fn startup_audit(
    connection: &Connection,
    config: &DiscoveryEndpointQuarantineRevocationConfig,
) -> Result<(), DiscoveryEndpointQuarantineRevocationError> {
    let (states, negatives, current_epoch) = load_meta_connection(connection)?;
    let actual_states = count_connection(
        connection,
        "SELECT COUNT(*) FROM discovery_endpoint_quarantine_policy_state_v1",
    )?;
    let actual_negatives = count_connection(
        connection,
        "SELECT COUNT(*) FROM discovery_endpoint_quarantine_negative_v1",
    )?;
    let maximum_negatives = config
        .max_states
        .checked_mul(config.max_negative_per_state)
        .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    if states != actual_states
        || negatives != actual_negatives
        || states > config.max_states
        || negatives > maximum_negatives
    {
        return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
    }
    let maximum_epoch: i64 = connection
        .query_row(
            "SELECT COALESCE(MAX(policy_epoch),0) FROM discovery_endpoint_quarantine_policy_state_v1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    if as_u64(maximum_epoch)? > current_epoch {
        return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
    }
    let mut statement = connection
        .prepare(
            "SELECT admission_commitment,positive_commitment,challenge_id,policy_epoch,
                    valid_until,state,revoked_at
             FROM discovery_endpoint_quarantine_policy_state_v1",
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let mut rows = statement
        .query([])
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?
    {
        let admission = array32(
            row.get(0)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let positive = array32(
            row.get(1)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let challenge = array32(
            row.get(2)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let epoch = as_u64(
            row.get(3)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let valid_until = as_u64(
            row.get(4)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let state = DiscoveryEndpointQuarantinePolicyState::from_i64(
            row.get(5)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let revoked_at: Option<i64> = row
            .get(6)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        if admission.iter().all(|byte| *byte == 0)
            || positive.iter().all(|byte| *byte == 0)
            || challenge.iter().all(|byte| *byte == 0)
            || epoch == 0
            || positive
                != satisfied_quarantine_evidence_commitment(
                    &admission,
                    &challenge,
                    epoch,
                    valid_until,
                )
            || (state == DiscoveryEndpointQuarantinePolicyState::Positive && revoked_at.is_some())
            || (state == DiscoveryEndpointQuarantinePolicyState::Revoked
                && revoked_at.map(as_u64).transpose()?.is_none())
        {
            return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
        }
    }
    let mut statement = connection
        .prepare(
            "SELECT evidence_id,n.admission_commitment,negative_commitment,observer_context,
                    n.policy_epoch,observed_at,expires_at,s.positive_commitment,s.challenge_id,
                    s.policy_epoch,s.valid_until
             FROM discovery_endpoint_quarantine_negative_v1 n
             JOIN discovery_endpoint_quarantine_policy_state_v1 s USING(admission_commitment)",
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let mut rows = statement
        .query([])
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?
    {
        let request = DiscoveryEndpointNegativeObservationRequest {
            evidence_id: array32(
                row.get(0)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            admission_commitment: array32(
                row.get(1)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            observer_context: array32(
                row.get(3)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            policy_epoch: as_u64(
                row.get(4)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            observed_at: as_u64(
                row.get(5)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            expires_at: as_u64(
                row.get(6)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            positive_commitment: array32(
                row.get(7)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            challenge_id: array32(
                row.get(8)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
        };
        let stored = array32(
            row.get(2)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let state_epoch = as_u64(
            row.get(9)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let state_valid_until = as_u64(
            row.get(10)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        if request.evidence_id.iter().all(|byte| *byte == 0)
            || request.observer_context.iter().all(|byte| *byte == 0)
            || request.policy_epoch == 0
            || request.policy_epoch != state_epoch
            || request.expires_at < request.observed_at
            || request.expires_at > state_valid_until
            || stored != negative_observation_commitment(&request)
        {
            return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
        }
    }
    Ok(())
}

fn load_state(
    tx: &Transaction<'_>,
    admission: &[u8; 32],
) -> Result<Option<StoredPolicyState>, DiscoveryEndpointQuarantineRevocationError> {
    let raw = tx
        .query_row(
            "SELECT positive_commitment,challenge_id,policy_epoch,valid_until,state
             FROM discovery_endpoint_quarantine_policy_state_v1 WHERE admission_commitment=?1",
            params![&admission[..]],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )
        .optional()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    raw.map(|(positive, challenge, epoch, valid_until, state)| {
        Ok(StoredPolicyState {
            positive_commitment: array32(positive)?,
            challenge_id: array32(challenge)?,
            policy_epoch: as_u64(epoch)?,
            valid_until: as_u64(valid_until)?,
            state: DiscoveryEndpointQuarantinePolicyState::from_i64(state)?,
        })
    })
    .transpose()
}

fn load_negative_commitment(
    tx: &Transaction<'_>,
    evidence_id: &[u8; 32],
) -> Result<Option<[u8; 32]>, DiscoveryEndpointQuarantineRevocationError> {
    tx.query_row(
        "SELECT negative_commitment FROM discovery_endpoint_quarantine_negative_v1 WHERE evidence_id=?1",
        params![&evidence_id[..]],
        |row| row.get::<_, Vec<u8>>(0),
    )
    .optional()
    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?
    .map(array32)
    .transpose()
}

fn negative_count(
    tx: &Transaction<'_>,
    admission: &[u8; 32],
) -> Result<usize, DiscoveryEndpointQuarantineRevocationError> {
    let count: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_quarantine_negative_v1 WHERE admission_commitment=?1",
            params![&admission[..]],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    usize::try_from(count).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)
}

fn advance_policy_epoch(
    tx: &Transaction<'_>,
    new_epoch: u64,
) -> Result<(usize, usize, u64), DiscoveryEndpointQuarantineRevocationError> {
    let (states, negatives, current_epoch) = load_meta(tx)?;
    if new_epoch <= current_epoch {
        return Ok((states, negatives, current_epoch));
    }
    let cascaded_negatives: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_quarantine_negative_v1 n
             JOIN discovery_endpoint_quarantine_policy_state_v1 s USING(admission_commitment)
             WHERE s.policy_epoch<?1",
            params![as_i64(new_epoch)?],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let removed_states = tx
        .execute(
            "DELETE FROM discovery_endpoint_quarantine_policy_state_v1 WHERE policy_epoch<?1",
            params![as_i64(new_epoch)?],
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let cascaded_negatives = usize::try_from(cascaded_negatives)
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    let retained_states = states
        .checked_sub(removed_states)
        .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    let retained_negatives = negatives
        .checked_sub(cascaded_negatives)
        .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    update_meta(tx, retained_states, retained_negatives, new_epoch)?;
    Ok((retained_states, retained_negatives, new_epoch))
}

fn cleanup_tx(
    tx: &Transaction<'_>,
    now: u64,
    limit: usize,
) -> Result<DiscoveryEndpointQuarantineRevocationCleanup, DiscoveryEndpointQuarantineRevocationError>
{
    let limit =
        i64::try_from(limit).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Rejected)?;
    let removed_negative_observations = tx
        .execute(
            "DELETE FROM discovery_endpoint_quarantine_negative_v1
             WHERE evidence_id IN (
               SELECT evidence_id FROM discovery_endpoint_quarantine_negative_v1
               WHERE expires_at<?1 ORDER BY expires_at,evidence_id LIMIT ?2
             )",
            params![as_i64(now)?, limit],
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let cascaded_negatives: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_quarantine_negative_v1
             WHERE admission_commitment IN (
               SELECT admission_commitment FROM discovery_endpoint_quarantine_policy_state_v1
               WHERE valid_until<?1 ORDER BY valid_until,admission_commitment LIMIT ?2
             )",
            params![as_i64(now)?, limit],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let removed_states = tx
        .execute(
            "DELETE FROM discovery_endpoint_quarantine_policy_state_v1
             WHERE admission_commitment IN (
               SELECT admission_commitment FROM discovery_endpoint_quarantine_policy_state_v1
               WHERE valid_until<?1 ORDER BY valid_until,admission_commitment LIMIT ?2
             )",
            params![as_i64(now)?, limit],
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let cascaded_negatives = usize::try_from(cascaded_negatives)
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    let removed_negatives = removed_negative_observations
        .checked_add(cascaded_negatives)
        .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    if removed_states > 0 || removed_negatives > 0 {
        let (states, negatives, epoch) = load_meta(tx)?;
        update_meta(
            tx,
            states
                .checked_sub(removed_states)
                .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            negatives
                .checked_sub(removed_negatives)
                .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            epoch,
        )?;
    }
    Ok(DiscoveryEndpointQuarantineRevocationCleanup {
        removed_states,
        removed_negative_observations: removed_negatives,
    })
}

fn load_meta(
    tx: &Transaction<'_>,
) -> Result<(usize, usize, u64), DiscoveryEndpointQuarantineRevocationError> {
    let (states, negatives, epoch): (i64, i64, i64) = tx
        .query_row(
            "SELECT states,negatives,current_policy_epoch
             FROM discovery_endpoint_quarantine_revocation_meta_v1 WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    Ok((
        usize::try_from(states).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        usize::try_from(negatives)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        as_u64(epoch)?,
    ))
}

fn load_meta_connection(
    connection: &Connection,
) -> Result<(usize, usize, u64), DiscoveryEndpointQuarantineRevocationError> {
    let (states, negatives, epoch): (i64, i64, i64) = connection
        .query_row(
            "SELECT states,negatives,current_policy_epoch
             FROM discovery_endpoint_quarantine_revocation_meta_v1 WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    Ok((
        usize::try_from(states).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        usize::try_from(negatives)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        as_u64(epoch)?,
    ))
}

fn update_meta(
    tx: &Transaction<'_>,
    states: usize,
    negatives: usize,
    epoch: u64,
) -> Result<(), DiscoveryEndpointQuarantineRevocationError> {
    tx.execute(
        "UPDATE discovery_endpoint_quarantine_revocation_meta_v1
         SET states=?1,negatives=?2,current_policy_epoch=?3 WHERE singleton=1",
        params![
            i64::try_from(states)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            i64::try_from(negatives)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            as_i64(epoch)?,
        ],
    )
    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    Ok(())
}

fn negative_observation_commitment(
    request: &DiscoveryEndpointNegativeObservationRequest,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(NEGATIVE_OBSERVATION_DOMAIN);
    hash.update(request.evidence_id);
    hash.update(request.positive_commitment);
    hash.update(request.admission_commitment);
    hash.update(request.challenge_id);
    hash.update(request.policy_epoch.to_be_bytes());
    hash.update(request.observer_context);
    hash.update(request.observed_at.to_be_bytes());
    hash.update(request.expires_at.to_be_bytes());
    hash.finalize().into()
}

fn count_connection(
    connection: &Connection,
    query: &str,
) -> Result<usize, DiscoveryEndpointQuarantineRevocationError> {
    let count: i64 = connection
        .query_row(query, [], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    usize::try_from(count).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)
}

fn finish_positive(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointQuarantinePositiveOutcome,
) -> Result<DiscoveryEndpointQuarantinePositiveOutcome, DiscoveryEndpointQuarantineRevocationError>
{
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    Ok(outcome)
}

fn finish_negative(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointQuarantineNegativeOutcome,
) -> Result<DiscoveryEndpointQuarantineNegativeOutcome, DiscoveryEndpointQuarantineRevocationError>
{
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    Ok(outcome)
}

fn array32(value: Vec<u8>) -> Result<[u8; 32], DiscoveryEndpointQuarantineRevocationError> {
    value
        .try_into()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)
}

fn as_i64(value: u64) -> Result<i64, DiscoveryEndpointQuarantineRevocationError> {
    i64::try_from(value).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Rejected)
}

fn as_u64(value: i64) -> Result<u64, DiscoveryEndpointQuarantineRevocationError> {
    u64::try_from(value).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::discovery_endpoint_attestation_inbox::DiscoveryEndpointCandidateFacts;
    use crate::services::discovery_endpoint_eligibility::{
        evaluate_endpoint_candidate, DiscoveryEndpointEligibilityPolicy,
        DiscoveryEndpointStakePolicyMode,
    };
    use crate::services::discovery_endpoint_quarantine::{
        quarantine_admission_commitment, DiscoveryEndpointFreshQuarantineAdmission,
        DiscoveryEndpointQuarantineConfig, SqliteDiscoveryEndpointQuarantineRegistry,
    };
    use crate::services::discovery_endpoint_quarantine_observation::{
        DiscoveryEndpointObservationDirection, DiscoveryEndpointQuarantineChallengeOutcome,
        DiscoveryEndpointQuarantineEvidenceRequest,
        DiscoveryEndpointQuarantineEvidenceVerificationError,
        DiscoveryEndpointQuarantineEvidenceVerifier, DiscoveryEndpointQuarantineObservationConfig,
        SqliteDiscoveryEndpointQuarantineObservationRegistry,
    };
    use tempfile::TempDir;

    const NOW: u64 = 2_000_000_000;

    struct AcceptPositiveEvidence;

    impl DiscoveryEndpointQuarantineEvidenceVerifier for AcceptPositiveEvidence {
        fn verify(
            &self,
            _request: &DiscoveryEndpointQuarantineEvidenceRequest,
        ) -> Result<(), DiscoveryEndpointQuarantineEvidenceVerificationError> {
            Ok(())
        }
    }

    struct FixedNegative(Result<(), DiscoveryEndpointNegativeEvidenceVerificationError>);

    impl DiscoveryEndpointNegativeEvidenceVerifier for FixedNegative {
        fn verify(
            &self,
            _request: &DiscoveryEndpointNegativeObservationRequest,
        ) -> Result<(), DiscoveryEndpointNegativeEvidenceVerificationError> {
            self.0
        }
    }

    #[derive(Clone, Copy)]
    struct SatisfiedFixture {
        fresh: DiscoveryEndpointFreshQuarantineAdmission,
        evidence: DiscoveryEndpointSatisfiedQuarantineEvidence,
    }

    fn tempdir() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("external test root");
        TempDir::new_in("target/test-temp").expect("tempdir")
    }

    fn config(
        directory: &TempDir,
        name: &str,
        max_states: usize,
        max_negative_per_state: usize,
    ) -> DiscoveryEndpointQuarantineRevocationConfig {
        DiscoveryEndpointQuarantineRevocationConfig {
            db_path: directory.path().join(name),
            max_states,
            max_negative_per_state,
            negative_ttl_secs: 10,
            cleanup_batch_size: 8,
        }
    }

    fn satisfied(
        directory: &TempDir,
        seed: u8,
        policy_epoch: u64,
        valid_until: u64,
    ) -> SatisfiedFixture {
        let facts = DiscoveryEndpointCandidateFacts {
            group_commitment: [seed; 32],
            descriptor_sequence: 7,
            distinct_observers: 2,
            overlap_started_at: NOW - 1,
            overlap_expires_at: valid_until,
            newest_observed_at: NOW - 1,
            newest_expires_at: valid_until,
        };
        let admission = evaluate_endpoint_candidate(
            &facts,
            DiscoveryEndpointEligibilityPolicy::new(
                2,
                60,
                policy_epoch,
                DiscoveryEndpointStakePolicyMode::Disabled,
            )
            .expect("policy"),
            NOW,
            None,
        )
        .into_quarantine_admission()
        .expect("admission");
        let exact_admission = quarantine_admission_commitment(&admission);
        let quarantine =
            SqliteDiscoveryEndpointQuarantineRegistry::open(DiscoveryEndpointQuarantineConfig {
                db_path: directory
                    .path()
                    .join(format!("quarantine-{seed}-{policy_epoch}.sqlite3")),
                max_entries: 4,
                cleanup_batch_size: 4,
            })
            .expect("quarantine");
        quarantine.record_at(admission, NOW).expect("record");
        let fresh = quarantine
            .fresh_admission_at(exact_admission, NOW)
            .expect("fresh lookup")
            .expect("fresh admission");
        let observation = SqliteDiscoveryEndpointQuarantineObservationRegistry::open(
            DiscoveryEndpointQuarantineObservationConfig {
                db_path: directory
                    .path()
                    .join(format!("observation-{seed}-{policy_epoch}.sqlite3")),
                max_challenges: 4,
                max_attempts_per_challenge: 4,
                challenge_ttl_secs: 60,
                minimum_observation_span_secs: 5,
                cleanup_batch_size: 4,
            },
        )
        .expect("observation");
        let challenge = match observation
            .begin_at(
                fresh,
                [seed.wrapping_add(1); 32],
                [seed.wrapping_add(2); 32],
                NOW,
                NOW,
            )
            .expect("challenge")
        {
            DiscoveryEndpointQuarantineChallengeOutcome::Issued(value) => value,
            other => panic!("unexpected challenge outcome {other:?}"),
        };
        observation
            .observe_at(
                challenge,
                [seed.wrapping_add(3); 32],
                [seed.wrapping_add(4); 32],
                DiscoveryEndpointObservationDirection::OutboundChallenge,
                [seed.wrapping_add(5); 32],
                NOW + 1,
                NOW + 1,
                Some(&AcceptPositiveEvidence),
            )
            .expect("outbound");
        observation
            .observe_at(
                challenge,
                [seed.wrapping_add(6); 32],
                [seed.wrapping_add(7); 32],
                DiscoveryEndpointObservationDirection::InboundProof,
                [seed.wrapping_add(8); 32],
                NOW + 6,
                NOW + 6,
                Some(&AcceptPositiveEvidence),
            )
            .expect("inbound");
        let evidence = observation
            .satisfied_evidence_at(challenge, NOW + 6)
            .expect("satisfied lookup")
            .expect("satisfied evidence");
        SatisfiedFixture { fresh, evidence }
    }

    #[test]
    fn negative_replay_conflict_revocation_restart_and_debug_are_deterministic() {
        let directory = tempdir();
        let evidence = satisfied(&directory, 0x31, 1, NOW + 120).evidence;
        let cfg = config(&directory, "revocation.sqlite3", 4, 4);
        let registry =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg.clone()).expect("open");
        assert_eq!(
            registry
                .retain_positive_at(evidence, NOW + 6)
                .expect("positive"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        assert_eq!(
            registry
                .record_negative_at(evidence, [0x41; 32], [0x51; 32], NOW + 7, NOW + 7, None,)
                .expect("missing verifier"),
            DiscoveryEndpointQuarantineNegativeOutcome::EvidenceMissing
        );
        assert_eq!(
            registry
                .record_negative_at(
                    evidence,
                    [0x41; 32],
                    [0x51; 32],
                    NOW + 7,
                    NOW + 7,
                    Some(&FixedNegative(Err(
                        DiscoveryEndpointNegativeEvidenceVerificationError::Invalid,
                    ))),
                )
                .expect("invalid verifier"),
            DiscoveryEndpointQuarantineNegativeOutcome::EvidenceRejected
        );
        assert_eq!(
            registry
                .record_negative_at(
                    evidence,
                    [0x41; 32],
                    [0x51; 32],
                    NOW + 7,
                    NOW + 7,
                    Some(&FixedNegative(Err(
                        DiscoveryEndpointNegativeEvidenceVerificationError::Unavailable,
                    ))),
                )
                .expect("unavailable verifier"),
            DiscoveryEndpointQuarantineNegativeOutcome::EvidenceUnavailable
        );
        assert_eq!(
            registry
                .record_negative_at(
                    evidence,
                    [0x41; 32],
                    [0x51; 32],
                    NOW + 7,
                    NOW + 7,
                    Some(&FixedNegative(Ok(()))),
                )
                .expect("negative"),
            DiscoveryEndpointQuarantineNegativeOutcome::Revoked
        );
        assert_eq!(
            registry
                .record_negative_at(evidence, [0x41; 32], [0x51; 32], NOW + 7, NOW + 8, None,)
                .expect("exact replay"),
            DiscoveryEndpointQuarantineNegativeOutcome::Existing
        );
        assert_eq!(
            registry
                .record_negative_at(
                    evidence,
                    [0x41; 32],
                    [0x52; 32],
                    NOW + 7,
                    NOW + 8,
                    Some(&FixedNegative(Ok(()))),
                )
                .expect("conflict"),
            DiscoveryEndpointQuarantineNegativeOutcome::Conflict
        );
        assert_eq!(
            registry
                .retain_positive_at(evidence, NOW + 8)
                .expect("irreversible"),
            DiscoveryEndpointQuarantinePositiveOutcome::Revoked
        );
        let debug_request = DiscoveryEndpointNegativeObservationRequest {
            evidence_id: [0x61; 32],
            positive_commitment: [0x62; 32],
            admission_commitment: [0x63; 32],
            challenge_id: [0x64; 32],
            policy_epoch: 1,
            observer_context: [0x65; 32],
            observed_at: NOW,
            expires_at: NOW + 1,
        };
        let debug = format!("{registry:?}{evidence:?}{debug_request:?}");
        for secret in [0x61, 0x62, 0x63, 0x64, 0x65] {
            assert!(!debug.contains(&hex::encode([secret; 32])));
        }
        assert!(!debug.contains("revocation.sqlite3"));
        drop(registry);
        let reopened =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg).expect("restart");
        assert_eq!(
            reopened.snapshot_at(NOW + 8).expect("snapshot"),
            DiscoveryEndpointQuarantineRevocationSnapshot {
                current_policy_epoch: 1,
                retained_states: 1,
                current_positive_states: 0,
                revoked_states: 1,
                retained_negative_observations: 1,
            }
        );
    }

    #[test]
    fn negative_first_and_higher_epoch_are_irreversible_in_one_policy_domain() {
        let directory = tempdir();
        let epoch_one = satisfied(&directory, 0x32, 1, NOW + 120).evidence;
        let epoch_two = satisfied(&directory, 0x33, 2, NOW + 120).evidence;
        let registry = SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(config(
            &directory,
            "epoch.sqlite3",
            4,
            4,
        ))
        .expect("open");
        assert_eq!(
            registry
                .record_negative_at(
                    epoch_two,
                    [0x42; 32],
                    [0x52; 32],
                    NOW + 7,
                    NOW + 7,
                    Some(&FixedNegative(Ok(()))),
                )
                .expect("negative first"),
            DiscoveryEndpointQuarantineNegativeOutcome::Revoked
        );
        assert_eq!(
            registry
                .retain_positive_at(epoch_two, NOW + 7)
                .expect("positive race"),
            DiscoveryEndpointQuarantinePositiveOutcome::Revoked
        );
        assert_eq!(
            registry
                .retain_positive_at(epoch_one, NOW + 7)
                .expect("old epoch positive"),
            DiscoveryEndpointQuarantinePositiveOutcome::OldPolicyEpoch
        );
        assert_eq!(
            registry
                .record_negative_at(
                    epoch_one,
                    [0x43; 32],
                    [0x53; 32],
                    NOW + 7,
                    NOW + 7,
                    Some(&FixedNegative(Ok(()))),
                )
                .expect("old epoch negative"),
            DiscoveryEndpointQuarantineNegativeOutcome::OldPolicyEpoch
        );
        assert_eq!(
            registry.snapshot_at(NOW + 7).expect("snapshot"),
            DiscoveryEndpointQuarantineRevocationSnapshot {
                current_policy_epoch: 2,
                retained_states: 1,
                current_positive_states: 0,
                revoked_states: 1,
                retained_negative_observations: 1,
            }
        );

        let capacity_epoch = SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(config(
            &directory,
            "epoch-capacity.sqlite3",
            1,
            1,
        ))
        .expect("capacity epoch");
        assert_eq!(
            capacity_epoch
                .retain_positive_at(epoch_one, NOW + 7)
                .expect("epoch one"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        assert_eq!(
            capacity_epoch
                .retain_positive_at(epoch_two, NOW + 7)
                .expect("epoch two replaces old capacity"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        assert_eq!(
            capacity_epoch.snapshot_at(NOW + 7).expect("snapshot"),
            DiscoveryEndpointQuarantineRevocationSnapshot {
                current_policy_epoch: 2,
                retained_states: 1,
                current_positive_states: 1,
                revoked_states: 0,
                retained_negative_observations: 0,
            }
        );
    }

    #[test]
    fn readiness_is_deterministic_private_and_revocation_invalidates_it() {
        let directory = tempdir();
        let fixture = satisfied(&directory, 0x36, 1, NOW + 120);
        let other = satisfied(&directory, 0x37, 1, NOW + 120);
        let registry = SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(config(
            &directory,
            "readiness.sqlite3",
            4,
            4,
        ))
        .expect("open");
        assert_eq!(
            registry
                .retain_positive_at(fixture.evidence, NOW + 6)
                .expect("positive"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        let first = registry
            .promotion_readiness_at(fixture.fresh, fixture.evidence, NOW + 6)
            .expect("evaluate")
            .expect("ready");
        let repeated = registry
            .promotion_readiness_at(fixture.fresh, fixture.evidence, NOW + 6)
            .expect("repeat")
            .expect("ready repeat");
        assert_eq!(first, repeated);
        assert_eq!(first.descriptor_sequence(), 7);
        assert_eq!(first.policy_epoch(), 1);
        assert_eq!(first.evaluated_at(), NOW + 6);
        assert!(first.valid_until() <= NOW + 126);
        assert_eq!(
            first.admission_commitment(),
            fixture.fresh.admission_commitment()
        );
        assert!(registry
            .verify_promotion_readiness_at(&first, NOW + 6)
            .expect("verify"));
        assert!(registry
            .promotion_readiness_at(other.fresh, fixture.evidence, NOW + 6)
            .expect("mismatched admission")
            .is_none());

        let debug = format!("{first:?}");
        for secret in [
            first.readiness_commitment(),
            first.admission_commitment(),
            first.group_commitment(),
            fixture.evidence.evidence_commitment(),
            fixture.evidence.challenge_id(),
        ] {
            assert!(!debug.contains(&hex::encode(secret)));
        }
        let mut tampered = first;
        tampered.descriptor_sequence = tampered.descriptor_sequence.saturating_add(1);
        assert!(!registry
            .verify_promotion_readiness_at(&tampered, NOW + 6)
            .expect("tamper rejection"));

        assert_eq!(
            registry
                .record_negative_at(
                    fixture.evidence,
                    [0x46; 32],
                    [0x56; 32],
                    NOW + 7,
                    NOW + 7,
                    Some(&FixedNegative(Ok(()))),
                )
                .expect("revoke"),
            DiscoveryEndpointQuarantineNegativeOutcome::Revoked
        );
        assert!(!registry
            .verify_promotion_readiness_at(&first, NOW + 7)
            .expect("revoked token"));
        assert!(registry
            .promotion_readiness_at(fixture.fresh, fixture.evidence, NOW + 7)
            .expect("reacquire after revoke")
            .is_none());
    }

    #[test]
    fn epoch_advance_and_expiry_prevent_readiness_reacquisition() {
        let directory = tempdir();
        let epoch_one = satisfied(&directory, 0x38, 1, NOW + 120);
        let epoch_two = satisfied(&directory, 0x39, 2, NOW + 120);
        let registry = SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(config(
            &directory,
            "readiness-epoch.sqlite3",
            4,
            4,
        ))
        .expect("open");
        registry
            .retain_positive_at(epoch_one.evidence, NOW + 6)
            .expect("epoch one");
        let old = registry
            .promotion_readiness_at(epoch_one.fresh, epoch_one.evidence, NOW + 6)
            .expect("epoch one readiness")
            .expect("epoch one ready");
        registry
            .retain_positive_at(epoch_two.evidence, NOW + 6)
            .expect("epoch two");
        assert!(!registry
            .verify_promotion_readiness_at(&old, NOW + 6)
            .expect("old epoch invalid"));
        assert!(registry
            .promotion_readiness_at(epoch_one.fresh, epoch_one.evidence, NOW + 6)
            .expect("old epoch reacquire")
            .is_none());
        assert!(registry
            .promotion_readiness_at(epoch_two.fresh, epoch_two.evidence, NOW + 121)
            .expect("expired")
            .is_none());
    }

    #[test]
    fn capacity_ttl_cleanup_corruption_and_routeability_boundaries_fail_closed() {
        let directory = tempdir();
        let first = satisfied(&directory, 0x34, 1, NOW + 20).evidence;
        let second = satisfied(&directory, 0x35, 1, NOW + 120).evidence;
        let cfg = config(&directory, "capacity.sqlite3", 1, 1);
        let registry =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg.clone()).expect("open");
        assert_eq!(
            registry.retain_positive_at(first, NOW + 6).expect("first"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        assert_eq!(
            registry
                .retain_positive_at(second, NOW + 6)
                .expect("capacity"),
            DiscoveryEndpointQuarantinePositiveOutcome::AtCapacity
        );
        assert_eq!(
            registry
                .record_negative_at(
                    first,
                    [0x44; 32],
                    [0x54; 32],
                    NOW + 7,
                    NOW + 7,
                    Some(&FixedNegative(Ok(()))),
                )
                .expect("negative"),
            DiscoveryEndpointQuarantineNegativeOutcome::Revoked
        );
        assert_eq!(
            registry
                .record_negative_at(
                    first,
                    [0x45; 32],
                    [0x55; 32],
                    NOW + 8,
                    NOW + 8,
                    Some(&FixedNegative(Ok(()))),
                )
                .expect("negative capacity"),
            DiscoveryEndpointQuarantineNegativeOutcome::AtCapacity
        );
        let cleanup = registry.cleanup_expired_at(NOW + 21, 8).expect("cleanup");
        assert_eq!(cleanup.removed_states, 1);
        assert_eq!(cleanup.removed_negative_observations, 1);
        assert_eq!(
            registry
                .retain_positive_at(first, NOW + 21)
                .expect("expired"),
            DiscoveryEndpointQuarantinePositiveOutcome::Expired
        );
        assert_eq!(
            registry
                .retain_positive_at(second, NOW + 21)
                .expect("after cleanup"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        let source = include_str!("discovery_endpoint_quarantine_revocation.rs");
        for forbidden in [
            concat!("Peer", "Store"),
            concat!("upsert_", "verified"),
            concat!("route_", "candidates"),
            concat!("services::", "routing"),
            concat!("promote_", "candidate"),
        ] {
            assert!(!source.contains(forbidden), "forbidden symbol: {forbidden}");
        }
        drop(registry);
        rusqlite::Connection::open(&cfg.db_path)
            .expect("database")
            .execute(
                "UPDATE discovery_endpoint_quarantine_revocation_meta_v1 SET states=9",
                [],
            )
            .expect("tamper");
        assert!(matches!(
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg),
            Err(DiscoveryEndpointQuarantineRevocationError::Corrupt)
        ));
    }
}
