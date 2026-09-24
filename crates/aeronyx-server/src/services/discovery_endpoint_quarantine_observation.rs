// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_quarantine_observation.rs
// ============================================================================
//! Durable challenge and observation policy for quarantined endpoint evidence.
//!
//! This state machine consumes only fresh opaque capabilities from the
//! quarantine registry. Evidence satisfaction remains quarantined state: this
//! module has no candidate identity, endpoint, descriptor, ranking, routing,
//! promotion, advertisement, server, API, or network authority.
// [PERMISSIONLESS-ENDPOINT-QUARANTINE-OBSERVATION 2026-09-24 by Codex] Keep
// evidence accumulation separate from every routeable peer projection.

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

const SCHEMA_VERSION: i64 = 1;
const MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
const MAX_CHALLENGES: usize = 65_536;
const MAX_ATTEMPTS_PER_CHALLENGE: usize = 64;
const MAX_CHALLENGE_TTL_SECS: u64 = 24 * 60 * 60;
const MAX_CLEANUP_BATCH: usize = 4_096;
const CHALLENGE_ID_DOMAIN: &[u8] = b"AeroNyx/EndpointQuarantineChallengeIdV1\0";
const CHALLENGE_REQUEST_DOMAIN: &[u8] = b"AeroNyx/EndpointQuarantineChallengeRequestV1\0";
const OBSERVATION_DOMAIN: &[u8] = b"AeroNyx/EndpointQuarantineObservationV1\0";
const SATISFIED_EVIDENCE_DOMAIN: &[u8] = b"AeroNyx/EndpointQuarantineSatisfiedEvidenceV1\0";

/// Bounded policy for an isolated challenge/observation registry.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineObservationConfig {
    pub(crate) db_path: PathBuf,
    pub(crate) max_challenges: usize,
    pub(crate) max_attempts_per_challenge: usize,
    pub(crate) challenge_ttl_secs: u64,
    pub(crate) minimum_observation_span_secs: u64,
    pub(crate) cleanup_batch_size: usize,
}

impl fmt::Debug for DiscoveryEndpointQuarantineObservationConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointQuarantineObservationConfig")
            .field("max_challenges", &self.max_challenges)
            .field(
                "max_attempts_per_challenge",
                &self.max_attempts_per_challenge,
            )
            .field("challenge_ttl_secs", &self.challenge_ttl_secs)
            .field(
                "minimum_observation_span_secs",
                &self.minimum_observation_span_secs,
            )
            .field("cleanup_batch_size", &self.cleanup_batch_size)
            .finish_non_exhaustive()
    }
}

/// Coarse failures that expose no challenge, candidate, observer, or path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DiscoveryEndpointQuarantineObservationError {
    #[error("endpoint quarantine observation rejected")]
    Rejected,
    #[error("endpoint quarantine observation schema unsupported")]
    UnsupportedSchema,
    #[error("endpoint quarantine observation registry corrupt")]
    Corrupt,
    #[error("endpoint quarantine observation registry unavailable")]
    Unavailable,
}

/// One of the two independently verified directions required by policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum DiscoveryEndpointObservationDirection {
    OutboundChallenge = 1,
    InboundProof = 2,
}

impl DiscoveryEndpointObservationDirection {
    fn from_i64(value: i64) -> Result<Self, DiscoveryEndpointQuarantineObservationError> {
        match value {
            1 => Ok(Self::OutboundChallenge),
            2 => Ok(Self::InboundProof),
            _ => Err(DiscoveryEndpointQuarantineObservationError::Corrupt),
        }
    }
}

/// Typed challenge handle; all commitments remain private and redacted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineChallenge {
    challenge_id: [u8; 32],
    admission_commitment: [u8; 32],
    request_commitment: [u8; 32],
    policy_version: u64,
    issued_at: u64,
    expires_at: u64,
}

impl fmt::Debug for DiscoveryEndpointQuarantineChallenge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointQuarantineChallenge")
            .field("policy_version", &self.policy_version)
            .field("issued_at", &self.issued_at)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

/// Registry-minted proof that one fresh challenge currently meets policy.
// [PERMISSIONLESS-ENDPOINT-QUARANTINE-REVOCATION 2026-09-24 by Codex] This
// capability is the only positive input accepted by the revocation boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointSatisfiedQuarantineEvidence {
    evidence_commitment: [u8; 32],
    admission_commitment: [u8; 32],
    challenge_id: [u8; 32],
    policy_epoch: u64,
    valid_until: u64,
}

impl DiscoveryEndpointSatisfiedQuarantineEvidence {
    pub(crate) const fn evidence_commitment(&self) -> [u8; 32] {
        self.evidence_commitment
    }

    pub(crate) const fn admission_commitment(&self) -> [u8; 32] {
        self.admission_commitment
    }

    pub(crate) const fn challenge_id(&self) -> [u8; 32] {
        self.challenge_id
    }

    pub(crate) const fn policy_epoch(&self) -> u64 {
        self.policy_epoch
    }

    pub(crate) const fn valid_until(&self) -> u64 {
        self.valid_until
    }
}

impl fmt::Debug for DiscoveryEndpointSatisfiedQuarantineEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointSatisfiedQuarantineEvidence")
            .field("policy_epoch", &self.policy_epoch)
            .field("valid_until", &self.valid_until)
            .finish_non_exhaustive()
    }
}

/// Challenge creation preserves exact retry while rejecting identifier reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointQuarantineChallengeOutcome {
    Issued(DiscoveryEndpointQuarantineChallenge),
    Existing(DiscoveryEndpointQuarantineChallenge),
    Conflict,
    Expired,
    AtCapacity,
}

/// Evidence state remains non-routeable even after its threshold is met.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointQuarantineEvidenceState {
    Pending,
    EvidenceSatisfied,
}

/// Observation mutation result with exact replay and conflict semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointQuarantineObservationOutcome {
    Recorded(DiscoveryEndpointQuarantineEvidenceState),
    Existing(DiscoveryEndpointQuarantineEvidenceState),
    Conflict,
    Expired,
    AtCapacity,
    EvidenceMissing,
    EvidenceRejected,
    EvidenceUnavailable,
}

/// Exact verification request sent to an injected cryptographic verifier.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineEvidenceRequest {
    pub(crate) challenge_id: [u8; 32],
    pub(crate) admission_commitment: [u8; 32],
    pub(crate) attempt_id: [u8; 32],
    pub(crate) observer_context: [u8; 32],
    pub(crate) direction: DiscoveryEndpointObservationDirection,
    pub(crate) evidence_commitment: [u8; 32],
    pub(crate) observed_at: u64,
}

impl fmt::Debug for DiscoveryEndpointQuarantineEvidenceRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointQuarantineEvidenceRequest")
            .field("direction", &self.direction)
            .field("observed_at", &self.observed_at)
            .finish_non_exhaustive()
    }
}

/// Coarse result from the replaceable evidence-verification boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointQuarantineEvidenceVerificationError {
    Invalid,
    Unavailable,
}

/// Mandatory verifier for already authenticated, commitment-bound evidence.
pub(crate) trait DiscoveryEndpointQuarantineEvidenceVerifier: Send + Sync {
    fn verify(
        &self,
        request: &DiscoveryEndpointQuarantineEvidenceRequest,
    ) -> Result<(), DiscoveryEndpointQuarantineEvidenceVerificationError>;
}

/// Aggregate-only status with no candidate or observer projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineObservationSnapshot {
    pub(crate) retained_challenges: usize,
    pub(crate) retained_observations: usize,
    pub(crate) fresh_challenges: usize,
    pub(crate) evidence_satisfied_challenges: usize,
}

/// Dedicated durable state that cannot emit a routeable candidate.
pub(crate) struct SqliteDiscoveryEndpointQuarantineObservationRegistry {
    config: DiscoveryEndpointQuarantineObservationConfig,
    connection: Mutex<Connection>,
    #[cfg(unix)]
    _database_parent: File,
}

impl fmt::Debug for SqliteDiscoveryEndpointQuarantineObservationRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteDiscoveryEndpointQuarantineObservationRegistry")
            .field("max_challenges", &self.config.max_challenges)
            .field(
                "max_attempts_per_challenge",
                &self.config.max_attempts_per_challenge,
            )
            .finish_non_exhaustive()
    }
}

impl SqliteDiscoveryEndpointQuarantineObservationRegistry {
    pub(crate) fn open(
        config: DiscoveryEndpointQuarantineObservationConfig,
    ) -> Result<Self, DiscoveryEndpointQuarantineObservationError> {
        validate_config(&config)?;
        let target = prepare_private_sqlite_target(&config.db_path)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        verify_private_file(&target.resolved_path, false)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        restrict_private_sqlite_permissions(&target.resolved_path)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        verify_sqlite_physical_integrity(&connection, "endpoint_quarantine_observation_startup")
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?;
        configure_full_durability(&connection, MINIMUM_SYNCHRONOUS_LEVEL)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        initialize_schema(&mut connection)?;
        startup_audit(&connection, &config)?;
        Ok(Self {
            config,
            connection: Mutex::new(connection),
            #[cfg(unix)]
            _database_parent: target.parent,
        })
    }

    /// Creates or exactly replays a challenge for one fresh admission.
    pub(crate) fn begin_at(
        &self,
        admission: DiscoveryEndpointFreshQuarantineAdmission,
        challenge_nonce: [u8; 32],
        challenger_context: [u8; 32],
        issued_at: u64,
        now: u64,
    ) -> Result<
        DiscoveryEndpointQuarantineChallengeOutcome,
        DiscoveryEndpointQuarantineObservationError,
    > {
        if now == 0
            || issued_at == 0
            || issued_at > now
            || challenge_nonce.iter().all(|byte| *byte == 0)
            || challenger_context.iter().all(|byte| *byte == 0)
            || admission
                .admission_commitment()
                .iter()
                .all(|byte| *byte == 0)
            || admission.policy_version() == 0
        {
            return Err(DiscoveryEndpointQuarantineObservationError::Rejected);
        }
        let expires_at = issued_at
            .checked_add(self.config.challenge_ttl_secs)
            .ok_or(DiscoveryEndpointQuarantineObservationError::Rejected)?
            .min(admission.valid_until());
        if admission.valid_until() < now || expires_at < now || expires_at < issued_at {
            return Ok(DiscoveryEndpointQuarantineChallengeOutcome::Expired);
        }
        let admission_commitment = admission.admission_commitment();
        let challenge_id =
            challenge_id(&admission_commitment, &challenge_nonce, &challenger_context);
        let request_commitment = challenge_request_commitment(
            &challenge_id,
            &admission_commitment,
            admission.policy_version(),
            &challenge_nonce,
            &challenger_context,
            issued_at,
            expires_at,
        );
        let challenge = DiscoveryEndpointQuarantineChallenge {
            challenge_id,
            admission_commitment,
            request_commitment,
            policy_version: admission.policy_version(),
            issued_at,
            expires_at,
        };
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        if let Some(existing) = load_challenge(&tx, &challenge_id)? {
            if existing.request_commitment != request_commitment {
                return finish(tx, DiscoveryEndpointQuarantineChallengeOutcome::Conflict);
            }
            let outcome = if existing.expires_at < now {
                DiscoveryEndpointQuarantineChallengeOutcome::Expired
            } else {
                DiscoveryEndpointQuarantineChallengeOutcome::Existing(existing)
            };
            return finish(tx, outcome);
        }
        cleanup_tx(&tx, now, self.config.cleanup_batch_size)?;
        let (challenges, observations) = load_meta(&tx)?;
        if challenges >= self.config.max_challenges {
            return finish(tx, DiscoveryEndpointQuarantineChallengeOutcome::AtCapacity);
        }
        tx.execute(
            "INSERT INTO discovery_endpoint_quarantine_challenge_v1(
               challenge_id,admission_commitment,request_commitment,challenge_nonce,
               challenger_context,policy_version,issued_at,expires_at
             ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                &challenge_id[..],
                &admission_commitment[..],
                &request_commitment[..],
                &challenge_nonce[..],
                &challenger_context[..],
                as_i64(admission.policy_version())?,
                as_i64(issued_at)?,
                as_i64(expires_at)?,
            ],
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        update_meta(
            &tx,
            challenges
                .checked_add(1)
                .ok_or(DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            observations,
        )?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        Ok(DiscoveryEndpointQuarantineChallengeOutcome::Issued(
            challenge,
        ))
    }

    /// Records one verifier-approved directional observation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn observe_at(
        &self,
        challenge: DiscoveryEndpointQuarantineChallenge,
        attempt_id: [u8; 32],
        observer_context: [u8; 32],
        direction: DiscoveryEndpointObservationDirection,
        evidence_commitment: [u8; 32],
        observed_at: u64,
        now: u64,
        verifier: Option<&dyn DiscoveryEndpointQuarantineEvidenceVerifier>,
    ) -> Result<
        DiscoveryEndpointQuarantineObservationOutcome,
        DiscoveryEndpointQuarantineObservationError,
    > {
        if now == 0
            || observed_at == 0
            || observed_at > now
            || attempt_id.iter().all(|byte| *byte == 0)
            || observer_context.iter().all(|byte| *byte == 0)
            || evidence_commitment.iter().all(|byte| *byte == 0)
        {
            return Err(DiscoveryEndpointQuarantineObservationError::Rejected);
        }
        let observation_commitment = observation_commitment(
            &challenge.challenge_id,
            &challenge.admission_commitment,
            &attempt_id,
            &observer_context,
            direction,
            &evidence_commitment,
            observed_at,
        );
        let request = DiscoveryEndpointQuarantineEvidenceRequest {
            challenge_id: challenge.challenge_id,
            admission_commitment: challenge.admission_commitment,
            attempt_id,
            observer_context,
            direction,
            evidence_commitment,
            observed_at,
        };
        {
            let mut connection = self
                .connection
                .lock()
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
            let tx = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
            if let Some(outcome) = preflight_observation(
                &tx,
                &challenge,
                &request,
                &observation_commitment,
                now,
                &self.config,
            )? {
                return finish_observation(tx, outcome);
            }
            tx.commit()
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        }
        let Some(verifier) = verifier else {
            return Ok(DiscoveryEndpointQuarantineObservationOutcome::EvidenceMissing);
        };
        match verifier.verify(&request) {
            Ok(()) => {}
            Err(DiscoveryEndpointQuarantineEvidenceVerificationError::Invalid) => {
                return Ok(DiscoveryEndpointQuarantineObservationOutcome::EvidenceRejected);
            }
            Err(DiscoveryEndpointQuarantineEvidenceVerificationError::Unavailable) => {
                return Ok(DiscoveryEndpointQuarantineObservationOutcome::EvidenceUnavailable);
            }
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        if let Some(outcome) = preflight_observation(
            &tx,
            &challenge,
            &request,
            &observation_commitment,
            now,
            &self.config,
        )? {
            return finish_observation(tx, outcome);
        }
        tx.execute(
            "INSERT INTO discovery_endpoint_quarantine_observation_v1(
               attempt_id,challenge_id,observer_context,direction,evidence_commitment,
               observation_commitment,observed_at
             ) VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                &attempt_id[..],
                &challenge.challenge_id[..],
                &observer_context[..],
                i64::from(direction as u8),
                &evidence_commitment[..],
                &observation_commitment[..],
                as_i64(observed_at)?,
            ],
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let (challenges, observations) = load_meta(&tx)?;
        update_meta(
            &tx,
            challenges,
            observations
                .checked_add(1)
                .ok_or(DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let state = evidence_state(
            &tx,
            &challenge.challenge_id,
            self.config.minimum_observation_span_secs,
        )?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        Ok(DiscoveryEndpointQuarantineObservationOutcome::Recorded(
            state,
        ))
    }

    /// Returns a positive capability only while the exact evidence is fresh.
    pub(crate) fn satisfied_evidence_at(
        &self,
        challenge: DiscoveryEndpointQuarantineChallenge,
        now: u64,
    ) -> Result<
        Option<DiscoveryEndpointSatisfiedQuarantineEvidence>,
        DiscoveryEndpointQuarantineObservationError,
    > {
        if now == 0 {
            return Err(DiscoveryEndpointQuarantineObservationError::Rejected);
        }
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let Some(stored) = load_challenge_connection(&connection, &challenge.challenge_id)? else {
            return Ok(None);
        };
        if stored != challenge {
            return Err(DiscoveryEndpointQuarantineObservationError::Corrupt);
        }
        if challenge.expires_at < now
            || evidence_state_connection(
                &connection,
                &challenge.challenge_id,
                self.config.minimum_observation_span_secs,
            )? != DiscoveryEndpointQuarantineEvidenceState::EvidenceSatisfied
        {
            return Ok(None);
        }
        Ok(Some(DiscoveryEndpointSatisfiedQuarantineEvidence {
            evidence_commitment: satisfied_quarantine_evidence_commitment(
                &challenge.admission_commitment,
                &challenge.challenge_id,
                challenge.policy_version,
                challenge.expires_at,
            ),
            admission_commitment: challenge.admission_commitment,
            challenge_id: challenge.challenge_id,
            policy_epoch: challenge.policy_version,
            valid_until: challenge.expires_at,
        }))
    }

    pub(crate) fn cleanup_expired_at(
        &self,
        now: u64,
        limit: usize,
    ) -> Result<usize, DiscoveryEndpointQuarantineObservationError> {
        if now == 0 || limit == 0 || limit > MAX_CLEANUP_BATCH {
            return Err(DiscoveryEndpointQuarantineObservationError::Rejected);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let removed = cleanup_tx(&tx, now, limit)?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        Ok(removed)
    }

    pub(crate) fn snapshot_at(
        &self,
        now: u64,
    ) -> Result<
        DiscoveryEndpointQuarantineObservationSnapshot,
        DiscoveryEndpointQuarantineObservationError,
    > {
        if now == 0 {
            return Err(DiscoveryEndpointQuarantineObservationError::Rejected);
        }
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let (retained_challenges, retained_observations) = load_meta_connection(&connection)?;
        let fresh: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM discovery_endpoint_quarantine_challenge_v1
                 WHERE expires_at>=?1",
                params![as_i64(now)?],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let mut satisfied = 0usize;
        let mut statement = connection
            .prepare(
                "SELECT challenge_id FROM discovery_endpoint_quarantine_challenge_v1
                 WHERE expires_at>=?1 ORDER BY challenge_id",
            )
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        let ids = statement
            .query_map(params![as_i64(now)?], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        for id in ids {
            let id =
                array32(id.map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?)?;
            if evidence_state_connection(
                &connection,
                &id,
                self.config.minimum_observation_span_secs,
            )? == DiscoveryEndpointQuarantineEvidenceState::EvidenceSatisfied
            {
                satisfied = satisfied
                    .checked_add(1)
                    .ok_or(DiscoveryEndpointQuarantineObservationError::Corrupt)?;
            }
        }
        Ok(DiscoveryEndpointQuarantineObservationSnapshot {
            retained_challenges,
            retained_observations,
            fresh_challenges: usize::try_from(fresh)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            evidence_satisfied_challenges: satisfied,
        })
    }
}

fn preflight_observation(
    tx: &Transaction<'_>,
    challenge: &DiscoveryEndpointQuarantineChallenge,
    request: &DiscoveryEndpointQuarantineEvidenceRequest,
    exact_observation_commitment: &[u8; 32],
    now: u64,
    config: &DiscoveryEndpointQuarantineObservationConfig,
) -> Result<
    Option<DiscoveryEndpointQuarantineObservationOutcome>,
    DiscoveryEndpointQuarantineObservationError,
> {
    let Some(stored) = load_challenge(tx, &challenge.challenge_id)? else {
        return Err(DiscoveryEndpointQuarantineObservationError::Rejected);
    };
    if stored != *challenge {
        return Ok(Some(
            DiscoveryEndpointQuarantineObservationOutcome::Conflict,
        ));
    }
    if let Some(existing) = load_observation_commitment(tx, &request.attempt_id)? {
        if existing != *exact_observation_commitment {
            return Ok(Some(
                DiscoveryEndpointQuarantineObservationOutcome::Conflict,
            ));
        }
        let state = evidence_state(
            tx,
            &challenge.challenge_id,
            config.minimum_observation_span_secs,
        )?;
        return Ok(Some(
            DiscoveryEndpointQuarantineObservationOutcome::Existing(state),
        ));
    }
    if challenge.expires_at < now
        || request.observed_at < challenge.issued_at
        || request.observed_at > challenge.expires_at
    {
        return Ok(Some(DiscoveryEndpointQuarantineObservationOutcome::Expired));
    }
    if challenge_observation_count(tx, &challenge.challenge_id)?
        >= config.max_attempts_per_challenge
    {
        return Ok(Some(
            DiscoveryEndpointQuarantineObservationOutcome::AtCapacity,
        ));
    }
    Ok(None)
}

fn validate_config(
    config: &DiscoveryEndpointQuarantineObservationConfig,
) -> Result<(), DiscoveryEndpointQuarantineObservationError> {
    if config.db_path.as_os_str().is_empty()
        || config.db_path == Path::new(":memory:")
        || config.max_challenges == 0
        || config.max_challenges > MAX_CHALLENGES
        || config.max_attempts_per_challenge < 2
        || config.max_attempts_per_challenge > MAX_ATTEMPTS_PER_CHALLENGE
        || config.challenge_ttl_secs == 0
        || config.challenge_ttl_secs > MAX_CHALLENGE_TTL_SECS
        || config.minimum_observation_span_secs == 0
        || config.minimum_observation_span_secs >= config.challenge_ttl_secs
        || config.cleanup_batch_size == 0
        || config.cleanup_batch_size > MAX_CLEANUP_BATCH
    {
        return Err(DiscoveryEndpointQuarantineObservationError::Rejected);
    }
    config
        .max_challenges
        .checked_mul(config.max_attempts_per_challenge)
        .ok_or(DiscoveryEndpointQuarantineObservationError::Rejected)?;
    Ok(())
}

fn initialize_schema(
    connection: &mut Connection,
) -> Result<(), DiscoveryEndpointQuarantineObservationError> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    let version: i64 = tx
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    if version == 0 {
        let foreign: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
        if foreign != 0 {
            return Err(DiscoveryEndpointQuarantineObservationError::UnsupportedSchema);
        }
        tx.execute_batch(
            "CREATE TABLE discovery_endpoint_quarantine_observation_meta_v1(
               singleton INTEGER PRIMARY KEY CHECK(singleton=1),
               challenges INTEGER NOT NULL,observations INTEGER NOT NULL
             );
             INSERT INTO discovery_endpoint_quarantine_observation_meta_v1 VALUES(1,0,0);
             CREATE TABLE discovery_endpoint_quarantine_challenge_v1(
               challenge_id BLOB PRIMARY KEY CHECK(length(challenge_id)=32),
               admission_commitment BLOB NOT NULL CHECK(length(admission_commitment)=32),
               request_commitment BLOB NOT NULL CHECK(length(request_commitment)=32),
               challenge_nonce BLOB NOT NULL CHECK(length(challenge_nonce)=32),
               challenger_context BLOB NOT NULL CHECK(length(challenger_context)=32),
               policy_version INTEGER NOT NULL,issued_at INTEGER NOT NULL,expires_at INTEGER NOT NULL
             );
             CREATE INDEX discovery_endpoint_quarantine_challenge_expiry_v1
               ON discovery_endpoint_quarantine_challenge_v1(expires_at,challenge_id);
             CREATE TABLE discovery_endpoint_quarantine_observation_v1(
               attempt_id BLOB PRIMARY KEY CHECK(length(attempt_id)=32),
               challenge_id BLOB NOT NULL CHECK(length(challenge_id)=32),
               observer_context BLOB NOT NULL CHECK(length(observer_context)=32),
               direction INTEGER NOT NULL CHECK(direction IN (1,2)),
               evidence_commitment BLOB NOT NULL CHECK(length(evidence_commitment)=32),
               observation_commitment BLOB NOT NULL UNIQUE CHECK(length(observation_commitment)=32),
               observed_at INTEGER NOT NULL,
               FOREIGN KEY(challenge_id) REFERENCES discovery_endpoint_quarantine_challenge_v1(challenge_id) ON DELETE CASCADE
             );
             CREATE INDEX discovery_endpoint_quarantine_observation_challenge_v1
               ON discovery_endpoint_quarantine_observation_v1(challenge_id,observed_at,attempt_id);
             PRAGMA user_version=1;",
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    } else if version != SCHEMA_VERSION {
        return Err(DiscoveryEndpointQuarantineObservationError::UnsupportedSchema);
    }
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)
}

fn startup_audit(
    connection: &Connection,
    config: &DiscoveryEndpointQuarantineObservationConfig,
) -> Result<(), DiscoveryEndpointQuarantineObservationError> {
    let (meta_challenges, meta_observations) = load_meta_connection(connection)?;
    let actual_challenges = count_connection(
        connection,
        "SELECT COUNT(*) FROM discovery_endpoint_quarantine_challenge_v1",
    )?;
    let actual_observations = count_connection(
        connection,
        "SELECT COUNT(*) FROM discovery_endpoint_quarantine_observation_v1",
    )?;
    let maximum_observations = config
        .max_challenges
        .checked_mul(config.max_attempts_per_challenge)
        .ok_or(DiscoveryEndpointQuarantineObservationError::Corrupt)?;
    if meta_challenges != actual_challenges
        || meta_observations != actual_observations
        || actual_challenges > config.max_challenges
        || actual_observations > maximum_observations
    {
        return Err(DiscoveryEndpointQuarantineObservationError::Corrupt);
    }
    let mut challenge_statement = connection
        .prepare(
            "SELECT challenge_id,admission_commitment,request_commitment,challenge_nonce,
                    challenger_context,policy_version,issued_at,expires_at
             FROM discovery_endpoint_quarantine_challenge_v1",
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    let mut rows = challenge_statement
        .query([])
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?
    {
        let challenge = DiscoveryEndpointQuarantineChallenge {
            challenge_id: array32(
                row.get(0)
                    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            )?,
            admission_commitment: array32(
                row.get(1)
                    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            )?,
            request_commitment: array32(
                row.get(2)
                    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            )?,
            policy_version: as_u64(
                row.get(5)
                    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            )?,
            issued_at: as_u64(
                row.get(6)
                    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            )?,
            expires_at: as_u64(
                row.get(7)
                    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            )?,
        };
        let nonce = array32(
            row.get(3)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let context = array32(
            row.get(4)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let policy = as_u64(
            row.get(5)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        if challenge.challenge_id != challenge_id(&challenge.admission_commitment, &nonce, &context)
            || challenge.request_commitment
                != challenge_request_commitment(
                    &challenge.challenge_id,
                    &challenge.admission_commitment,
                    policy,
                    &nonce,
                    &context,
                    challenge.issued_at,
                    challenge.expires_at,
                )
            || policy == 0
            || nonce.iter().all(|byte| *byte == 0)
            || context.iter().all(|byte| *byte == 0)
            || challenge.expires_at < challenge.issued_at
        {
            return Err(DiscoveryEndpointQuarantineObservationError::Corrupt);
        }
    }
    let mut observation_statement = connection
        .prepare(
            "SELECT attempt_id,challenge_id,observer_context,direction,evidence_commitment,
                    observation_commitment,observed_at
             FROM discovery_endpoint_quarantine_observation_v1",
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    let mut rows = observation_statement
        .query([])
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?
    {
        let attempt = array32(
            row.get(0)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let challenge = array32(
            row.get(1)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let observer = array32(
            row.get(2)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let direction = DiscoveryEndpointObservationDirection::from_i64(
            row.get(3)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let evidence = array32(
            row.get(4)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let stored = array32(
            row.get(5)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let observed_at = as_u64(
            row.get(6)
                .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
        let parent = connection
            .query_row(
                "SELECT admission_commitment,issued_at,expires_at FROM discovery_endpoint_quarantine_challenge_v1 WHERE challenge_id=?1",
                params![&challenge[..]],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
            )
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?;
        let admission = array32(parent.0)?;
        let issued_at = as_u64(parent.1)?;
        let expires_at = as_u64(parent.2)?;
        if attempt.iter().all(|byte| *byte == 0)
            || observer.iter().all(|byte| *byte == 0)
            || evidence.iter().all(|byte| *byte == 0)
            || observed_at < issued_at
            || observed_at > expires_at
            || stored
                != observation_commitment(
                    &challenge,
                    &admission,
                    &attempt,
                    &observer,
                    direction,
                    &evidence,
                    observed_at,
                )
        {
            return Err(DiscoveryEndpointQuarantineObservationError::Corrupt);
        }
    }
    Ok(())
}

fn load_challenge(
    tx: &Transaction<'_>,
    challenge_id: &[u8; 32],
) -> Result<Option<DiscoveryEndpointQuarantineChallenge>, DiscoveryEndpointQuarantineObservationError>
{
    load_challenge_connection(tx, challenge_id)
}

fn load_challenge_connection(
    connection: &Connection,
    challenge_id: &[u8; 32],
) -> Result<Option<DiscoveryEndpointQuarantineChallenge>, DiscoveryEndpointQuarantineObservationError>
{
    let raw = connection
        .query_row(
            "SELECT challenge_id,admission_commitment,request_commitment,policy_version,issued_at,expires_at
             FROM discovery_endpoint_quarantine_challenge_v1 WHERE challenge_id=?1",
            params![&challenge_id[..]],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        )
        .optional()
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    raw.map(|(id, admission, request, policy, issued, expires)| {
        Ok(DiscoveryEndpointQuarantineChallenge {
            challenge_id: array32(id)?,
            admission_commitment: array32(admission)?,
            request_commitment: array32(request)?,
            policy_version: as_u64(policy)?,
            issued_at: as_u64(issued)?,
            expires_at: as_u64(expires)?,
        })
    })
    .transpose()
}

fn load_observation_commitment(
    tx: &Transaction<'_>,
    attempt_id: &[u8; 32],
) -> Result<Option<[u8; 32]>, DiscoveryEndpointQuarantineObservationError> {
    tx.query_row(
        "SELECT observation_commitment FROM discovery_endpoint_quarantine_observation_v1 WHERE attempt_id=?1",
        params![&attempt_id[..]],
        |row| row.get::<_, Vec<u8>>(0),
    )
    .optional()
    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?
    .map(array32)
    .transpose()
}

fn challenge_observation_count(
    tx: &Transaction<'_>,
    challenge_id: &[u8; 32],
) -> Result<usize, DiscoveryEndpointQuarantineObservationError> {
    let value: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_quarantine_observation_v1 WHERE challenge_id=?1",
            params![&challenge_id[..]],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    usize::try_from(value).map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)
}

fn evidence_state(
    tx: &Transaction<'_>,
    challenge_id: &[u8; 32],
    minimum_span: u64,
) -> Result<DiscoveryEndpointQuarantineEvidenceState, DiscoveryEndpointQuarantineObservationError> {
    evidence_state_query(tx, challenge_id, minimum_span)
}

fn evidence_state_connection(
    connection: &Connection,
    challenge_id: &[u8; 32],
    minimum_span: u64,
) -> Result<DiscoveryEndpointQuarantineEvidenceState, DiscoveryEndpointQuarantineObservationError> {
    evidence_state_query(connection, challenge_id, minimum_span)
}

fn evidence_state_query(
    connection: &Connection,
    challenge_id: &[u8; 32],
    minimum_span: u64,
) -> Result<DiscoveryEndpointQuarantineEvidenceState, DiscoveryEndpointQuarantineObservationError> {
    let (observers, outbound, inbound, earliest, latest): (
        i64,
        i64,
        i64,
        Option<i64>,
        Option<i64>,
    ) = connection
        .query_row(
            "SELECT COUNT(DISTINCT observer_context),
                    COALESCE(MAX(direction=1),0),COALESCE(MAX(direction=2),0),
                    MIN(observed_at),MAX(observed_at)
             FROM discovery_endpoint_quarantine_observation_v1 WHERE challenge_id=?1",
            params![&challenge_id[..]],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    let span_satisfied = match (earliest, latest) {
        (Some(earliest), Some(latest)) => {
            let earliest = as_u64(earliest)?;
            let latest = as_u64(latest)?;
            latest.saturating_sub(earliest) >= minimum_span
        }
        _ => false,
    };
    if observers >= 2 && outbound == 1 && inbound == 1 && span_satisfied {
        Ok(DiscoveryEndpointQuarantineEvidenceState::EvidenceSatisfied)
    } else {
        Ok(DiscoveryEndpointQuarantineEvidenceState::Pending)
    }
}

fn cleanup_tx(
    tx: &Transaction<'_>,
    now: u64,
    limit: usize,
) -> Result<usize, DiscoveryEndpointQuarantineObservationError> {
    let limit =
        i64::try_from(limit).map_err(|_| DiscoveryEndpointQuarantineObservationError::Rejected)?;
    let observations: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_quarantine_observation_v1
             WHERE challenge_id IN (
               SELECT challenge_id FROM discovery_endpoint_quarantine_challenge_v1
               WHERE expires_at<?1 ORDER BY expires_at,challenge_id LIMIT ?2
             )",
            params![as_i64(now)?, limit],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    let removed = tx
        .execute(
            "DELETE FROM discovery_endpoint_quarantine_challenge_v1
             WHERE challenge_id IN (
               SELECT challenge_id FROM discovery_endpoint_quarantine_challenge_v1
               WHERE expires_at<?1 ORDER BY expires_at,challenge_id LIMIT ?2
             )",
            params![as_i64(now)?, limit],
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    if removed > 0 {
        let observations = usize::try_from(observations)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?;
        let (challenges, retained_observations) = load_meta(tx)?;
        update_meta(
            tx,
            challenges
                .checked_sub(removed)
                .ok_or(DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            retained_observations
                .checked_sub(observations)
                .ok_or(DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        )?;
    }
    Ok(removed)
}

fn load_meta(
    tx: &Transaction<'_>,
) -> Result<(usize, usize), DiscoveryEndpointQuarantineObservationError> {
    let (challenges, observations): (i64, i64) = tx
        .query_row(
            "SELECT challenges,observations FROM discovery_endpoint_quarantine_observation_meta_v1 WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?;
    Ok((
        usize::try_from(challenges)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        usize::try_from(observations)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
    ))
}

fn load_meta_connection(
    connection: &Connection,
) -> Result<(usize, usize), DiscoveryEndpointQuarantineObservationError> {
    let (challenges, observations): (i64, i64) = connection
        .query_row(
            "SELECT challenges,observations FROM discovery_endpoint_quarantine_observation_meta_v1 WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?;
    Ok((
        usize::try_from(challenges)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
        usize::try_from(observations)
            .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
    ))
}

fn update_meta(
    tx: &Transaction<'_>,
    challenges: usize,
    observations: usize,
) -> Result<(), DiscoveryEndpointQuarantineObservationError> {
    tx.execute(
        "UPDATE discovery_endpoint_quarantine_observation_meta_v1 SET challenges=?1,observations=?2 WHERE singleton=1",
        params![
            i64::try_from(challenges).map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?,
            i64::try_from(observations).map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)?
        ],
    )
    .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    Ok(())
}

fn count_connection(
    connection: &Connection,
    query: &str,
) -> Result<usize, DiscoveryEndpointQuarantineObservationError> {
    let count: i64 = connection
        .query_row(query, [], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    usize::try_from(count).map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)
}

fn finish(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointQuarantineChallengeOutcome,
) -> Result<DiscoveryEndpointQuarantineChallengeOutcome, DiscoveryEndpointQuarantineObservationError>
{
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    Ok(outcome)
}

fn finish_observation(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointQuarantineObservationOutcome,
) -> Result<
    DiscoveryEndpointQuarantineObservationOutcome,
    DiscoveryEndpointQuarantineObservationError,
> {
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Unavailable)?;
    Ok(outcome)
}

fn challenge_id(admission: &[u8; 32], nonce: &[u8; 32], context: &[u8; 32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(CHALLENGE_ID_DOMAIN);
    hash.update(admission);
    hash.update(nonce);
    hash.update(context);
    hash.finalize().into()
}

#[allow(clippy::too_many_arguments)]
fn challenge_request_commitment(
    challenge_id: &[u8; 32],
    admission: &[u8; 32],
    policy_version: u64,
    nonce: &[u8; 32],
    context: &[u8; 32],
    issued_at: u64,
    expires_at: u64,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(CHALLENGE_REQUEST_DOMAIN);
    hash.update(challenge_id);
    hash.update(admission);
    hash.update(policy_version.to_be_bytes());
    hash.update(nonce);
    hash.update(context);
    hash.update(issued_at.to_be_bytes());
    hash.update(expires_at.to_be_bytes());
    hash.finalize().into()
}

#[allow(clippy::too_many_arguments)]
fn observation_commitment(
    challenge_id: &[u8; 32],
    admission: &[u8; 32],
    attempt_id: &[u8; 32],
    observer: &[u8; 32],
    direction: DiscoveryEndpointObservationDirection,
    evidence: &[u8; 32],
    observed_at: u64,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(OBSERVATION_DOMAIN);
    hash.update(challenge_id);
    hash.update(admission);
    hash.update(attempt_id);
    hash.update(observer);
    hash.update([direction as u8]);
    hash.update(evidence);
    hash.update(observed_at.to_be_bytes());
    hash.finalize().into()
}

pub(crate) fn satisfied_quarantine_evidence_commitment(
    admission: &[u8; 32],
    challenge_id: &[u8; 32],
    policy_epoch: u64,
    valid_until: u64,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(SATISFIED_EVIDENCE_DOMAIN);
    hash.update(admission);
    hash.update(challenge_id);
    hash.update(policy_epoch.to_be_bytes());
    hash.update(valid_until.to_be_bytes());
    hash.finalize().into()
}

fn array32(value: Vec<u8>) -> Result<[u8; 32], DiscoveryEndpointQuarantineObservationError> {
    value
        .try_into()
        .map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)
}

fn as_i64(value: u64) -> Result<i64, DiscoveryEndpointQuarantineObservationError> {
    i64::try_from(value).map_err(|_| DiscoveryEndpointQuarantineObservationError::Rejected)
}

fn as_u64(value: i64) -> Result<u64, DiscoveryEndpointQuarantineObservationError> {
    u64::try_from(value).map_err(|_| DiscoveryEndpointQuarantineObservationError::Corrupt)
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
        quarantine_admission_commitment, DiscoveryEndpointQuarantineConfig,
        SqliteDiscoveryEndpointQuarantineRegistry,
    };
    use tempfile::TempDir;

    const NOW: u64 = 2_000_000_000;

    struct AcceptEvidence;

    impl DiscoveryEndpointQuarantineEvidenceVerifier for AcceptEvidence {
        fn verify(
            &self,
            _request: &DiscoveryEndpointQuarantineEvidenceRequest,
        ) -> Result<(), DiscoveryEndpointQuarantineEvidenceVerificationError> {
            Ok(())
        }
    }

    struct RejectEvidence(DiscoveryEndpointQuarantineEvidenceVerificationError);

    impl DiscoveryEndpointQuarantineEvidenceVerifier for RejectEvidence {
        fn verify(
            &self,
            _request: &DiscoveryEndpointQuarantineEvidenceRequest,
        ) -> Result<(), DiscoveryEndpointQuarantineEvidenceVerificationError> {
            Err(self.0)
        }
    }

    fn tempdir() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("external test root");
        TempDir::new_in("target/test-temp").expect("tempdir")
    }

    fn config(dir: &TempDir, maximum: usize) -> DiscoveryEndpointQuarantineObservationConfig {
        DiscoveryEndpointQuarantineObservationConfig {
            db_path: dir.path().join("endpoint-observation.sqlite3"),
            max_challenges: maximum,
            max_attempts_per_challenge: 4,
            challenge_ttl_secs: 60,
            minimum_observation_span_secs: 5,
            cleanup_batch_size: 8,
        }
    }

    fn fresh_admission(
        directory: &TempDir,
        group_seed: u8,
        valid_until: u64,
    ) -> (
        SqliteDiscoveryEndpointQuarantineRegistry,
        DiscoveryEndpointFreshQuarantineAdmission,
    ) {
        let facts = DiscoveryEndpointCandidateFacts {
            group_commitment: [group_seed; 32],
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
                1,
                DiscoveryEndpointStakePolicyMode::Disabled,
            )
            .expect("policy"),
            NOW,
            None,
        )
        .into_quarantine_admission()
        .expect("admission");
        let exact_commitment = quarantine_admission_commitment(&admission);
        let registry =
            SqliteDiscoveryEndpointQuarantineRegistry::open(DiscoveryEndpointQuarantineConfig {
                db_path: directory.path().join("endpoint-quarantine.sqlite3"),
                max_entries: 8,
                cleanup_batch_size: 8,
            })
            .expect("quarantine");
        registry.record_at(admission, NOW).expect("record");
        let fresh = registry
            .fresh_admission_at(exact_commitment, NOW)
            .expect("lookup")
            .expect("fresh");
        (registry, fresh)
    }

    fn challenge(
        registry: &SqliteDiscoveryEndpointQuarantineObservationRegistry,
        admission: DiscoveryEndpointFreshQuarantineAdmission,
        nonce: u8,
        issued_at: u64,
        now: u64,
    ) -> DiscoveryEndpointQuarantineChallenge {
        match registry
            .begin_at(admission, [nonce; 32], [0x81; 32], issued_at, now)
            .expect("begin")
        {
            DiscoveryEndpointQuarantineChallengeOutcome::Issued(value)
            | DiscoveryEndpointQuarantineChallengeOutcome::Existing(value) => value,
            other => panic!("unexpected challenge result {other:?}"),
        }
    }

    #[test]
    fn challenge_exact_replay_conflict_expiry_and_restart_are_deterministic() {
        let directory = tempdir();
        let (_quarantine, admission) = fresh_admission(&directory, 0x31, NOW + 120);
        let cfg = config(&directory, 2);
        let registry =
            SqliteDiscoveryEndpointQuarantineObservationRegistry::open(cfg.clone()).expect("open");
        let first = challenge(&registry, admission, 0x41, NOW, NOW);
        assert!(matches!(
            registry
                .begin_at(admission, [0x41; 32], [0x81; 32], NOW, NOW + 1)
                .expect("replay"),
            DiscoveryEndpointQuarantineChallengeOutcome::Existing(value) if value == first
        ));
        assert_eq!(
            registry
                .begin_at(admission, [0x41; 32], [0x81; 32], NOW + 1, NOW + 1)
                .expect("conflict"),
            DiscoveryEndpointQuarantineChallengeOutcome::Conflict
        );
        drop(registry);
        let reopened =
            SqliteDiscoveryEndpointQuarantineObservationRegistry::open(cfg).expect("restart");
        assert!(matches!(
            reopened
                .begin_at(admission, [0x41; 32], [0x81; 32], NOW, NOW + 61)
                .expect("expired replay"),
            DiscoveryEndpointQuarantineChallengeOutcome::Expired
        ));
    }

    #[test]
    fn evidence_requires_verifier_two_directions_distinct_contexts_and_time() {
        let directory = tempdir();
        let (quarantine, admission) = fresh_admission(&directory, 0x32, NOW + 120);
        let registry =
            SqliteDiscoveryEndpointQuarantineObservationRegistry::open(config(&directory, 2))
                .expect("open");
        let challenge = challenge(&registry, admission, 0x42, NOW, NOW);
        let debug_request = DiscoveryEndpointQuarantineEvidenceRequest {
            challenge_id: [0x41; 32],
            admission_commitment: [0x42; 32],
            attempt_id: [0x43; 32],
            observer_context: [0x44; 32],
            direction: DiscoveryEndpointObservationDirection::OutboundChallenge,
            evidence_commitment: [0x45; 32],
            observed_at: NOW + 1,
        };
        let debug = format!("{registry:?}{challenge:?}{debug_request:?}");
        for secret in [0x41, 0x42, 0x43, 0x44, 0x45] {
            assert!(!debug.contains(&hex::encode([secret; 32])));
        }
        assert!(!debug.contains("endpoint-observation.sqlite3"));
        assert_eq!(
            registry
                .observe_at(
                    challenge,
                    [0x51; 32],
                    [0x61; 32],
                    DiscoveryEndpointObservationDirection::OutboundChallenge,
                    [0x71; 32],
                    NOW + 1,
                    NOW + 1,
                    None,
                )
                .expect("missing verifier"),
            DiscoveryEndpointQuarantineObservationOutcome::EvidenceMissing
        );
        assert_eq!(
            registry
                .observe_at(
                    challenge,
                    [0x51; 32],
                    [0x61; 32],
                    DiscoveryEndpointObservationDirection::OutboundChallenge,
                    [0x71; 32],
                    NOW + 1,
                    NOW + 1,
                    Some(&RejectEvidence(
                        DiscoveryEndpointQuarantineEvidenceVerificationError::Invalid,
                    )),
                )
                .expect("invalid evidence"),
            DiscoveryEndpointQuarantineObservationOutcome::EvidenceRejected
        );
        assert_eq!(
            registry
                .observe_at(
                    challenge,
                    [0x51; 32],
                    [0x61; 32],
                    DiscoveryEndpointObservationDirection::OutboundChallenge,
                    [0x71; 32],
                    NOW + 1,
                    NOW + 1,
                    Some(&RejectEvidence(
                        DiscoveryEndpointQuarantineEvidenceVerificationError::Unavailable,
                    )),
                )
                .expect("unavailable evidence"),
            DiscoveryEndpointQuarantineObservationOutcome::EvidenceUnavailable
        );
        assert_eq!(
            registry
                .observe_at(
                    challenge,
                    [0x51; 32],
                    [0x61; 32],
                    DiscoveryEndpointObservationDirection::OutboundChallenge,
                    [0x71; 32],
                    NOW + 1,
                    NOW + 1,
                    Some(&AcceptEvidence),
                )
                .expect("first"),
            DiscoveryEndpointQuarantineObservationOutcome::Recorded(
                DiscoveryEndpointQuarantineEvidenceState::Pending
            )
        );
        assert!(matches!(
            registry
                .observe_at(
                    challenge,
                    [0x51; 32],
                    [0x61; 32],
                    DiscoveryEndpointObservationDirection::OutboundChallenge,
                    [0x71; 32],
                    NOW + 1,
                    NOW + 2,
                    None,
                )
                .expect("exact replay"),
            DiscoveryEndpointQuarantineObservationOutcome::Existing(
                DiscoveryEndpointQuarantineEvidenceState::Pending
            )
        ));
        assert_eq!(
            registry
                .observe_at(
                    challenge,
                    [0x51; 32],
                    [0x61; 32],
                    DiscoveryEndpointObservationDirection::OutboundChallenge,
                    [0x72; 32],
                    NOW + 1,
                    NOW + 2,
                    Some(&AcceptEvidence),
                )
                .expect("conflict"),
            DiscoveryEndpointQuarantineObservationOutcome::Conflict
        );
        assert_eq!(
            registry
                .observe_at(
                    challenge,
                    [0x52; 32],
                    [0x62; 32],
                    DiscoveryEndpointObservationDirection::InboundProof,
                    [0x72; 32],
                    NOW + 6,
                    NOW + 6,
                    Some(&AcceptEvidence),
                )
                .expect("second"),
            DiscoveryEndpointQuarantineObservationOutcome::Recorded(
                DiscoveryEndpointQuarantineEvidenceState::EvidenceSatisfied
            )
        );
        assert_eq!(
            registry.snapshot_at(NOW + 6).expect("snapshot"),
            DiscoveryEndpointQuarantineObservationSnapshot {
                retained_challenges: 1,
                retained_observations: 2,
                fresh_challenges: 1,
                evidence_satisfied_challenges: 1,
            }
        );
        assert_eq!(
            quarantine.snapshot_at(NOW + 6).expect("quarantine"),
            super::super::discovery_endpoint_quarantine::DiscoveryEndpointQuarantineSnapshot {
                retained_candidates: 1,
                fresh_candidates: 1,
                maximum_valid_until: Some(NOW + 120),
            }
        );
    }

    #[test]
    fn missing_stake_expired_admission_capacity_cleanup_and_corruption_fail_closed() {
        let directory = tempdir();
        let facts = DiscoveryEndpointCandidateFacts {
            group_commitment: [0x33; 32],
            descriptor_sequence: 8,
            distinct_observers: 2,
            overlap_started_at: NOW,
            overlap_expires_at: NOW + 120,
            newest_observed_at: NOW,
            newest_expires_at: NOW + 120,
        };
        assert!(evaluate_endpoint_candidate(
            &facts,
            DiscoveryEndpointEligibilityPolicy::new(
                2,
                60,
                1,
                DiscoveryEndpointStakePolicyMode::Required,
            )
            .expect("policy"),
            NOW,
            None,
        )
        .into_quarantine_admission()
        .is_none());

        let (stale_quarantine, stale_admission) = fresh_admission(&directory, 0x35, NOW + 1);
        let stale_registry = SqliteDiscoveryEndpointQuarantineObservationRegistry::open(
            DiscoveryEndpointQuarantineObservationConfig {
                db_path: directory.path().join("stale-observation.sqlite3"),
                ..config(&directory, 1)
            },
        )
        .expect("stale registry");
        assert_eq!(
            stale_registry
                .begin_at(stale_admission, [0x45; 32], [0x82; 32], NOW, NOW + 2,)
                .expect("stale admission"),
            DiscoveryEndpointQuarantineChallengeOutcome::Expired
        );
        assert_eq!(
            stale_quarantine
                .snapshot_at(NOW + 2)
                .expect("stale quarantine")
                .fresh_candidates,
            0
        );

        let (_quarantine, admission) = fresh_admission(&directory, 0x34, NOW + 120);
        let cfg = config(&directory, 1);
        let registry =
            SqliteDiscoveryEndpointQuarantineObservationRegistry::open(cfg.clone()).expect("open");
        challenge(&registry, admission, 0x43, NOW, NOW);
        assert_eq!(
            registry
                .begin_at(admission, [0x44; 32], [0x81; 32], NOW, NOW)
                .expect("capacity"),
            DiscoveryEndpointQuarantineChallengeOutcome::AtCapacity
        );
        assert_eq!(
            registry.cleanup_expired_at(NOW + 61, 1).expect("cleanup"),
            1
        );
        assert_eq!(
            registry.snapshot_at(NOW + 61).expect("empty"),
            DiscoveryEndpointQuarantineObservationSnapshot {
                retained_challenges: 0,
                retained_observations: 0,
                fresh_challenges: 0,
                evidence_satisfied_challenges: 0,
            }
        );
        drop(registry);
        rusqlite::Connection::open(&cfg.db_path)
            .expect("database")
            .execute(
                "UPDATE discovery_endpoint_quarantine_observation_meta_v1 SET challenges=1",
                [],
            )
            .expect("tamper");
        assert!(matches!(
            SqliteDiscoveryEndpointQuarantineObservationRegistry::open(cfg),
            Err(DiscoveryEndpointQuarantineObservationError::Corrupt)
        ));
    }

    #[test]
    fn source_boundary_has_no_routeability_or_automatic_promotion_symbols() {
        let source = include_str!("discovery_endpoint_quarantine_observation.rs");
        for forbidden in [
            concat!("Peer", "Store"),
            concat!("upsert_", "verified"),
            concat!("route_", "candidates"),
            concat!("services::", "routing"),
            concat!("promote_", "candidate"),
        ] {
            assert!(
                !source.contains(forbidden),
                "forbidden routeability symbol: {forbidden}"
            );
        }
    }
}
