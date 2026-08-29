// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/attempt_journal.rs
// ============================================
//! Identity-sealed private continuation state for ambiguous replica attempts.
//!
//! ## Creation Reason
//! Replacement and provisioning may create terminal credentials, one-time
//! reply sessions, or request material before a workflow snapshot is durable.
//! A restart must recover that exact private attempt instead of regenerating
//! credentials or blindly repeating a storage side effect.
//!
//! ## Main Functionality
//! - Seals bounded opaque adapter state before the dispatch transition.
//! - Binds it to work identity, immutable action, attempt, and evidence window.
//! - Applies an independent monotonic sequence and bounded retention window.
//! - Opens state only against the exact authenticated restored execution.
//! - Redacts Debug output and zeroizes private state when ownership ends.
//!
//! ## Dependencies
//! - `execution.rs`: side-effect-free dispatch readiness and exact attempt.
//! - `sealed_local.rs`: identity-bound XChaCha20-Poly1305 container.
//! - `snapshot.rs`: authenticated workflow state used after restart.
//! - `protocol::blind_vault`: immutable replacement/provisioning actions.
//!
//! ## Main Logical Flow
//! 1. Build the exact terminal request/session state without sending it.
//! 2. Seal and atomically persist this journal using a new sequence.
//! 3. Dispatch with the same source timestamp, then persist the workflow.
//! 4. Only after both local records are durable, send the prepared request.
//! 5. After restart, open the journal against the restored in-flight item.
//!
//! ## Important Note For The Next Developer
//! - This is private local persistence, never a wire, ledger, or API payload.
//! - The opaque state must be the exact prepared attempt, not a reconstruction.
//! - Keep the accepted journal sequence in separately protected high-water state.
//! - Never log, clone unnecessarily, or expose the private continuation bytes.
//! - Delete a journal only after accepted evidence or explicit safe resolution.
//!
//! Last Modified: v1.3.0-PreparedRecoveryAuthentication - Added an
//! identity-authenticated proof that a durable journal stopped before send.
//! v1.2.0-TypedContinuation - Added errors shared by the
//! recoverable onion reply continuation codec.
//! v1.1.0-PreparedAttempt - Added a typed
//! persist-before-dispatch handle.
//! v1.0.0-PrivateAttemptJournal - Initial fail-closed format.
//! ============================================

use std::{fmt, mem};

use bincode::Options;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::Zeroize;

use super::sealed_local::{open_identity_bound, seal_identity_bound, IdentitySealedLocalError};
use super::{
    persistence::sealed_record_commitment, require_timestamp, BlindVaultReplicaDispatchReadiness,
    BlindVaultReplicaExecution, BlindVaultReplicaRecoveryStore, BlindVaultReplicaRestoredExecution,
    BlindVaultReplicaWorkId, BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError,
};
use crate::crypto::keys::IdentityKeyPair;
use crate::protocol::blind_vault::BlindVaultReplicaAction;

const ATTEMPT_JOURNAL_MAGIC: [u8; 4] = *b"AXRJ";
const ATTEMPT_JOURNAL_VERSION_V1: u16 = 1;
const ATTEMPT_JOURNAL_KEY_SALT: &[u8] = b"AeroNyx-BlindVault-Attempt-Journal-Key-v1";
const ATTEMPT_JOURNAL_KEY_INFO: &[u8] = b"AeroNyx-BlindVault-Attempt-Journal-State-v1";
const ACTION_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Replica-Action-v1";

/// Maximum opaque request/session/credential state retained for one attempt.
pub const MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES: usize = 32 * 1024;

/// Maximum complete identity-sealed private attempt journal container.
pub const MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES: usize = 64 * 1024;

/// Maximum time private attempt material may remain recoverable after dispatch.
pub const MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_RETENTION_MS: u64 = 7 * 24 * 60 * 60 * 1_000;

#[derive(Serialize, Deserialize)]
struct AttemptJournalBodyV1 {
    journal_sequence: u64,
    workflow_id: [u8; 16],
    work_sequence: u16,
    attempt: u8,
    dispatched_at_ms: u64,
    evidence_deadline_ms: u64,
    retain_until_ms: u64,
    action_commitment: [u8; 32],
    private_state: Vec<u8>,
}

impl Drop for AttemptJournalBodyV1 {
    fn drop(&mut self) {
        self.private_state.zeroize();
    }
}

/// Authenticated private continuation state recovered for one exact attempt.
///
/// Debug output is deliberately redacted. Ownership should remain within the
/// source-side persistence/orchestration boundary and never enter telemetry.
pub struct BlindVaultReplicaAttemptJournal {
    journal_sequence: u64,
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    dispatched_at_ms: u64,
    evidence_deadline_ms: u64,
    retain_until_ms: u64,
    private_state: Vec<u8>,
}

/// Proof that one authenticated durable journal never crossed dispatch commit.
///
/// This value contains no private continuation bytes. It can only be created
/// by opening the identity-sealed journal against an authenticated restored
/// workflow whose exact work item is still dispatch-ready at the bound time.
pub struct BlindVaultReplicaAuthenticatedPreparedAttempt {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    journal_sequence: u64,
    journal_commitment: [u8; 32],
}

/// Prepared journal plus immutable binding required for one dispatch commit.
///
/// [BLIND-VAULT-PREPARED-ATTEMPT 2026-08-29 by Codex] This value closes the
/// ordering gap between sealing bytes and mutating the workflow. The caller
/// persists `sealed_journal()` first, then passes the same handle to
/// `commit_prepared_attempt_dispatch`. Debug output never includes bytes,
/// work identity, target identity, lease identity, or action commitment.
pub struct BlindVaultReplicaPreparedAttemptJournal {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    planned_dispatch_at_ms: u64,
    evidence_deadline_ms: u64,
    retain_until_ms: u64,
    journal_sequence: u64,
    action_commitment: [u8; 32],
    sealed_journal: Vec<u8>,
}

impl BlindVaultReplicaAttemptJournal {
    /// Authenticated sequence used by the separate rollback high-water mark.
    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }

    /// Exact restored work item to which the private state is bound.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    /// Exact bounded attempt to which the private state is bound.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    /// Source timestamp used for the original dispatch transition.
    #[must_use]
    pub const fn dispatched_at_ms(&self) -> u64 {
        self.dispatched_at_ms
    }

    /// Original evidence acceptance deadline.
    #[must_use]
    pub const fn evidence_deadline_ms(&self) -> u64 {
        self.evidence_deadline_ms
    }

    /// Last source timestamp at which this journal may be opened.
    #[must_use]
    pub const fn retain_until_ms(&self) -> u64 {
        self.retain_until_ms
    }

    /// Borrows the exact opaque adapter state without copying it.
    #[must_use]
    pub fn private_state(&self) -> &[u8] {
        &self.private_state
    }

    /// Transfers the exact opaque adapter state to the recovery owner.
    #[must_use]
    pub fn into_private_state(mut self) -> Vec<u8> {
        mem::take(&mut self.private_state)
    }
}

impl Drop for BlindVaultReplicaAttemptJournal {
    fn drop(&mut self) {
        self.private_state.zeroize();
    }
}

impl BlindVaultReplicaAuthenticatedPreparedAttempt {
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }

    /// Atomically removes only this authenticated pre-dispatch journal.
    pub fn abort_recovery<Store>(&self, store: &mut Store) -> Result<(), Store::Error>
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        store.abort_prepared_attempt(self.journal_sequence, self.journal_commitment)
    }
}

impl fmt::Debug for BlindVaultReplicaAuthenticatedPreparedAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaAuthenticatedPreparedAttempt")
            .field("attempt", &self.attempt)
            .field("journal_sequence", &self.journal_sequence)
            .field("journal_commitment", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl BlindVaultReplicaPreparedAttemptJournal {
    /// Exact work item whose private continuation state was sealed.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    /// Exact attempt predicted by side-effect-free readiness.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    /// Timestamp that must be used for the workflow dispatch transition.
    #[must_use]
    pub const fn planned_dispatch_at_ms(&self) -> u64 {
        self.planned_dispatch_at_ms
    }

    /// Evidence deadline bound into both journal and workflow transition.
    #[must_use]
    pub const fn evidence_deadline_ms(&self) -> u64 {
        self.evidence_deadline_ms
    }

    /// Last source timestamp at which recovery may open the journal.
    #[must_use]
    pub const fn retain_until_ms(&self) -> u64 {
        self.retain_until_ms
    }

    /// Monotonic sequence that must advance before dispatch is committed.
    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }

    /// Borrows authenticated ciphertext for atomic restrictive persistence.
    #[must_use]
    pub fn sealed_journal(&self) -> &[u8] {
        &self.sealed_journal
    }
}

impl Drop for BlindVaultReplicaPreparedAttemptJournal {
    fn drop(&mut self) {
        self.sealed_journal.zeroize();
        self.action_commitment.zeroize();
    }
}

impl fmt::Debug for BlindVaultReplicaPreparedAttemptJournal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaPreparedAttemptJournal")
            .field("journal_sequence", &self.journal_sequence)
            .field("attempt", &self.attempt)
            .field("sealed_journal", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaAttemptJournal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaAttemptJournal")
            .field("journal_sequence", &self.journal_sequence)
            .field("attempt", &self.attempt)
            .field("private_state", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Fail-closed local journal construction and restoration errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaAttemptJournalError {
    /// The workflow rejected the supplied work identity, time, or readiness.
    #[error(transparent)]
    Workflow(#[from] BlindVaultReplicaWorkflowError),
    /// Only replacement and provisioning require private attempt recovery.
    #[error("blind vault replica action does not require a private attempt journal")]
    JournalNotRequired,
    /// Dispatch readiness was not the exact ready state required for sealing.
    #[error("blind vault replica attempt is not ready for journal preparation")]
    DispatchNotReady,
    /// Opaque adapter state was empty.
    #[error("blind vault replica attempt private state is empty")]
    PrivateStateEmpty,
    /// Opaque state or its encrypted container exceeded the fixed bound.
    #[error("blind vault replica attempt journal is too large")]
    TooLarge,
    /// Sequence zero cannot participate in rollback protection.
    #[error("blind vault replica attempt journal sequence is invalid")]
    SequenceInvalid,
    /// Retention did not cover evidence recovery or exceeded the fixed bound.
    #[error("blind vault replica attempt journal lifetime is invalid")]
    LifetimeInvalid,
    /// The authenticated private continuation state is no longer retainable.
    #[error("blind vault replica attempt journal has expired")]
    Expired,
    /// Bytes did not match the versioned local journal encoding.
    #[error("blind vault replica attempt journal is malformed")]
    Malformed,
    /// Bytes use an unsupported local journal version.
    #[error("blind vault replica attempt journal version is unsupported")]
    VersionUnsupported,
    /// Key derivation or AEAD authentication failed.
    #[error("blind vault replica attempt journal authentication failed")]
    AuthenticationFailed,
    /// An authenticated journal predates the protected high-water mark.
    #[error("blind vault replica attempt journal rollback was detected")]
    RollbackDetected,
    /// Journal binding did not match the authenticated restored execution.
    #[error("blind vault replica attempt journal does not match restored state")]
    StateMismatch,
    /// Typed continuation bytes were truncated or internally inconsistent.
    #[error("blind vault replica attempt continuation is malformed")]
    ContinuationMalformed,
    /// Typed continuation bytes use an unsupported local format version.
    #[error("blind vault replica attempt continuation version is unsupported")]
    ContinuationVersionUnsupported,
    /// Typed continuation recovery requires at least one reply session.
    #[error("blind vault replica attempt continuation requires a reply session")]
    ReplySessionsRequired,
    /// Typed continuation exceeded the fixed single-use reply-session count.
    #[error("blind vault replica attempt continuation has too many reply sessions")]
    TooManyReplySessions,
}

impl BlindVaultReplicaExecution {
    /// Prepares a persist-before-dispatch handle for one mutating attempt.
    ///
    /// Persist `sealed_journal()` and its independent sequence high-water mark
    /// atomically before calling `commit_prepared_attempt_dispatch`. Dropping
    /// this value before commit has no workflow or network side effect.
    pub fn prepare_attempt_journal_for_dispatch(
        &self,
        identity: &IdentityKeyPair,
        work_id: BlindVaultReplicaWorkId,
        planned_dispatch_at_ms: u64,
        journal_sequence: u64,
        retain_until_ms: u64,
        private_state: &[u8],
    ) -> Result<BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaAttemptJournalError> {
        let sealed_journal = self.seal_attempt_journal_for_dispatch(
            identity,
            work_id,
            planned_dispatch_at_ms,
            journal_sequence,
            retain_until_ms,
            private_state,
        )?;
        let BlindVaultReplicaDispatchReadiness::Ready {
            attempt,
            evidence_deadline_ms,
        } = self.dispatch_readiness(work_id, planned_dispatch_at_ms)?
        else {
            return Err(BlindVaultReplicaAttemptJournalError::DispatchNotReady);
        };
        let action_commitment = self
            .items()
            .iter()
            .find(|item| item.id() == work_id)
            .and_then(|item| journal_action_commitment(item.action()))
            .ok_or(BlindVaultReplicaAttemptJournalError::StateMismatch)?;
        Ok(BlindVaultReplicaPreparedAttemptJournal {
            work_id,
            attempt,
            planned_dispatch_at_ms,
            evidence_deadline_ms,
            retain_until_ms,
            journal_sequence,
            action_commitment,
            sealed_journal,
        })
    }

    /// Commits the in-memory dispatch transition for a persisted handle.
    ///
    /// After success, seal and atomically persist a fresh workflow snapshot;
    /// only then send the exact request represented by the private journal.
    /// Any changed readiness or action binding fails without state mutation.
    pub fn commit_prepared_attempt_dispatch(
        &mut self,
        prepared: &BlindVaultReplicaPreparedAttemptJournal,
    ) -> Result<u8, BlindVaultReplicaAttemptJournalError> {
        let current_commitment = self
            .items()
            .iter()
            .find(|item| item.id() == prepared.work_id)
            .and_then(|item| journal_action_commitment(item.action()))
            .ok_or(BlindVaultReplicaAttemptJournalError::StateMismatch)?;
        if current_commitment != prepared.action_commitment {
            return Err(BlindVaultReplicaAttemptJournalError::StateMismatch);
        }
        let readiness =
            self.dispatch_readiness(prepared.work_id, prepared.planned_dispatch_at_ms)?;
        if readiness
            != (BlindVaultReplicaDispatchReadiness::Ready {
                attempt: prepared.attempt,
                evidence_deadline_ms: prepared.evidence_deadline_ms,
            })
        {
            return Err(BlindVaultReplicaAttemptJournalError::DispatchNotReady);
        }
        self.dispatch(prepared.work_id, prepared.planned_dispatch_at_ms)
            .map_err(BlindVaultReplicaAttemptJournalError::from)
    }

    /// Seals exact private attempt material for the typed prepared handle.
    ///
    /// [BLIND-VAULT-PRIVATE-ATTEMPT-JOURNAL 2026-08-29 by Codex] Persist the
    /// This low-level helper remains private so callers cannot bypass the
    /// persist-before-dispatch ordering enforced by the public prepared type.
    fn seal_attempt_journal_for_dispatch(
        &self,
        identity: &IdentityKeyPair,
        work_id: BlindVaultReplicaWorkId,
        planned_dispatch_at_ms: u64,
        journal_sequence: u64,
        retain_until_ms: u64,
        private_state: &[u8],
    ) -> Result<Vec<u8>, BlindVaultReplicaAttemptJournalError> {
        require_timestamp(planned_dispatch_at_ms)?;
        if journal_sequence == 0 {
            return Err(BlindVaultReplicaAttemptJournalError::SequenceInvalid);
        }
        if private_state.is_empty() {
            return Err(BlindVaultReplicaAttemptJournalError::PrivateStateEmpty);
        }
        if private_state.len() > MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES {
            return Err(BlindVaultReplicaAttemptJournalError::TooLarge);
        }

        let readiness = self.dispatch_readiness(work_id, planned_dispatch_at_ms)?;
        let BlindVaultReplicaDispatchReadiness::Ready {
            attempt,
            evidence_deadline_ms,
        } = readiness
        else {
            return Err(BlindVaultReplicaAttemptJournalError::DispatchNotReady);
        };
        let item = self
            .items()
            .iter()
            .find(|item| item.id() == work_id)
            .ok_or(BlindVaultReplicaAttemptJournalError::StateMismatch)?;
        let action_commitment = journal_action_commitment(item.action())
            .ok_or(BlindVaultReplicaAttemptJournalError::JournalNotRequired)?;
        validate_journal_lifetime(
            planned_dispatch_at_ms,
            evidence_deadline_ms,
            retain_until_ms,
        )?;

        let body = AttemptJournalBodyV1 {
            journal_sequence,
            workflow_id: work_id.workflow_id(),
            work_sequence: work_id.sequence(),
            attempt,
            dispatched_at_ms: planned_dispatch_at_ms,
            evidence_deadline_ms,
            retain_until_ms,
            action_commitment,
            private_state: private_state.to_vec(),
        };
        let mut plaintext = journal_options()
            .serialize(&body)
            .map_err(|_| BlindVaultReplicaAttemptJournalError::TooLarge)?;
        let sealed = seal_identity_bound(
            identity,
            ATTEMPT_JOURNAL_MAGIC,
            ATTEMPT_JOURNAL_VERSION_V1,
            ATTEMPT_JOURNAL_KEY_SALT,
            ATTEMPT_JOURNAL_KEY_INFO,
            &plaintext,
            MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES,
        );
        plaintext.zeroize();
        sealed.map_err(map_sealed_local_error)
    }
}

impl BlindVaultReplicaRestoredExecution {
    /// Opens private state only for the exact authenticated in-flight attempt.
    ///
    /// `minimum_journal_sequence` is zero only before the first accepted
    /// journal. After success, persist `journal_sequence()` in independently
    /// protected monotonic state before acting on the returned private bytes.
    pub fn open_attempt_journal(
        &self,
        identity: &IdentityKeyPair,
        journal: &[u8],
        minimum_journal_sequence: u64,
        now_ms: u64,
    ) -> Result<BlindVaultReplicaAttemptJournal, BlindVaultReplicaAttemptJournalError> {
        let mut body =
            decode_attempt_journal_body(identity, journal, minimum_journal_sequence, now_ms)?;
        if now_ms > body.retain_until_ms {
            return Err(BlindVaultReplicaAttemptJournalError::Expired);
        }

        let work_id = BlindVaultReplicaWorkId {
            workflow_id: body.workflow_id,
            sequence: body.work_sequence,
        };
        let item = self
            .execution()
            .items()
            .iter()
            .find(|item| item.id() == work_id)
            .ok_or(BlindVaultReplicaAttemptJournalError::StateMismatch)?;
        let expected_commitment = journal_action_commitment(item.action())
            .ok_or(BlindVaultReplicaAttemptJournalError::StateMismatch)?;
        let expected_state = BlindVaultReplicaWorkState::AwaitingEvidence {
            attempt: body.attempt,
            dispatched_at_ms: body.dispatched_at_ms,
            evidence_deadline_ms: body.evidence_deadline_ms,
        };
        if item.state() != expected_state || body.action_commitment != expected_commitment {
            return Err(BlindVaultReplicaAttemptJournalError::StateMismatch);
        }

        Ok(BlindVaultReplicaAttemptJournal {
            journal_sequence: body.journal_sequence,
            work_id,
            attempt: body.attempt,
            dispatched_at_ms: body.dispatched_at_ms,
            evidence_deadline_ms: body.evidence_deadline_ms,
            retain_until_ms: body.retain_until_ms,
            private_state: mem::take(&mut body.private_state),
        })
    }

    /// Authenticates a durable journal against the pre-dispatch snapshot.
    ///
    /// [BLIND-VAULT-PREPARED-RECOVERY-AUTH 2026-08-29 by Codex] This path
    /// intentionally does not expose private continuation bytes and permits
    /// authentication after retention expiry. The typed ordering guarantees
    /// that a matching pre-dispatch snapshot means no request was sent.
    pub fn authenticate_prepared_attempt_journal(
        &self,
        identity: &IdentityKeyPair,
        journal: &[u8],
        minimum_journal_sequence: u64,
        now_ms: u64,
    ) -> Result<BlindVaultReplicaAuthenticatedPreparedAttempt, BlindVaultReplicaAttemptJournalError>
    {
        let body =
            decode_attempt_journal_body(identity, journal, minimum_journal_sequence, now_ms)?;
        let work_id = BlindVaultReplicaWorkId {
            workflow_id: body.workflow_id,
            sequence: body.work_sequence,
        };
        let item = self
            .execution()
            .items()
            .iter()
            .find(|item| item.id() == work_id)
            .ok_or(BlindVaultReplicaAttemptJournalError::StateMismatch)?;
        let expected_commitment = journal_action_commitment(item.action())
            .ok_or(BlindVaultReplicaAttemptJournalError::StateMismatch)?;
        let expected_readiness = BlindVaultReplicaDispatchReadiness::Ready {
            attempt: body.attempt,
            evidence_deadline_ms: body.evidence_deadline_ms,
        };
        if self
            .execution()
            .dispatch_readiness(work_id, body.dispatched_at_ms)?
            != expected_readiness
            || body.action_commitment != expected_commitment
        {
            return Err(BlindVaultReplicaAttemptJournalError::StateMismatch);
        }
        Ok(BlindVaultReplicaAuthenticatedPreparedAttempt {
            work_id,
            attempt: body.attempt,
            journal_sequence: body.journal_sequence,
            journal_commitment: sealed_record_commitment(journal),
        })
    }
}

fn decode_attempt_journal_body(
    identity: &IdentityKeyPair,
    journal: &[u8],
    minimum_journal_sequence: u64,
    now_ms: u64,
) -> Result<AttemptJournalBodyV1, BlindVaultReplicaAttemptJournalError> {
    require_timestamp(now_ms)?;
    let mut plaintext = open_identity_bound(
        identity,
        journal,
        ATTEMPT_JOURNAL_MAGIC,
        ATTEMPT_JOURNAL_VERSION_V1,
        ATTEMPT_JOURNAL_KEY_SALT,
        ATTEMPT_JOURNAL_KEY_INFO,
        MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES,
    )
    .map_err(map_sealed_local_error)?;
    let body = journal_options()
        .deserialize::<AttemptJournalBodyV1>(&plaintext)
        .map_err(|_| BlindVaultReplicaAttemptJournalError::Malformed);
    plaintext.zeroize();
    let body = body?;

    if body.journal_sequence == 0 {
        return Err(BlindVaultReplicaAttemptJournalError::SequenceInvalid);
    }
    if body.journal_sequence < minimum_journal_sequence {
        return Err(BlindVaultReplicaAttemptJournalError::RollbackDetected);
    }
    if body.private_state.is_empty() {
        return Err(BlindVaultReplicaAttemptJournalError::PrivateStateEmpty);
    }
    if body.private_state.len() > MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES {
        return Err(BlindVaultReplicaAttemptJournalError::TooLarge);
    }
    validate_journal_lifetime(
        body.dispatched_at_ms,
        body.evidence_deadline_ms,
        body.retain_until_ms,
    )?;
    if now_ms < body.dispatched_at_ms {
        return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange.into());
    }
    Ok(body)
}

fn validate_journal_lifetime(
    dispatched_at_ms: u64,
    evidence_deadline_ms: u64,
    retain_until_ms: u64,
) -> Result<(), BlindVaultReplicaAttemptJournalError> {
    let retention_ms = retain_until_ms
        .checked_sub(dispatched_at_ms)
        .ok_or(BlindVaultReplicaAttemptJournalError::LifetimeInvalid)?;
    if evidence_deadline_ms < dispatched_at_ms
        || retain_until_ms < evidence_deadline_ms
        || retention_ms > MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_RETENTION_MS
    {
        return Err(BlindVaultReplicaAttemptJournalError::LifetimeInvalid);
    }
    Ok(())
}

fn journal_action_commitment(action: BlindVaultReplicaAction) -> Option<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(ACTION_COMMITMENT_DOMAIN);
    match action {
        BlindVaultReplicaAction::ReplaceReplica { node_id, lease_id } => {
            // [BLIND-VAULT-JOURNAL-ACTION-COMMITMENT 2026-08-29 by Codex]
            // Explicit tags freeze this local commitment independently from
            // Rust enum layout and serializer implementation details.
            hasher.update([4u8]);
            hasher.update(node_id);
            hasher.update(lease_id);
        }
        BlindVaultReplicaAction::ProvisionReplicas { count } => {
            hasher.update([5u8]);
            hasher.update([count]);
        }
        BlindVaultReplicaAction::RenewLease { .. }
        | BlindVaultReplicaAction::ReconcileInventory { .. }
        | BlindVaultReplicaAction::RetryObservation { .. } => return None,
    }
    Some(hasher.finalize().into())
}

fn map_sealed_local_error(error: IdentitySealedLocalError) -> BlindVaultReplicaAttemptJournalError {
    match error {
        IdentitySealedLocalError::TooLarge => BlindVaultReplicaAttemptJournalError::TooLarge,
        IdentitySealedLocalError::Malformed => BlindVaultReplicaAttemptJournalError::Malformed,
        IdentitySealedLocalError::UnsupportedVersion => {
            BlindVaultReplicaAttemptJournalError::VersionUnsupported
        }
        IdentitySealedLocalError::AuthenticationFailed => {
            BlindVaultReplicaAttemptJournalError::AuthenticationFailed
        }
    }
}

fn journal_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES as u64)
        .reject_trailing_bytes()
}
