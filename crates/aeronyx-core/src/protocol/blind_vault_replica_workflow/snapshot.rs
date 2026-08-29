// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/snapshot.rs
// ============================================
//! Identity-sealed restart persistence for source-owned replica workflows.
//!
//! ## Creation Reason
//! A multi-request replica action may span an App restart. Repeating an
//! ambiguous attempt can duplicate storage work or retire the wrong lifecycle
//! generation, while plaintext workflow state leaks terminal/lease correlation.
//!
//! ## Main Functionality
//! - Encodes a version-frozen V1 body independent from public network wire.
//! - Derives a dedicated key from source identity with HKDF-SHA256.
//! - Seals state with XChaCha20-Poly1305 and authenticates the full header.
//! - Revalidates planner, timing, retry, dependency, and capacity invariants.
//!
//! ## Important Note For The Next Developer
//! - This format is local encrypted persistence, never discovery/ledger/API.
//! - Do not add ciphertext, object ids, manifests, contacts, routes, or URLs.
//! - Never mutate V1 enum layouts; add a new container/body version instead.
//! - Keep the accepted sequence high-water mark in separately protected state.
//! - Authentication failure and malformed state must remain fail-closed.
//!
//! Last Modified: v1.1.0-RollbackGuard - Added authenticated monotonic restore
//! sequencing and explicit sensitive-buffer cleanup on every handled path.
//! v1.0.0-SealedRestartSnapshot - Initial identity-bound V1.
//! ============================================

use bincode::Options;
use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hkdf::Hkdf;
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use zeroize::Zeroize;

use super::{
    BlindVaultReplicaDispatchFailure, BlindVaultReplicaExecution, BlindVaultReplicaExecutionPolicy,
    BlindVaultReplicaRestoredExecution, BlindVaultReplicaWorkId, BlindVaultReplicaWorkItem,
    BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError,
    MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES, MAX_BLIND_VAULT_REPLICA_WORK_ITEMS,
};
use crate::crypto::keys::IdentityKeyPair;
use crate::protocol::blind_vault::{
    BlindVaultReplicaAction, BlindVaultReplicaPlan, BlindVaultReplicaPlanHealth,
};

const RESTART_SNAPSHOT_MAGIC: [u8; 4] = *b"AXRS";
const RESTART_SNAPSHOT_VERSION_V1: u16 = 1;
const RESTART_SNAPSHOT_HEADER_BYTES: usize = 4 + 2 + 24;
const RESTART_SNAPSHOT_TAG_BYTES: usize = 16;
const RESTART_SNAPSHOT_KEY_SALT: &[u8] = b"AeroNyx-BlindVault-Replica-Restart-Key-v1";
const RESTART_SNAPSHOT_KEY_INFO: &[u8] = b"AeroNyx-BlindVault-Replica-Restart-State-v1";

#[derive(Debug, Serialize, Deserialize)]
struct RestartSnapshotBodyV1 {
    snapshot_sequence: u64,
    workflow_id: [u8; 16],
    created_at_ms: u64,
    #[serde(with = "PlanHealthV1")]
    source_plan_health: BlindVaultReplicaPlanHealth,
    source_configured_replicas: u8,
    source_live_verified_replicas: u8,
    source_live_matching_replicas: u8,
    maximum_attempts: u8,
    evidence_timeout_ms: u64,
    maximum_in_flight: u8,
    items: Vec<WorkItemV1>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkItemV1 {
    sequence: u16,
    #[serde(with = "ActionV1")]
    action: BlindVaultReplicaAction,
    #[serde(with = "WorkStateV1")]
    state: BlindVaultReplicaWorkState,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(remote = "BlindVaultReplicaPlanHealth")]
enum PlanHealthV1 {
    Healthy,
    MaintenanceDue,
    Degraded,
    QuorumUnavailable,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(remote = "BlindVaultReplicaAction")]
enum ActionV1 {
    RenewLease {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        expected_expires_at_ms: u64,
    },
    ReconcileInventory {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        expected_object_count: u64,
        observed_object_count: u64,
        expected_ciphertext_bytes: u64,
        observed_ciphertext_bytes: u64,
        expected_inventory_commitment: [u8; 32],
        observed_inventory_commitment: [u8; 32],
    },
    RetryObservation {
        node_id: [u8; 32],
        lease_id: [u8; 32],
    },
    ReplaceReplica {
        node_id: [u8; 32],
        lease_id: [u8; 32],
    },
    ProvisionReplicas {
        count: u8,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(remote = "BlindVaultReplicaDispatchFailure")]
enum DispatchFailureV1 {
    TransportUnavailable,
    TerminalUnavailable,
    TerminalRejected,
    EvidenceTimeout,
    StalePlan,
    CapacityUnavailable,
    PolicyRejected,
    LocalConstructionFailed,
    InlineResponseUnsupported,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(remote = "BlindVaultReplicaWorkState")]
enum WorkStateV1 {
    AwaitingAuthorization,
    Authorized {
        authorized_at_ms: u64,
    },
    AwaitingEvidence {
        attempt: u8,
        dispatched_at_ms: u64,
        evidence_deadline_ms: u64,
    },
    EvidenceAccepted {
        attempt: u8,
        verified_at_ms: u64,
    },
    RetryableFailure {
        attempt: u8,
        failed_at_ms: u64,
        retry_not_before_ms: u64,
        #[serde(with = "DispatchFailureV1")]
        failure: BlindVaultReplicaDispatchFailure,
    },
    PermanentFailure {
        attempt: u8,
        failed_at_ms: u64,
        #[serde(with = "DispatchFailureV1")]
        failure: BlindVaultReplicaDispatchFailure,
    },
    Exhausted {
        attempt: u8,
        failed_at_ms: u64,
        #[serde(with = "DispatchFailureV1")]
        failure: BlindVaultReplicaDispatchFailure,
    },
    Cancelled {
        cancelled_at_ms: u64,
    },
}

impl BlindVaultReplicaExecution {
    /// Seals source-local execution state for authenticated restart.
    ///
    /// [BLIND-VAULT-SEALED-RESTART-SNAPSHOT 2026-08-29 by Codex] The caller
    /// remains responsible for restrictive file permissions and atomic file
    /// replacement. `snapshot_sequence` must increase monotonically and its
    /// accepted high-water mark must be protected separately from this file.
    /// The sealed body contains no ciphertext or private manifest.
    pub fn seal_restart_snapshot(
        &self,
        identity: &IdentityKeyPair,
        snapshot_sequence: u64,
    ) -> Result<Vec<u8>, BlindVaultReplicaWorkflowError> {
        if snapshot_sequence == 0 {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotSequenceInvalid);
        }
        let body = RestartSnapshotBodyV1::from_execution(self, snapshot_sequence);
        let mut plaintext = snapshot_options()
            .serialize(&body)
            .map_err(|_| BlindVaultReplicaWorkflowError::RestartSnapshotMalformed)?;
        if plaintext
            .len()
            .saturating_add(RESTART_SNAPSHOT_HEADER_BYTES + RESTART_SNAPSHOT_TAG_BYTES)
            > MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES
        {
            plaintext.zeroize();
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotTooLarge);
        }

        let mut nonce = [0u8; 24];
        OsRng.fill_bytes(&mut nonce);
        let header = snapshot_header(nonce);
        let mut key = match derive_snapshot_key(identity) {
            Ok(key) => key,
            Err(error) => {
                plaintext.zeroize();
                return Err(error);
            }
        };
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
        let encrypted = cipher.encrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: &plaintext,
                aad: &header,
            },
        );
        plaintext.zeroize();
        key.zeroize();
        let ciphertext = encrypted
            .map_err(|_| BlindVaultReplicaWorkflowError::RestartSnapshotAuthenticationFailed)?;

        let mut snapshot = header;
        snapshot.extend_from_slice(&ciphertext);
        Ok(snapshot)
    }

    /// Opens and revalidates one source-local restart snapshot.
    ///
    /// `restored_at_ms` is the source clock at load time. Historical event
    /// timestamps may not be in its future; retry and evidence deadlines may.
    /// Pass zero as `minimum_snapshot_sequence` only before the first accepted
    /// snapshot, then persist the returned sequence in secure monotonic state.
    pub fn open_restart_snapshot(
        identity: &IdentityKeyPair,
        snapshot: &[u8],
        minimum_snapshot_sequence: u64,
        restored_at_ms: u64,
    ) -> Result<BlindVaultReplicaRestoredExecution, BlindVaultReplicaWorkflowError> {
        if snapshot.len() > MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotTooLarge);
        }
        if snapshot.len() < RESTART_SNAPSHOT_HEADER_BYTES + RESTART_SNAPSHOT_TAG_BYTES
            || snapshot[..4] != RESTART_SNAPSHOT_MAGIC
        {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotMalformed);
        }
        let version = u16::from_be_bytes([snapshot[4], snapshot[5]]);
        if version != RESTART_SNAPSHOT_VERSION_V1 {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotVersionUnsupported);
        }
        let mut nonce = [0u8; 24];
        nonce.copy_from_slice(&snapshot[6..RESTART_SNAPSHOT_HEADER_BYTES]);
        let header = &snapshot[..RESTART_SNAPSHOT_HEADER_BYTES];
        let ciphertext = &snapshot[RESTART_SNAPSHOT_HEADER_BYTES..];
        let mut key = derive_snapshot_key(identity)?;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
        let decrypted = cipher.decrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: ciphertext,
                aad: header,
            },
        );
        key.zeroize();
        let mut plaintext = decrypted
            .map_err(|_| BlindVaultReplicaWorkflowError::RestartSnapshotAuthenticationFailed)?;
        let body = snapshot_options()
            .deserialize::<RestartSnapshotBodyV1>(&plaintext)
            .map_err(|_| BlindVaultReplicaWorkflowError::RestartSnapshotMalformed);
        plaintext.zeroize();
        let body = body?;
        let snapshot_sequence = body.snapshot_sequence;
        if snapshot_sequence == 0 {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotSequenceInvalid);
        }
        if snapshot_sequence < minimum_snapshot_sequence {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotRollbackDetected);
        }
        let execution = body.into_execution();
        execution.validate_restored_state(restored_at_ms)?;
        Ok(BlindVaultReplicaRestoredExecution {
            execution,
            snapshot_sequence,
        })
    }

    fn validate_restored_state(
        &self,
        restored_at_ms: u64,
    ) -> Result<(), BlindVaultReplicaWorkflowError> {
        if restored_at_ms == 0
            || self.created_at_ms == 0
            || restored_at_ms < self.created_at_ms
            || self.workflow_id == [0; 16]
            || self.items.len() > MAX_BLIND_VAULT_REPLICA_WORK_ITEMS
        {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotStateInvalid);
        }
        self.policy
            .validate()
            .map_err(|_| BlindVaultReplicaWorkflowError::RestartSnapshotStateInvalid)?;
        if self.maximum_in_flight == 0
            || usize::from(self.maximum_in_flight) > MAX_BLIND_VAULT_REPLICA_WORK_ITEMS
        {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotStateInvalid);
        }

        let plan = BlindVaultReplicaPlan {
            health: self.source_plan_health,
            configured_replicas: self.source_configured_replicas,
            live_verified_replicas: self.source_live_verified_replicas,
            live_matching_replicas: self.source_live_matching_replicas,
            actions: self.items.iter().map(|item| item.action).collect(),
        };
        plan.validate_shape()
            .map_err(|_| BlindVaultReplicaWorkflowError::RestartSnapshotStateInvalid)?;

        for (index, item) in self.items.iter().enumerate() {
            if item.id.workflow_id != self.workflow_id
                || usize::from(item.id.sequence) != index
                || !valid_restored_work_state(
                    item.state,
                    self.created_at_ms,
                    restored_at_ms,
                    self.policy,
                )
            {
                return Err(BlindVaultReplicaWorkflowError::RestartSnapshotStateInvalid);
            }
        }
        if self.in_flight_count() > usize::from(self.maximum_in_flight)
            || !restored_target_dependencies_are_valid(&self.items)
        {
            return Err(BlindVaultReplicaWorkflowError::RestartSnapshotStateInvalid);
        }
        Ok(())
    }
}

impl RestartSnapshotBodyV1 {
    fn from_execution(execution: &BlindVaultReplicaExecution, snapshot_sequence: u64) -> Self {
        Self {
            snapshot_sequence,
            workflow_id: execution.workflow_id,
            created_at_ms: execution.created_at_ms,
            source_plan_health: execution.source_plan_health,
            source_configured_replicas: execution.source_configured_replicas,
            source_live_verified_replicas: execution.source_live_verified_replicas,
            source_live_matching_replicas: execution.source_live_matching_replicas,
            maximum_attempts: execution.policy.maximum_attempts,
            evidence_timeout_ms: execution.policy.evidence_timeout_ms,
            maximum_in_flight: execution.maximum_in_flight,
            items: execution
                .items
                .iter()
                .copied()
                .map(WorkItemV1::from)
                .collect(),
        }
    }

    fn into_execution(self) -> BlindVaultReplicaExecution {
        let workflow_id = self.workflow_id;
        BlindVaultReplicaExecution {
            workflow_id,
            created_at_ms: self.created_at_ms,
            source_plan_health: self.source_plan_health,
            source_configured_replicas: self.source_configured_replicas,
            source_live_verified_replicas: self.source_live_verified_replicas,
            source_live_matching_replicas: self.source_live_matching_replicas,
            policy: BlindVaultReplicaExecutionPolicy {
                maximum_attempts: self.maximum_attempts,
                evidence_timeout_ms: self.evidence_timeout_ms,
            },
            maximum_in_flight: self.maximum_in_flight,
            items: self
                .items
                .into_iter()
                .map(|item| item.into_work_item(workflow_id))
                .collect(),
        }
    }
}

impl WorkItemV1 {
    fn into_work_item(self, workflow_id: [u8; 16]) -> BlindVaultReplicaWorkItem {
        BlindVaultReplicaWorkItem {
            id: BlindVaultReplicaWorkId {
                workflow_id,
                sequence: self.sequence,
            },
            action: self.action,
            state: self.state,
        }
    }
}

impl From<BlindVaultReplicaWorkItem> for WorkItemV1 {
    fn from(value: BlindVaultReplicaWorkItem) -> Self {
        Self {
            sequence: value.id.sequence,
            action: value.action,
            state: value.state,
        }
    }
}

fn valid_restored_work_state(
    state: BlindVaultReplicaWorkState,
    created_at_ms: u64,
    restored_at_ms: u64,
    policy: BlindVaultReplicaExecutionPolicy,
) -> bool {
    let valid_attempt = |attempt: u8| attempt > 0 && attempt <= policy.maximum_attempts;
    match state {
        BlindVaultReplicaWorkState::AwaitingAuthorization => true,
        BlindVaultReplicaWorkState::Authorized { authorized_at_ms } => {
            (created_at_ms..=restored_at_ms).contains(&authorized_at_ms)
        }
        BlindVaultReplicaWorkState::AwaitingEvidence {
            attempt,
            dispatched_at_ms,
            evidence_deadline_ms,
        } => {
            valid_attempt(attempt)
                && (created_at_ms..=restored_at_ms).contains(&dispatched_at_ms)
                && dispatched_at_ms.checked_add(policy.evidence_timeout_ms)
                    == Some(evidence_deadline_ms)
        }
        BlindVaultReplicaWorkState::EvidenceAccepted {
            attempt,
            verified_at_ms,
        } => valid_attempt(attempt) && (created_at_ms..=restored_at_ms).contains(&verified_at_ms),
        BlindVaultReplicaWorkState::RetryableFailure {
            attempt,
            failed_at_ms,
            retry_not_before_ms,
            failure,
        } => {
            attempt > 0
                && attempt < policy.maximum_attempts
                && (created_at_ms..=restored_at_ms).contains(&failed_at_ms)
                && retry_not_before_ms >= failed_at_ms
                && failure.is_retryable()
        }
        BlindVaultReplicaWorkState::PermanentFailure {
            attempt,
            failed_at_ms,
            failure,
        } => {
            valid_attempt(attempt)
                && (created_at_ms..=restored_at_ms).contains(&failed_at_ms)
                && !failure.is_retryable()
        }
        BlindVaultReplicaWorkState::Exhausted {
            attempt,
            failed_at_ms,
            failure,
        } => {
            attempt == policy.maximum_attempts
                && (created_at_ms..=restored_at_ms).contains(&failed_at_ms)
                && failure.is_retryable()
        }
        BlindVaultReplicaWorkState::Cancelled { cancelled_at_ms } => {
            (created_at_ms..=restored_at_ms).contains(&cancelled_at_ms)
        }
    }
}

fn restored_target_dependencies_are_valid(items: &[BlindVaultReplicaWorkItem]) -> bool {
    for (index, item) in items.iter().enumerate() {
        let Some(target) = item.action.target() else {
            continue;
        };
        let is_dispatched_or_terminal = !matches!(
            item.state,
            BlindVaultReplicaWorkState::AwaitingAuthorization
                | BlindVaultReplicaWorkState::Authorized { .. }
        );
        if is_dispatched_or_terminal
            && items[..index].iter().any(|prior| {
                prior.action.target() == Some(target)
                    && !matches!(
                        prior.state,
                        BlindVaultReplicaWorkState::EvidenceAccepted { .. }
                    )
            })
        {
            return false;
        }
        if matches!(
            item.state,
            BlindVaultReplicaWorkState::AwaitingEvidence { .. }
        ) && items[index + 1..].iter().any(|other| {
            other.action.target() == Some(target)
                && matches!(
                    other.state,
                    BlindVaultReplicaWorkState::AwaitingEvidence { .. }
                )
        }) {
            return false;
        }
    }
    true
}

fn derive_snapshot_key(
    identity: &IdentityKeyPair,
) -> Result<[u8; 32], BlindVaultReplicaWorkflowError> {
    let mut identity_secret = identity.to_bytes();
    let hkdf = Hkdf::<Sha256>::new(Some(RESTART_SNAPSHOT_KEY_SALT), &identity_secret);
    identity_secret.zeroize();
    let mut key = [0u8; 32];
    let mut info = Vec::with_capacity(RESTART_SNAPSHOT_KEY_INFO.len() + 32);
    info.extend_from_slice(RESTART_SNAPSHOT_KEY_INFO);
    info.extend_from_slice(&identity.public_key_bytes());
    if hkdf.expand(&info, &mut key).is_err() {
        key.zeroize();
        info.zeroize();
        return Err(BlindVaultReplicaWorkflowError::RestartSnapshotAuthenticationFailed);
    }
    info.zeroize();
    Ok(key)
}

fn snapshot_header(nonce: [u8; 24]) -> Vec<u8> {
    let mut header = Vec::with_capacity(RESTART_SNAPSHOT_HEADER_BYTES);
    header.extend_from_slice(&RESTART_SNAPSHOT_MAGIC);
    header.extend_from_slice(&RESTART_SNAPSHOT_VERSION_V1.to_be_bytes());
    header.extend_from_slice(&nonce);
    header
}

fn snapshot_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES as u64)
        .reject_trailing_bytes()
}
