// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/prepared_effect.rs
// ============================================
//! Payload-blind binding for one prepared replica workflow attempt.
//!
//! ## Creation Reason
//! A durable attempt journal proves ordering, but an adapter could otherwise
//! persist one request and send different bytes after receiving its permit.
//! The source therefore commits the exact ordered terminal effects before any
//! journal or network transition is allowed.
//!
//! ## Main Functionality
//! - Commits each terminal purpose, payload length, and opaque payload bytes.
//! - Validates compound effects against the immutable action contract.
//! - Binds the ordered set to workflow, work item, attempt, and evidence window.
//! - Matches send-time bytes without retaining plaintext or ciphertext payloads.
//! - Redacts commitments from Debug output and zeroizes them on drop.
//!
//! ## Dependencies
//! - `execution.rs`: side-effect-free attempt readiness.
//! - `BlindVaultReplicaDispatchContract`: required terminal stage ordering.
//! - `OnionRoutePurpose`: canonical purpose negotiation.
//! - `prepared_effect/commitment.rs`: domain-separated commitments.
//! - `prepared_effect/contract.rs`: compound action stage validation.
//!
//! ## Main Logical Flow
//! 1. The source constructs every encrypted terminal payload for one attempt.
//! 2. This module validates their ordered purposes against the planned action.
//! 3. It commits payload bytes and exact workflow timing without retaining data.
//! 4. Later durability and transport layers verify the same ordered effects.
//!
//! ## Important Note For The Next Developer
//! - This is source-local authorization state, never a wire or ledger payload.
//! - Never store payload bytes, routes, endpoints, identities, or credentials.
//! - Route identity is deliberately excluded so a failed path can be replaced.
//! - A network adapter must match each payload immediately before sending it.
//! - Do not weaken ordered contract validation into an unordered purpose set.
//!
//! Last Modified: v1.0.0-PreparedTerminalEffects - Initial exact payload-blind
//! binding and compound-action stage validation.
//! ============================================

use std::fmt;

use thiserror::Error;
use zeroize::Zeroize;

use self::commitment::{terminal_effect_payload_commitment, terminal_effect_set_commitment};
use self::contract::validate_effect_contract;
use super::{
    BlindVaultReplicaDispatchReadiness, BlindVaultReplicaExecution, BlindVaultReplicaWorkId,
    BlindVaultReplicaWorkflowError,
};
use crate::protocol::onion::OnionRoutePurpose;

mod commitment;
mod contract;

/// Maximum opaque terminal payload accepted into one effect commitment.
pub const MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECT_BYTES: usize = 256 * 1024;

/// Maximum ordered terminal effects bound to one workflow attempt.
pub const MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECTS: usize = 64;

/// One payload-blind terminal effect prepared by the source.
///
/// [BLIND-VAULT-PREPARED-EFFECT 2026-08-29 by Codex] The commitment binds the
/// canonical purpose as well as bytes so an adapter cannot relabel an otherwise
/// identical encrypted payload for a more privileged terminal operation.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaTerminalEffect {
    purpose: OnionRoutePurpose,
    payload_length: u32,
    payload_commitment: [u8; 32],
}

impl BlindVaultReplicaTerminalEffect {
    fn commit(
        purpose: OnionRoutePurpose,
        payload: &[u8],
    ) -> Result<Self, BlindVaultReplicaPreparedEffectError> {
        if payload.is_empty() {
            return Err(BlindVaultReplicaPreparedEffectError::EmptyPayload);
        }
        if payload.len() > MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECT_BYTES {
            return Err(BlindVaultReplicaPreparedEffectError::PayloadTooLarge {
                actual: payload.len(),
            });
        }
        let payload_length = u32::try_from(payload.len()).map_err(|_| {
            BlindVaultReplicaPreparedEffectError::PayloadTooLarge {
                actual: payload.len(),
            }
        })?;
        Ok(Self {
            purpose,
            payload_length,
            payload_commitment: terminal_effect_payload_commitment(
                purpose,
                payload_length,
                payload,
            ),
        })
    }

    /// Canonical terminal purpose bound to this payload.
    #[must_use]
    pub const fn purpose(&self) -> OnionRoutePurpose {
        self.purpose
    }

    /// Exact committed payload length.
    #[must_use]
    pub const fn payload_length(&self) -> u32 {
        self.payload_length
    }

    /// Verifies send-time bytes against this exact prepared effect.
    #[must_use]
    pub fn matches_payload(&self, purpose: OnionRoutePurpose, payload: &[u8]) -> bool {
        if purpose != self.purpose || payload.len() != self.payload_length as usize {
            return false;
        }
        terminal_effect_payload_commitment(purpose, self.payload_length, payload)
            == self.payload_commitment
    }
}

impl fmt::Debug for BlindVaultReplicaTerminalEffect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaTerminalEffect")
            .field("purpose", &self.purpose)
            .field("payload_length", &self.payload_length)
            .field("payload_commitment", &"[REDACTED]")
            .finish()
    }
}

impl Drop for BlindVaultReplicaTerminalEffect {
    fn drop(&mut self) {
        self.payload_commitment.zeroize();
    }
}

/// Ordered effects authorized for one exact dispatch-ready attempt.
pub struct BlindVaultReplicaPreparedEffectSet {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    planned_dispatch_at_ms: u64,
    evidence_deadline_ms: u64,
    effects: Vec<BlindVaultReplicaTerminalEffect>,
    commitment: [u8; 32],
}

impl BlindVaultReplicaPreparedEffectSet {
    /// Exact source-local work item bound to this set.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    /// Exact bounded attempt predicted by side-effect-free readiness.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    /// Source timestamp that must later be used for the dispatch transition.
    #[must_use]
    pub const fn planned_dispatch_at_ms(&self) -> u64 {
        self.planned_dispatch_at_ms
    }

    /// Exact evidence deadline derived by the workflow policy.
    #[must_use]
    pub const fn evidence_deadline_ms(&self) -> u64 {
        self.evidence_deadline_ms
    }

    /// Ordered, payload-blind effects for transport orchestration.
    #[must_use]
    pub fn effects(&self) -> &[BlindVaultReplicaTerminalEffect] {
        &self.effects
    }

    /// Verifies one indexed send-time payload without exposing commitments.
    #[must_use]
    pub fn matches_payload(
        &self,
        index: usize,
        purpose: OnionRoutePurpose,
        payload: &[u8],
    ) -> bool {
        self.effects
            .get(index)
            .is_some_and(|effect| effect.matches_payload(purpose, payload))
    }
}

impl fmt::Debug for BlindVaultReplicaPreparedEffectSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaPreparedEffectSet")
            .field("attempt", &self.attempt)
            .field("planned_dispatch_at_ms", &self.planned_dispatch_at_ms)
            .field("evidence_deadline_ms", &self.evidence_deadline_ms)
            .field("effect_count", &self.effects.len())
            .field("commitment", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl Drop for BlindVaultReplicaPreparedEffectSet {
    fn drop(&mut self) {
        self.commitment.zeroize();
    }
}

/// Fail-closed prepared-effect construction errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaPreparedEffectError {
    /// Workflow readiness or work identity validation failed.
    #[error(transparent)]
    Workflow(#[from] BlindVaultReplicaWorkflowError),
    /// The exact work item is not dispatch-ready at the supplied timestamp.
    #[error("blind vault replica terminal effects were prepared while dispatch was not ready")]
    DispatchNotReady {
        readiness: BlindVaultReplicaDispatchReadiness,
    },
    /// At least one terminal effect is required for every action contract.
    #[error("blind vault replica terminal effect set is empty")]
    EmptyEffectSet,
    /// The effect count exceeded the bounded attempt contract.
    #[error("blind vault replica terminal effect set is too large: {actual}")]
    TooManyEffects { actual: usize },
    /// Empty terminal payloads cannot represent a complete encrypted request.
    #[error("blind vault replica terminal effect payload is empty")]
    EmptyPayload,
    /// One opaque terminal payload exceeded the bounded onion carrier.
    #[error("blind vault replica terminal effect payload is too large: {actual}")]
    PayloadTooLarge { actual: usize },
    /// Ordered effects did not implement the immutable action contract.
    #[error("blind vault replica terminal effects violate the dispatch contract")]
    ContractMismatch,
}

impl BlindVaultReplicaExecution {
    /// Validates and commits the exact terminal effects for one ready attempt.
    ///
    /// [BLIND-VAULT-PREPARED-EFFECT-SET 2026-08-29 by Codex] This operation is
    /// side-effect free. It does not dispatch, persist, retain payload bytes,
    /// or authorize a route; it only creates the prerequisite binding.
    pub fn prepare_terminal_effects<'payload, Effects>(
        &self,
        work_id: BlindVaultReplicaWorkId,
        planned_dispatch_at_ms: u64,
        effects: Effects,
    ) -> Result<BlindVaultReplicaPreparedEffectSet, BlindVaultReplicaPreparedEffectError>
    where
        Effects: IntoIterator<Item = (OnionRoutePurpose, &'payload [u8])>,
    {
        let readiness = self.dispatch_readiness(work_id, planned_dispatch_at_ms)?;
        let (attempt, evidence_deadline_ms) = match readiness {
            BlindVaultReplicaDispatchReadiness::Ready {
                attempt,
                evidence_deadline_ms,
            } => (attempt, evidence_deadline_ms),
            readiness => {
                return Err(BlindVaultReplicaPreparedEffectError::DispatchNotReady { readiness })
            }
        };
        let work_item = self
            .items()
            .iter()
            .find(|item| item.id() == work_id)
            .ok_or(BlindVaultReplicaWorkflowError::WorkItemNotFound)?;
        let effects = effects
            .into_iter()
            .map(|(purpose, payload)| BlindVaultReplicaTerminalEffect::commit(purpose, payload))
            .collect::<Result<Vec<_>, _>>()?;
        if effects.is_empty() {
            return Err(BlindVaultReplicaPreparedEffectError::EmptyEffectSet);
        }
        if effects.len() > MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECTS {
            return Err(BlindVaultReplicaPreparedEffectError::TooManyEffects {
                actual: effects.len(),
            });
        }
        validate_effect_contract(work_item.dispatch_contract(), &effects)?;
        let commitment = terminal_effect_set_commitment(
            work_id,
            attempt,
            planned_dispatch_at_ms,
            evidence_deadline_ms,
            &effects,
        );
        Ok(BlindVaultReplicaPreparedEffectSet {
            work_id,
            attempt,
            planned_dispatch_at_ms,
            evidence_deadline_ms,
            effects,
            commitment,
        })
    }
}
