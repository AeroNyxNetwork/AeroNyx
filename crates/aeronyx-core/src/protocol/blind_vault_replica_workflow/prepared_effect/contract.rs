// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       prepared_effect/contract.rs
// ============================================
//! Ordered dispatch-contract validation for prepared terminal effects.
//!
//! ## Creation Reason
//! Compound replica actions have different stage machines. Keeping those
//! validators outside the effect model makes their security rules reviewable.
//!
//! ## Main Functionality
//! - Validates exact single-request contracts.
//! - Enforces write-before-delete reconciliation.
//! - Enforces admit/populate/verify/retire replacement ordering.
//! - Enforces one admit/populate/verify group per provisioned replica.
//!
//! ## Dependencies
//! - Parent `prepared_effect` domain objects.
//! - `BlindVaultReplicaDispatchContract`: immutable action requirements.
//!
//! ## Main Logical Flow
//! 1. Select the validator for the immutable dispatch contract.
//! 2. Walk the ordered effect purposes without inspecting payload bytes.
//! 3. Reject missing, reordered, duplicated, or trailing stages.
//!
//! ## Important Note For The Next Developer
//! - Zero writes are valid for an already-complete private manifest.
//! - Verification is mandatory even when no object mutation is required.
//! - Never treat stages as an unordered set.
//!
//! Last Modified: v1.0.0-PreparedEffectContracts - Initial split from the
//! prepared-effect domain model.
//! ============================================

use super::{BlindVaultReplicaPreparedEffectError, BlindVaultReplicaTerminalEffect};
use crate::protocol::blind_vault_replica_workflow::BlindVaultReplicaDispatchContract;
use crate::protocol::onion::OnionRoutePurpose;

/// [BLIND-VAULT-PREPARED-EFFECT-CONTRACT 2026-08-29 by Codex] Every branch
/// accepts one canonical order only, preventing adapter-specific interpretations.
pub(super) fn validate_effect_contract(
    contract: BlindVaultReplicaDispatchContract,
    effects: &[BlindVaultReplicaTerminalEffect],
) -> Result<(), BlindVaultReplicaPreparedEffectError> {
    let valid = match contract {
        BlindVaultReplicaDispatchContract::SingleTerminalRequest { purpose } => {
            effects.len() == 1 && effects[0].purpose == purpose
        }
        BlindVaultReplicaDispatchContract::ReconcileInventory {
            write_purpose,
            delete_purpose,
            verification_purpose,
        } => validate_reconcile(effects, write_purpose, delete_purpose, verification_purpose),
        BlindVaultReplicaDispatchContract::ReplaceReplica {
            admission_purpose,
            write_purpose,
            verification_purpose,
            retirement_purpose,
        } => validate_replace(
            effects,
            admission_purpose,
            write_purpose,
            verification_purpose,
            retirement_purpose,
        ),
        BlindVaultReplicaDispatchContract::ProvisionReplicas {
            admission_purpose,
            write_purpose,
            verification_purpose,
            count,
        } => validate_provision(
            effects,
            admission_purpose,
            write_purpose,
            verification_purpose,
            count,
        ),
    };
    valid
        .then_some(())
        .ok_or(BlindVaultReplicaPreparedEffectError::ContractMismatch)
}

fn validate_reconcile(
    effects: &[BlindVaultReplicaTerminalEffect],
    write: OnionRoutePurpose,
    delete: OnionRoutePurpose,
    verify: OnionRoutePurpose,
) -> bool {
    let Some((last, mutations)) = effects.split_last() else {
        return false;
    };
    if last.purpose != verify {
        return false;
    }
    let mut deleting = false;
    mutations.iter().all(|effect| {
        if effect.purpose == delete {
            deleting = true;
            true
        } else {
            !deleting && effect.purpose == write
        }
    })
}

fn validate_replace(
    effects: &[BlindVaultReplicaTerminalEffect],
    admit: OnionRoutePurpose,
    write: OnionRoutePurpose,
    verify: OnionRoutePurpose,
    retire: OnionRoutePurpose,
) -> bool {
    effects.len() >= 3
        && effects
            .first()
            .is_some_and(|effect| effect.purpose == admit)
        && effects
            .get(effects.len() - 2)
            .is_some_and(|effect| effect.purpose == verify)
        && effects
            .last()
            .is_some_and(|effect| effect.purpose == retire)
        && effects[1..effects.len() - 2]
            .iter()
            .all(|effect| effect.purpose == write)
}

fn validate_provision(
    effects: &[BlindVaultReplicaTerminalEffect],
    admit: OnionRoutePurpose,
    write: OnionRoutePurpose,
    verify: OnionRoutePurpose,
    count: u8,
) -> bool {
    let mut cursor = 0usize;
    for _ in 0..count {
        if !matches!(effects.get(cursor), Some(effect) if effect.purpose == admit) {
            return false;
        }
        cursor += 1;
        while effects
            .get(cursor)
            .is_some_and(|effect| effect.purpose == write)
        {
            cursor += 1;
        }
        if !matches!(effects.get(cursor), Some(effect) if effect.purpose == verify) {
            return false;
        }
        cursor += 1;
    }
    cursor == effects.len()
}
