// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       prepared_effect/restart_codec.rs
// ============================================
//! Bounded restart codec for payload-blind terminal effect bindings.
//!
//! ## Creation Reason
//! After an ambiguous restart, a total hash alone cannot verify each rebuilt
//! send. The sealed attempt journal must recover ordered purpose, length, and
//! payload commitment while still retaining no payload bytes.
//!
//! ## Main Functionality
//! - Encodes a stable V1 source-local effect summary.
//! - Parses explicit purpose codes and bounded fixed-size entries.
//! - Revalidates the action contract and complete attempt commitment.
//! - Rejects unknown versions, purposes, lengths, counts, and trailing bytes.
//!
//! ## Dependencies
//! - Parent prepared-effect domain model.
//! - `commitment.rs`: canonical complete-set commitment.
//! - `contract.rs`: immutable action stage validation.
//!
//! ## Main Logical Flow
//! 1. Encode one fixed header plus one fixed-size record per effect.
//! 2. Restore records only against journal-authenticated attempt metadata.
//! 3. Revalidate ordered dispatch semantics.
//! 4. Recompute and compare the complete effect-set commitment.
//!
//! ## Important Note For The Next Developer
//! - This format belongs only inside the identity-sealed attempt journal.
//! - Purpose codes are stable persistence values, not Rust enum ordinals.
//! - Do not add routes, endpoints, payloads, account identity, or contacts.
//!
//! Last Modified: v1.1.0-WorkflowVisibilityBoundary - Exposed restart binding
//! helpers only to the enclosing replica workflow for sibling recovery paths.
//! v1.0.0-PreparedEffectRestartCodec - Initial bounded V1
//! source-local encoding and fail-closed restoration.
//! ============================================

use super::commitment::terminal_effect_set_commitment;
use super::contract::validate_effect_contract;
use super::{
    BlindVaultReplicaPreparedEffectError, BlindVaultReplicaPreparedEffectSet,
    BlindVaultReplicaTerminalEffect, MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECTS,
    MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECT_BYTES,
};
use crate::protocol::blind_vault_replica_workflow::{
    BlindVaultReplicaDispatchContract, BlindVaultReplicaWorkId,
};
use crate::protocol::onion::OnionRoutePurpose;

const RESTART_BINDING_MAGIC: [u8; 4] = *b"AXPE";
const RESTART_BINDING_VERSION_V1: u16 = 1;
const RESTART_BINDING_HEADER_BYTES: usize = 4 + 2 + 2 + 32;
const RESTART_BINDING_EFFECT_BYTES: usize = 1 + 4 + 32;

impl BlindVaultReplicaPreparedEffectSet {
    /// Encodes only payload-blind state for an identity-sealed local journal.
    // [CORE-BUILD-BOUNDARY 2026-08-31 by Codex] Recovery orchestration lives
    // in workflow sibling modules; expose this capability only to that parent
    // domain, never to the crate or public API.
    pub(in crate::protocol::blind_vault_replica_workflow) fn encode_restart_binding(
        &self,
    ) -> Vec<u8> {
        let mut encoded = Vec::with_capacity(
            RESTART_BINDING_HEADER_BYTES.saturating_add(
                self.effects
                    .len()
                    .saturating_mul(RESTART_BINDING_EFFECT_BYTES),
            ),
        );
        encoded.extend_from_slice(&RESTART_BINDING_MAGIC);
        encoded.extend_from_slice(&RESTART_BINDING_VERSION_V1.to_be_bytes());
        encoded.extend_from_slice(&(self.effects.len() as u16).to_be_bytes());
        encoded.extend_from_slice(&self.commitment);
        for effect in &self.effects {
            encoded.push(purpose_code(effect.purpose));
            encoded.extend_from_slice(&effect.payload_length.to_be_bytes());
            encoded.extend_from_slice(&effect.payload_commitment);
        }
        encoded
    }

    /// Restores a binding only against authenticated journal attempt metadata.
    ///
    /// [BLIND-VAULT-PREPARED-EFFECT-RESTORE 2026-08-29 by Codex] Both ordered
    /// contract validation and the complete transcript commitment run before
    /// the caller receives a send-time matching capability.
    pub(in crate::protocol::blind_vault_replica_workflow) fn decode_restart_binding(
        work_id: BlindVaultReplicaWorkId,
        attempt: u8,
        planned_dispatch_at_ms: u64,
        evidence_deadline_ms: u64,
        contract: BlindVaultReplicaDispatchContract,
        encoded: &[u8],
    ) -> Result<Self, BlindVaultReplicaPreparedEffectError> {
        if encoded.len() < RESTART_BINDING_HEADER_BYTES || encoded[..4] != RESTART_BINDING_MAGIC {
            return Err(BlindVaultReplicaPreparedEffectError::RestartBindingMalformed);
        }
        if u16::from_be_bytes([encoded[4], encoded[5]]) != RESTART_BINDING_VERSION_V1 {
            return Err(BlindVaultReplicaPreparedEffectError::RestartBindingVersionUnsupported);
        }
        let effect_count = usize::from(u16::from_be_bytes([encoded[6], encoded[7]]));
        if effect_count == 0 || effect_count > MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECTS {
            return Err(BlindVaultReplicaPreparedEffectError::RestartBindingMalformed);
        }
        let expected_length = RESTART_BINDING_HEADER_BYTES
            .checked_add(effect_count.saturating_mul(RESTART_BINDING_EFFECT_BYTES))
            .ok_or(BlindVaultReplicaPreparedEffectError::RestartBindingMalformed)?;
        if encoded.len() != expected_length {
            return Err(BlindVaultReplicaPreparedEffectError::RestartBindingMalformed);
        }

        let mut persisted_commitment = [0u8; 32];
        persisted_commitment.copy_from_slice(&encoded[8..RESTART_BINDING_HEADER_BYTES]);
        let mut effects = Vec::with_capacity(effect_count);
        let mut offset = RESTART_BINDING_HEADER_BYTES;
        for _ in 0..effect_count {
            let purpose = purpose_from_code(encoded[offset])
                .ok_or(BlindVaultReplicaPreparedEffectError::RestartBindingMalformed)?;
            offset += 1;
            let payload_length = u32::from_be_bytes(
                encoded[offset..offset + 4]
                    .try_into()
                    .map_err(|_| BlindVaultReplicaPreparedEffectError::RestartBindingMalformed)?,
            );
            offset += 4;
            if payload_length == 0
                || usize::try_from(payload_length).map_or(true, |length| {
                    length > MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECT_BYTES
                })
            {
                return Err(BlindVaultReplicaPreparedEffectError::RestartBindingMalformed);
            }
            let mut payload_commitment = [0u8; 32];
            payload_commitment.copy_from_slice(&encoded[offset..offset + 32]);
            offset += 32;
            effects.push(BlindVaultReplicaTerminalEffect {
                purpose,
                payload_length,
                payload_commitment,
            });
        }

        validate_effect_contract(contract, &effects)?;
        let commitment = terminal_effect_set_commitment(
            work_id,
            attempt,
            planned_dispatch_at_ms,
            evidence_deadline_ms,
            &effects,
        );
        if commitment != persisted_commitment {
            return Err(BlindVaultReplicaPreparedEffectError::RestartBindingCommitmentMismatch);
        }
        Ok(Self {
            work_id,
            attempt,
            planned_dispatch_at_ms,
            evidence_deadline_ms,
            effects,
            commitment,
        })
    }

    pub(in crate::protocol::blind_vault_replica_workflow) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

fn purpose_code(purpose: OnionRoutePurpose) -> u8 {
    match purpose {
        OnionRoutePurpose::MessageRelay => 1,
        OnionRoutePurpose::BlindVaultPut => 2,
        OnionRoutePurpose::BlindVaultPull => 3,
        OnionRoutePurpose::BlindVaultDelete => 4,
        OnionRoutePurpose::BlindVaultLeaseAdmission => 5,
        OnionRoutePurpose::BlindVaultPutReceipt => 6,
        OnionRoutePurpose::BlindVaultLeaseRetire => 7,
        OnionRoutePurpose::BlindVaultLeaseRenewal => 8,
        OnionRoutePurpose::BlindVaultLeaseStatus => 9,
        OnionRoutePurpose::BlindVaultLeaseInventory => 10,
    }
}

fn purpose_from_code(code: u8) -> Option<OnionRoutePurpose> {
    match code {
        1 => Some(OnionRoutePurpose::MessageRelay),
        2 => Some(OnionRoutePurpose::BlindVaultPut),
        3 => Some(OnionRoutePurpose::BlindVaultPull),
        4 => Some(OnionRoutePurpose::BlindVaultDelete),
        5 => Some(OnionRoutePurpose::BlindVaultLeaseAdmission),
        6 => Some(OnionRoutePurpose::BlindVaultPutReceipt),
        7 => Some(OnionRoutePurpose::BlindVaultLeaseRetire),
        8 => Some(OnionRoutePurpose::BlindVaultLeaseRenewal),
        9 => Some(OnionRoutePurpose::BlindVaultLeaseStatus),
        10 => Some(OnionRoutePurpose::BlindVaultLeaseInventory),
        _ => None,
    }
}
