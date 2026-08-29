// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       prepared_effect/commitment.rs
// ============================================
//! Domain-separated commitments for prepared terminal effects.
//!
//! ## Creation Reason
//! Hash transcript details are security-sensitive and should be independently
//! reviewable without mixing them into workflow orchestration.
//!
//! ## Main Functionality
//! - Commits canonical purpose, payload length, and opaque payload bytes.
//! - Commits exact workflow attempt timing and ordered effect commitments.
//!
//! ## Dependencies
//! - Parent prepared-effect domain objects.
//! - SHA-256 from `sha2`.
//!
//! ## Main Logical Flow
//! 1. Prefix every transcript with a versioned domain.
//! 2. Encode fixed-width integers in network byte order.
//! 3. Include canonical purpose bytes before payload or child commitment.
//!
//! ## Important Note For The Next Developer
//! - Changing a domain or transcript is a persistence compatibility change.
//! - Never omit order, purpose, length, work identity, attempt, or deadline.
//!
//! Last Modified: v1.0.0-PreparedEffectCommitments - Initial split from the
//! prepared-effect domain model.
//! ============================================

use sha2::{Digest, Sha256};

use super::BlindVaultReplicaTerminalEffect;
use crate::protocol::blind_vault_replica_workflow::BlindVaultReplicaWorkId;
use crate::protocol::onion::OnionRoutePurpose;

const PAYLOAD_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Replica-Terminal-Effect-Payload-v1";
const SET_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Replica-Terminal-Effect-Set-v1";

/// [BLIND-VAULT-PREPARED-EFFECT-COMMITMENT 2026-08-29 by Codex] Purpose and
/// length are explicit transcript fields rather than implicit adapter context.
pub(super) fn terminal_effect_payload_commitment(
    purpose: OnionRoutePurpose,
    payload_length: u32,
    payload: &[u8],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PAYLOAD_DOMAIN);
    update_purpose(&mut hasher, purpose);
    hasher.update(payload_length.to_be_bytes());
    hasher.update(payload);
    hasher.finalize().into()
}

pub(super) fn terminal_effect_set_commitment(
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    planned_dispatch_at_ms: u64,
    evidence_deadline_ms: u64,
    effects: &[BlindVaultReplicaTerminalEffect],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SET_DOMAIN);
    hasher.update(work_id.workflow_id());
    hasher.update(work_id.sequence().to_be_bytes());
    hasher.update([attempt]);
    hasher.update(planned_dispatch_at_ms.to_be_bytes());
    hasher.update(evidence_deadline_ms.to_be_bytes());
    hasher.update((effects.len() as u16).to_be_bytes());
    for effect in effects {
        update_purpose(&mut hasher, effect.purpose);
        hasher.update(effect.payload_length.to_be_bytes());
        hasher.update(effect.payload_commitment);
    }
    hasher.finalize().into()
}

fn update_purpose(hasher: &mut Sha256, purpose: OnionRoutePurpose) {
    let value = purpose.as_str().as_bytes();
    hasher.update((value.len() as u16).to_be_bytes());
    hasher.update(value);
}
